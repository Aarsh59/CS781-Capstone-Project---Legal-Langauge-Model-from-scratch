"""PCR data: load queries + candidate pools, ByteBPE encoding (cached), chunk selection, BM25, Micro-F1@k
(plumbing; declare it).

Files (spec 13.3): Phase_7_PCR/{Train/train.json + train_candidates.json, Dev/dev.json + dev_candidates.json,
Test/test.json + test_candidates.json}. Each split's queries are ranked ONLY against that split's own pool.
The loader accepts a list of records or a dict keyed by case id; the id field may be called id / case_id / query_id /
candidate_id / doc_id; the text field (text / document / sentences / content) may be a list of sentences (joined with
one space, like the whitespace-normalised Phase-1 data) or a string. Relevance: "relevant_candidates" (spec), also
accepted: relevant / candidates / labels. Test relevance is never read (load_split drops it).

Chunking: same functions as Phases 4-5 (even | headtail | overlap windows of L tokens, at most k per document).
Metric (spec 13.4): Micro-P/R/F1@7 = hits summed over queries / (7 x queries) and / (gold relevant summed over
queries); the gold set is complete (never truncated), so recall counts relevant cases that are not in the top 7.
"""
import hashlib
import json
import math
import os
import re
from collections import Counter

import numpy as np

SPLITS = {"train": ("Train", "train.json", "train_candidates.json"),
          "dev": ("Dev", "dev.json", "dev_candidates.json"),
          "test": ("Test", "test.json", "test_candidates.json")}
ID_KEYS = ("id", "case_id", "query_id", "candidate_id", "doc_id")
TEXT_KEYS = ("text", "document", "sentences", "content")
REL_KEYS = ("relevant_candidates", "relevant", "candidates", "labels")
TOP_K = 7


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _first(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def _text(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return " ".join(str(s).strip() for s in v if str(s).strip())
    raise TypeError(f"unexpected text type {type(v).__name__}")


def _records(obj):
    """-> list of (id, record dict) from a list of records or an {id: record | text} dict."""
    if isinstance(obj, dict):
        for wrap in ("data", "queries", "candidates", "cases"):        # {"queries": [...]} style wrappers
            if wrap in obj and isinstance(obj[wrap], (list, dict)) and len(obj) <= 3:
                return _records(obj[wrap])
        out = []
        for k, v in obj.items():
            out.append((str(k), v if isinstance(v, dict) else {"text": v}))
        return out
    assert isinstance(obj, list), f"unexpected top-level JSON type {type(obj).__name__}"
    out = []
    for r in obj:
        k = _first(r, ID_KEYS)
        assert k, f"no id field in record with keys {sorted(r)[:10]}"
        out.append((str(r[k]), r))
    return out


def _docs(obj, with_rel):
    docs = []
    for i, r in _records(obj):
        tk = _first(r, TEXT_KEYS)
        assert tk, f"record {i}: no text field (keys {sorted(r)[:10]})"
        d = {"id": i, "text": _text(r[tk])}
        if with_rel:
            rk = _first(r, REL_KEYS)
            d["relevant"] = [str(x) for x in r[rk]] if rk and r[rk] is not None else None
        docs.append(d)
    ids = [d["id"] for d in docs]
    assert len(set(ids)) == len(ids), f"{len(ids) - len(set(ids))} duplicate ids"
    return docs


def load_split(data_dir, split):
    """-> (queries [{"id", "text", "relevant" (None for test)}], pool [{"id", "text"}])."""
    sub, qf, cf = SPLITS[split]
    queries = _docs(json.load(open(os.path.join(data_dir, sub, qf), encoding="utf-8")), with_rel=True)
    pool = _docs(json.load(open(os.path.join(data_dir, sub, cf), encoding="utf-8")), with_rel=False)
    if split == "test":
        for q in queries:
            q["relevant"] = None                       # test labels are private: never read
    else:
        assert all(q["relevant"] is not None for q in queries), f"{split}: queries without relevant_candidates"
    return queries, pool


def schema_report(data_dir):
    """first-record keys / types of every file + basic counts (printed by the notebook before anything else)."""
    rep = {}
    for split, (sub, qf, cf) in SPLITS.items():
        for f in (qf, cf):
            obj = json.load(open(os.path.join(data_dir, sub, f), encoding="utf-8"))
            recs = _records(obj)
            i, r = recs[0]
            rep[f] = {"top_level": type(obj).__name__, "n": len(recs), "first_id": i,
                      "keys": {k: type(v).__name__ for k, v in list(r.items())[:8]}}
    return rep


def split_stats(queries, pool):
    pid = {d["id"] for d in pool}
    st = {"queries": len(queries), "pool": len(pool),
          "pool_words_median": int(np.median([len(d["text"].split()) for d in pool])),
          "query_words_median": int(np.median([len(d["text"].split()) for d in queries])),
          "queries_in_own_pool": sum(q["id"] in pid for q in queries)}
    if queries and queries[0]["relevant"] is not None:
        rel = [len(q["relevant"]) for q in queries]
        st.update(rel_mean=round(float(np.mean(rel)), 2), rel_median=float(np.median(rel)), rel_max=max(rel),
                  rel_min=min(rel), rel_missing_from_pool=sum(c not in pid for q in queries for c in q["relevant"]),
                  rel_total=sum(rel))
    return st


def norm_hash(text):
    return hashlib.md5(" ".join(text.split()).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# tokens + chunks (unchanged from Phase 4 / 5)
# ---------------------------------------------------------------------------

def encode_docs(tok, docs, cache_path=None):
    """-> list of np.int32 token arrays (one per doc, same order). Cached by document hashes."""
    key = hashlib.md5(("|".join(d["id"] + ":" + norm_hash(d["text"]) for d in docs)).encode()).hexdigest()
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as z:
            if str(z["key"]) == key:
                ids, off = z["ids"], z["offsets"]
                return [ids[off[i]:off[i + 1]] for i in range(len(docs))]
    out = [np.asarray(tok.encode(d["text"]), dtype=np.int32) for d in docs]
    assert all(len(a) > 0 for a in out), "a document encodes to zero tokens"
    if cache_path:
        offsets = np.concatenate([[0], np.cumsum([len(a) for a in out])]).astype(np.int64)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path + ".tmp.npz", ids=np.concatenate(out), offsets=offsets, key=key)
        os.replace(cache_path + ".tmp.npz", cache_path)
    return out


def _spread(n_windows, k):
    if n_windows <= k:
        return list(range(n_windows))
    idx = sorted({int(round(x)) for x in np.linspace(0, n_windows - 1, k)})
    assert len(idx) == k, (n_windows, k, idx)
    return idx


def chunk_spans(n, L, strategy, k, stride=None):
    """-> list of (start, end) token spans for a document of n tokens, in text order, no duplicates."""
    assert n > 0 and L > 0 and k >= 1
    if n <= L:
        return [(0, n)]
    if strategy in ("even", "headtail"):
        W = math.ceil(n / L)
        grid = [(i * L, min(n, (i + 1) * L)) for i in range(W)]
        if W <= k:
            return grid
        if strategy == "even":
            return [grid[i] for i in _spread(W, k)]
        h, t = math.ceil(k / 2), k // 2
        head = [(i * L, (i + 1) * L) for i in range(h)]
        tail = [(n - (j + 1) * L, n - j * L) for j in range(t)][::-1]
        return head + tail
    if strategy == "overlap":
        assert stride and 0 < stride <= L, "overlap needs 0 < stride <= L"
        starts = list(range(0, n - L + 1, stride))
        if starts[-1] != n - L:
            starts.append(n - L)
        wins = [(s, s + L) for s in starts]
        return [wins[i] for i in _spread(len(wins), k)]
    raise ValueError(f"unknown chunking strategy {strategy!r}: even | headtail | overlap")


def build_items(docs, token_arrays, L, strategy, k, stride=None):
    """-> list of {"id", "chunks": [np.int32 arrays], "n_tokens"}"""
    items = []
    for d, a in zip(docs, token_arrays):
        spans = chunk_spans(len(a), L, strategy, k, stride)
        chunks = [a[s:e] for s, e in spans]
        items.append({"id": d["id"], "chunks": chunks, "n_tokens": int(len(a))})
    return items


def coverage(items):
    return sum(sum(len(c) for c in it["chunks"]) for it in items) / max(1, sum(it["n_tokens"] for it in items))


# ---------------------------------------------------------------------------
# BM25 (lexical; used for hard negatives, as a local reference score and for optional score fusion)
# ---------------------------------------------------------------------------

WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return [w for w in WORD.findall(text.lower()) if len(w) > 1]


class BM25:
    """Okapi BM25 over lower-cased words (length >= 2). score(q, d) = sum over the query's DISTINCT words t of
    idf(t) * tf(t,d) (k1+1) / (tf(t,d) + k1 (1 - b + b |d| / avgdl)), idf = ln(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.ids = [d["id"] for d in docs]
        tfs = [Counter(words(d["text"])) for d in docs]
        dl = np.array([sum(c.values()) for c in tfs], dtype=np.float64)
        avg = dl.mean() if len(dl) else 1.0
        df = Counter(t for c in tfs for t in c)
        N = len(docs)
        post = {}
        for j, c in enumerate(tfs):
            norm = k1 * (1 - b + b * dl[j] / avg)
            for t, f in c.items():
                post.setdefault(t, ([], []))
                post[t][0].append(j)
                post[t][1].append(f * (k1 + 1) / (f + norm))
        self.post = {t: (np.asarray(js, dtype=np.int32),
                         np.asarray(ws, dtype=np.float32) * np.float32(math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))))
                     for t, (js, ws) in post.items()}
        self.N = N

    def scores(self, text):
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(words(text)):
            p = self.post.get(t)
            if p is not None:
                np.add.at(s, p[0], p[1])
        return s

    def score_matrix(self, queries):
        return np.stack([self.scores(q["text"]) for q in queries]) if queries else np.zeros((0, self.N), np.float32)


# ---------------------------------------------------------------------------
# ranking + metrics
# ---------------------------------------------------------------------------

def rank(S, qids, cids, k, exclude_self=True):
    """S (n_queries, n_pool) scores -> {qid: top-k candidate ids, best first}. A query's own id is never returned."""
    out = {}
    cidx = {c: j for j, c in enumerate(cids)}
    for i, q in enumerate(qids):
        row = np.asarray(S[i], dtype=np.float64).copy()
        if exclude_self and q in cidx:
            row[cidx[q]] = -np.inf
        top = np.argsort(-row, kind="stable")[:k]
        out[q] = [cids[j] for j in top]
    return out


def prf_at_k(ranked, gold, k=TOP_K):
    """micro precision / recall / F1 @k. ranked {qid: [ids]}, gold {qid: [ids]} (complete relevance sets)."""
    hits = sum(len(set(ranked[q][:k]) & set(g)) for q, g in gold.items())
    n_ret = k * len(gold)
    n_rel = sum(len(set(g)) for g in gold.values())
    p = hits / n_ret if n_ret else 0.0
    r = hits / n_rel if n_rel else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"micro_p": round(p, 4), "micro_r": round(r, 4), "micro_f1": round(f, 4), "hits": hits, "k": k}


def report(S, qids, cids, gold, ks=(1, 3, 5, 7, 10, 20)):
    """Micro-F1@7 (the metric) + P/R/F1 at other cut-offs and recall@100 for analysis."""
    deep = rank(S, qids, cids, max(max(ks), 100))
    rep = {"at7": prf_at_k(deep, gold, TOP_K)}
    rep["curve"] = {str(k): prf_at_k(deep, gold, k)["micro_f1"] for k in ks}
    rep["recall@100"] = prf_at_k(deep, gold, 100)["micro_r"]
    return rep


def write_prediction(S, qids, cids, path):
    """prediction.json (spec 13.3.5): {query id: [7 unique candidate ids from this split's pool, best first]}."""
    pred = rank(S, qids, cids, TOP_K)
    pool = set(cids)
    assert len(pred) == len(qids) and all(len(v) == TOP_K and len(set(v)) == TOP_K and set(v) <= pool
                                          for v in pred.values()), "prediction.json format check failed"
    with open(path, "w") as f:
        json.dump(pred, f, indent=1)
    return pred
