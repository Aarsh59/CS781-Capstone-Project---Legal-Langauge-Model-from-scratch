"""LSI data: load, exact-duplicate check, ByteBPE encoding (cached), chunk selection, multi-label metrics
(plumbing; declare it).

Documents: {"id", "text", "labels": [statute ids]}. The labels of a TEST document are never read: load_split("test")
drops them. (Chunking functions are unchanged from Phase 4.)

Train de-duplication (your spec, section 3): exact duplicates (same whitespace-normalised text) are reduced to one copy;
a duplicate group with conflicting label SETS is dropped entirely and reported.
Dev is kept as released (3,505 documents) for the official dev score; dev_excl_test marks the dev documents whose
text also occurs in test.json (text only, no labels), for the second, test-free dev score.

Chunks (token windows of length L = the model's input limit):
  even      non-overlapping windows [0, L), [L, 2L), ... (the last may be shorter); if there are more than k, k windows
            at evenly spaced positions round(linspace(0, W-1, k)) -> always the first and the last window, text order.
  headtail  if the document has <= k windows: all of them. Otherwise ceil(k/2) windows from the start ([0, L), [L, 2L),
            ...) and floor(k/2) windows ending at the document end ([n-L, n), [n-2L, n-L), ...), in text order. They
            never overlap because n > k*L.
  overlap   windows of length L every `stride` tokens, the last one ending at the document end; k evenly spaced ones
            (as in "even") if there are more than k.
A document that fits in L tokens is one chunk with all its tokens.
"""
import hashlib
import json
import math
import os

import numpy as np

PAD_ID = 0
SPLIT_FILES = {"train": ("Train", "train.json"), "dev": ("Train", "dev.json"), "test": ("Test", "test.json")}   # Phase_5_LSI/


def norm_hash(text):
    return hashlib.md5(" ".join(str(text).split()).encode("utf-8")).hexdigest()


def load_split(data_dir, split):
    sub, fn = SPLIT_FILES[split]
    raw = json.load(open(os.path.join(data_dir, sub, fn), encoding="utf-8"))
    if isinstance(raw, dict):                                   # {id: record} layout
        raw = [dict(v, id=k) for k, v in raw.items()]
    docs = []
    for r in raw:
        if split == "test":
            labs = None                                         # test labels are never read
        else:
            labs = r.get("labels", r.get("label"))
            labs = [labs] if isinstance(labs, int) else labs
            assert isinstance(labs, list) and all(isinstance(x, int) for x in labs), f"{split} {r.get('id')}: labels {labs!r}"
            labs = sorted(set(labs))
        docs.append({"id": str(r["id"]), "text": str(r["text"]), "labels": labs})
    ids = [d["id"] for d in docs]
    if split == "test":
        assert len(set(ids)) == len(ids), "duplicate ids in test.json"
    return docs


def dedup_train(docs):
    """-> (kept docs, report). One copy per exact text; conflicting-label groups are dropped."""
    groups = {}
    for d in docs:
        groups.setdefault(norm_hash(d["text"]), []).append(d)
    kept, conflicts = [], 0
    for g in groups.values():
        if len({tuple(d["labels"]) for d in g}) > 1:
            conflicts += 1
            continue
        kept.append(g[0])
    rep = {"train_docs_released": len(docs), "train_docs_distinct": len(groups), "duplicates_removed": len(docs) - len(groups),
           "conflicting_label_groups_dropped": conflicts, "train_docs_used": len(kept)}
    return kept, rep


def mark_dev_excl_test(dev, test):
    test_h = {norm_hash(d["text"]) for d in test}
    for d in dev:
        d["in_test"] = norm_hash(d["text"]) in test_h
    return sum(d["in_test"] for d in dev)


def encode_docs(tok, docs, cache_path=None):
    """-> list of np.int32 token arrays (one per doc, same order). Cached by document hashes + tokenizer file."""
    key = hashlib.md5(("|".join(norm_hash(d["text"]) for d in docs)).encode()).hexdigest()
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as z:
            if str(z["key"]) == key:
                ids, off = z["ids"], z["offsets"]          # read ONCE: every z["ids"] access re-reads the whole file
                return [ids[off[i]:off[i + 1]] for i in range(len(docs))]   # views into one array
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
        assert head[-1][1] <= tail[0][0] if tail else True
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
    """-> list of {"id", "label", "chunks": [np.int32 arrays], "n_tokens", "spans", "in_test"?}"""
    items = []
    for d, a in zip(docs, token_arrays):
        spans = chunk_spans(len(a), L, strategy, k, stride)
        chunks = [a[s:e] for s, e in spans]
        assert all(0 < len(c) <= L for c in chunks)
        it = {"id": d["id"], "labels": d["labels"], "chunks": chunks, "n_tokens": int(len(a)), "spans": spans}
        if "in_test" in d:
            it["in_test"] = d["in_test"]
        items.append(it)
    return items


def coverage(items):
    """share of document tokens that the selected chunks cover (overlaps counted once)."""
    cov = []
    for it in items:
        covered, last = 0, 0
        for s, e in it["spans"]:
            covered += max(0, e - max(s, last))
            last = max(last, e)
        cov.append(covered / it["n_tokens"])
    return float(np.mean(cov))


def multilabel_scores(gold, pred, label_space):
    """gold, pred: lists of label-id lists (same order); label_space: all statute ids of the task.
    macro_f1      = mean F1 over the labels that occur in this set's gold (the usual definition when the evaluator
                    derives the label dimension from the reference data);
    macro_f1_all  = mean F1 over the whole label space (labels absent from gold and never predicted count as F1 0);
    micro_f1, subset_accuracy, per-label P/R/F1/support."""
    idx = {l: i for i, l in enumerate(label_space)}
    n, L = len(gold), len(label_space)
    G = np.zeros((n, L), bool); P = np.zeros((n, L), bool)
    for r, (g, p) in enumerate(zip(gold, pred)):
        G[r, [idx[x] for x in g if x in idx]] = True
        P[r, [idx[x] for x in p if x in idx]] = True
    tp = (G & P).sum(0); fp = (~G & P).sum(0); fn = (G & ~P).sum(0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0); rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    present = G.sum(0) > 0
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    return {"macro_f1": round(float(f1[present].mean()) if present.any() else 0.0, 4),
            "macro_f1_all": round(float(f1.mean()), 4),
            "macro_p": round(float(prec[present].mean()) if present.any() else 0.0, 4),
            "macro_r": round(float(rec[present].mean()) if present.any() else 0.0, 4),
            "micro_f1": round(float(2 * TP / max(2 * TP + FP + FN, 1)), 4),
            "subset_accuracy": round(float((G == P).all(1).mean()), 4),
            "labels_in_gold": int(present.sum()), "n": n,
            "mean_pred_labels": round(float(P.sum(1).mean()), 3), "mean_gold_labels": round(float(G.sum(1).mean()), 3),
            "per_label": {str(l): {"p": round(float(prec[i]), 4), "r": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4),
                                   "support": int(G[:, i].sum())} for i, l in enumerate(label_space)}}
