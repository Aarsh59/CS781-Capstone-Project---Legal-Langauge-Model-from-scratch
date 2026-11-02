#!/usr/bin/env python3
"""CPU data audit + build of genuinely new pretraining data for the B2 context extension (plumbing; declare it).

  python data_audit.py data_audit_config.json

Stages (every stage writes into cfg["out_dir"]; nothing existing is modified):
  A. REPLAY the exact Phase-1 selection: the verbatim code of capstone_phase1_arch12.ipynb cells [17] (helpers) and
     [19] (data build) is executed with the same constants, seeds and tokenizer, with one instrumentation patch that
     records each kept document's doc_hash (md5 of the first 2,000 normalised characters, the build's own key).
     The replayed train.bin is compared token-for-token (md5 + length) with the ORIGINAL train.bin: only a perfect
     match counts as a reconstruction of which documents B and B2 were trained on.
  B. SCAN every High Court and Tribunal record. Each document is classified: dropped_short (< MIN_WORDS, as in the
     build), used_in_original (hash among the selected docs), exact duplicate (hash seen in training / validation /
     earlier in the scan), near-duplicate of training text, contamination (overlap with Phase-1 test prompts, the
     SC validation slice, or any Phase 2-7 test/dev/validation file), or candidate.
     Near-duplicates: word 13-gram shingles, deterministically subsampled (hash mod sample_mod == 0); overlap = share of
     a document's sampled shingles found in the reference index. Thresholds: cfg["near_dup"].
  C. ENCODE the surviving candidates with the exact legal_bpe.json (same worker as the build; same MAX_FERTILITY drop),
     hash-splitting heldout_percent of documents into per-source held-out evaluation files (never trained on).
  C2. SC + ACTS: every training document once into supreme_court_unique.bin / acts_unique.bin (repeats only, but
      separately weightable in the mixture; the SC validation slice is excluded).
  D. MANIFEST (manifest_ctxext.json): the reconstructed original manifest, replay verification, per-source counts
     at every step, overlap distributions, contamination sources, and the token counts available for training.
"""
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import zlib
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(sys.argv[1]))
OUT, SCR = cfg["out_dir"], cfg["scratch_dir"]
os.makedirs(OUT, exist_ok=True); os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, cfg.get("p1_code_dir", os.path.join(HERE, "p1")))
import a1_tokenizer                                       # noqa: E402  (verbatim Phase-1 tokenizer code)
from a1_tokenizer import ByteBPE                          # noqa: E402

T0 = time.time()


def log(msg, **kv):
    rec = {"t_min": round((time.time() - T0) / 60, 1), "msg": msg, **kv}
    print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, "data_audit_log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def md5_file(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


tokenizer = ByteBPE.from_json(cfg["tokenizer"])
V, EOT_ID = tokenizer.get_vocab_size(), tokenizer.eot_id
P1 = cfg["phase1_constants"]                              # cell [4] values (copied, not chosen)

# ======================================================================= A. replay
ns = {"os": os, "json": json, "time": time, "random": random, "hashlib": hashlib, "np": np, "shutil": shutil, "glob": glob,
      "Counter": Counter, "mp": mp, "a1_tokenizer": a1_tokenizer, "tokenizer": tokenizer, "V": V, "EOT_ID": EOT_ID,
      "N_WORKERS": cfg.get("workers") or os.cpu_count() or 4, "SOURCES": cfg["corpus_files"], "NEED_DATA": True,
      "DATA_DIR": os.path.join(SCR, "replay"), "fert": None, **P1}
os.makedirs(ns["DATA_DIR"], exist_ok=True)
exec(open(os.path.join(cfg.get("p1_code_dir", os.path.join(HERE, "p1")), "cell17_defs.py")).read(), ns)
cell19 = open(os.path.join(cfg.get("p1_code_dir", os.path.join(HERE, "p1")), "cell19.py")).read()
PATCHES = [   # instrumentation only: record doc_hash of every kept doc (same order as the build's offsets)
    ("    lengths, scores, stats = [], [], Counter()\n",
     "    lengths, scores, stats = [], [], Counter()\n    kept_hashes, _hl = [], []\n\n    def _tap(it):\n"
     "        for _t in it:\n            _hl.append(doc_hash(_t))\n            yield _t\n"),
    ("        for arr, n_words in pool.imap(_encode_doc, texts, chunksize=8):\n",
     "        for _k, (arr, n_words) in enumerate(pool.imap(_encode_doc, _tap(texts), chunksize=8)):\n"),
    ("            lengths.append(len(arr))\n", "            lengths.append(len(arr))\n            kept_hashes.append(_hl[_k])\n"),
    ('    return {"bin": bin_path, "offsets": offsets, "scores": np.asarray(scores, np.float32), "stats": dict(stats)}\n',
     '    return {"bin": bin_path, "offsets": offsets, "scores": np.asarray(scores, np.float32), "stats": dict(stats), "hashes": kept_hashes}\n'),
    ('SCRATCH = "/tmp/cs781_scratch"\n', f'SCRATCH = {os.path.join(SCR, "build_scratch")!r}\n'),
]
for a, b in PATCHES:
    assert cell19.count(a) == 1, f"patch anchor not found exactly once: {a[:60]!r}"
    cell19 = cell19.replace(a, b)
log("replay: start (verbatim cell [19] + doc_hash instrumentation)")
exec(cell19, ns)
enc, picks = ns["encoded"], ns["all_picks"]
replay_bin = os.path.join(ns["DATA_DIR"], "train.bin")
rep = {"replayed_train_tokens": int(ns["train_len"]), "replayed_md5": md5_file(replay_bin)}
if cfg.get("original_train_bin") and os.path.exists(cfg["original_train_bin"]):
    rep["original_train_tokens"] = int(os.path.getsize(cfg["original_train_bin"]) // 2)
    rep["original_md5"] = md5_file(cfg["original_train_bin"])
    rep["exact_match"] = rep["original_md5"] == rep["replayed_md5"]
else:
    rep["exact_match"] = None
    rep["note"] = "original train.bin not attached: replay cannot be verified"
if cfg.get("original_val_bin") and os.path.exists(cfg["original_val_bin"]):
    rep["val_exact_match"] = md5_file(cfg["original_val_bin"]) == md5_file(os.path.join(ns["DATA_DIR"], "val.bin"))
log("replay: done", **rep)
if rep["exact_match"] is False and not cfg.get("continue_if_replay_mismatch", False):
    json.dump({"replay": rep, "reconstructed_original_manifest": ns["manifest"]}, open(os.path.join(OUT, "manifest_ctxext.json"), "w"), indent=2)
    sys.exit("REPLAY MISMATCH: the reconstruction is not the data B/B2 saw (different corpus revision?). Stopping before the scan.")

picked = {s: set() for s in enc}
times_in_train = Counter()
for s, i in picks:
    h = enc[s]["hashes"][i]
    picked[s].add(h); times_in_train[(s, h)] += 1
train_hashes = set().union(*picked.values())
val_texts = ns["val_texts"]
val_hashes = {ns["doc_hash"](t) for t in val_texts}

# ======================================================================= near-duplicate index
NDC = cfg["near_dup"]
K, MOD = NDC["shingle_words"], NDC["sample_mod"]
_WORD = {}
PW = (np.uint64(1000003) ** np.arange(K - 1, -1, -1, dtype=np.uint64)).astype(np.uint64)


def fingerprints(text):
    w = text.split()
    if len(w) < K:
        return np.zeros(0, np.uint64)
    ids = np.fromiter((_WORD.get(x) if x in _WORD else _WORD.setdefault(x, zlib.crc32(x.encode()) | 1) for x in w),
                      dtype=np.uint64, count=len(w))
    n = len(ids) - K + 1
    h = np.zeros(n, np.uint64)
    for j in range(K):
        h += ids[j:j + n] * PW[j]
    h ^= h >> np.uint64(29)
    return np.unique(h[(h % np.uint64(MOD)) == 0])


def build_index(texts_iter):
    parts = [fingerprints(t) for t in texts_iter]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.uint64)


def overlap(fp, index):
    # index is sorted (np.unique): binary search per shingle. np.isin would re-sort the whole index for every document
    if not (len(fp) and len(index)):
        return 0.0
    i = np.searchsorted(index, fp)
    i[i == len(index)] = 0
    return float((index[i] == fp).mean())


log("index: training text fingerprints")
norm, clean_min = ns["normalize_ws"], P1["MIN_WORDS"]


def training_texts():
    for s in ("supreme_court", "acts"):
        for raw in ns["iter_texts"](cfg["corpus_files"][s]):
            t = norm(raw)
            if t.count(" ") + 1 >= clean_min and ns["doc_hash"](t) in picked[s]:
                yield t
    for s in ("high_courts", "tribunals"):         # the selected HC/Tribunal docs came from the seeded random-seek sample
        sizes = {x: os.path.getsize(cfg["corpus_files"][x]) for x in ("high_courts", "tribunals")}
        rest = ns["rest_budget"]
        scan_chars = P1["HT_SCAN_FACTOR"] * rest * ns["sc_chars_per_token"]
        for raw in ns["sample_texts_random_seek"](cfg["corpus_files"][s], int(scan_chars * sizes[s] / sum(sizes.values())), seed=P1["SEED"] + 3):
            t = norm(raw)
            if ns["doc_hash"](t) in picked[s]:
                yield t


train_index = build_index(training_texts())
protected_texts, protected_sources = list(val_texts), Counter({"phase1_sc_validation_slice": len(val_texts)})
for path in cfg.get("protected_files", []):
    def strings(o):
        if isinstance(o, str):
            yield o
        elif isinstance(o, list):
            if o and all(isinstance(x, str) for x in o):
                yield " ".join(o)
            else:
                for x in o:
                    yield from strings(x)
        elif isinstance(o, dict):
            for x in o.values():
                yield from strings(x)
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        log("protected file unreadable", path=path, error=repr(e)[:200]); continue
    got = [norm(s) for s in strings(data) if len(s.split()) >= NDC["protected_min_words"]]
    protected_texts += got; protected_sources[os.path.relpath(path, cfg.get("protected_root", "/"))] = len(got)
protected_index = build_index(protected_texts)
protected_hashes = {ns["doc_hash"](t) for t in protected_texts}
log("index: done", train_fingerprints=int(len(train_index)), protected_fingerprints=int(len(protected_index)),
    protected_texts=len(protected_texts))

# ======================================================================= B. scan HC / Tribunals
def classify(text):
    fp = fingerprints(text)
    return overlap(fp, train_index), overlap(fp, protected_index), len(fp)


cand_paths, stats, ovl_hist = {}, {}, {}
bins = np.array([0, .01, .05, .1, .2, .3, .5, .7, .9, 1.0001])
for s in ("high_courts", "tribunals"):
    st = Counter(); seen_scan = set(); hist_t = np.zeros(len(bins) - 1, int); hist_p = np.zeros(len(bins) - 1, int)
    keep_path = os.path.join(SCR, f"candidates_{s}.jsonl")
    with open(keep_path, "w", encoding="utf-8") as out:
        batch = []

        def flush(batch):
            with mp.get_context("fork").Pool(ns["N_WORKERS"]) as pool:
                res = pool.map(classify, [t for t, _ in batch], chunksize=64)
            for (t, h), (ot, op, nf) in zip(batch, res):
                hist_t[np.searchsorted(bins, ot, side="right") - 1] += 1
                hist_p[np.searchsorted(bins, op, side="right") - 1] += 1
                if op >= NDC["protected_threshold"]:
                    st["contamination_near_dup"] += 1
                elif ot >= NDC["train_threshold"]:
                    st["near_dup_of_training"] += 1
                else:
                    st["candidate"] += 1
                    out.write(json.dumps({"h": str(h), "t": t}) + "\n")
        for raw in ns["iter_texts"](cfg["corpus_files"][s]):
            st["records"] += 1
            t = norm(raw)
            if t.count(" ") + 1 < clean_min:
                st["dropped_short"] += 1; continue
            h = ns["doc_hash"](t)
            if h in picked[s]:
                st["used_in_original_train"] += 1; continue
            if h in train_hashes:
                st["exact_dup_of_training"] += 1; continue
            if h in protected_hashes:
                st["contamination_exact"] += 1; continue
            if h in seen_scan:
                st["exact_dup_within_source"] += 1; continue
            seen_scan.add(h)
            st["chars_after_exact_dedup"] += len(t)
            batch.append((t, h))
            if len(batch) >= cfg.get("scan_batch_docs", 20000):
                flush(batch); batch = []
                log(f"scan {s}: progress", records=st["records"], candidates=st["candidate"], near_dup=st["near_dup_of_training"],
                    contamination=st["contamination_near_dup"] + st["contamination_exact"], used_in_original=st["used_in_original_train"])
        if batch:
            flush(batch)
    stats[s], cand_paths[s] = st, keep_path
    ovl_hist[s] = {"bins": bins.tolist(), "overlap_with_training": hist_t.tolist(), "overlap_with_protected": hist_p.tolist()}
    log(f"scan {s}: done", **st)

# ======================================================================= C. encode
held = cfg["heldout_percent"]


def heldout(text):
    return int(hashlib.md5(("ctxext-heldout:" + text[:2000]).encode("utf-8", errors="ignore")).hexdigest(), 16) % 100 < held


files = {}
for s in ("high_courts", "tribunals"):
    def texts():
        for line in open(cand_paths[s], encoding="utf-8"):
            yield json.loads(line)["t"]
    tr_path, ho_path = os.path.join(OUT, f"new_{s}.bin"), os.path.join(OUT, f"heldout_{s}.bin")
    st = stats[s]
    from collections import deque
    flags = deque()                                          # held-out flag per text, in imap (= input) order

    def tapped():
        for t in texts():
            flags.append(heldout(t))
            yield t
    with open(tr_path, "wb") as ftr, open(ho_path, "wb") as fho, tokenizer.pool(ns["N_WORKERS"]) as pool:
        for arr, n_words in pool.imap(a1_tokenizer.encode_array_worker, tapped(), chunksize=8):
            is_held = flags.popleft()
            if len(arr) / n_words > P1["MAX_FERTILITY"]:
                st["dropped_fertility"] += 1; continue
            dst = fho if is_held else ftr
            key = "heldout" if dst is fho else "train"
            arr.tofile(dst); np.array([EOT_ID], np.uint16).tofile(dst)
            st[f"{key}_docs"] += 1; st[f"{key}_tokens"] += int(len(arr)); st[f"{key}_words"] += int(n_words)
    files[s] = {"train_bin": tr_path, "heldout_bin": ho_path}
    log(f"encode {s}: done", train_docs=st["train_docs"], train_tokens=st["train_tokens"], heldout_docs=st["heldout_docs"],
        heldout_tokens=st["heldout_tokens"])

# ======================================================================= C2. SC + Acts, each training document once
# (already seen in Phase 1: only repeats are possible. Written separately so the mixture can weight SC and Acts directly;
#  the SC validation slice is excluded because only documents selected for train.bin are written.)
for s in ("supreme_court", "acts"):
    path, st = os.path.join(OUT, f"{s}_unique.bin"), Counter()

    def texts_s(s=s):
        for raw in ns["iter_texts"](cfg["corpus_files"][s]):
            t = norm(raw)
            if t.count(" ") + 1 >= clean_min:
                h = ns["doc_hash"](t)
                if h in picked[s] and h not in st:              # each selected doc once (first occurrence)
                    st[h] = 1
                    yield t
    n_docs = n_tok = 0
    with open(path, "wb") as f, tokenizer.pool(ns["N_WORKERS"]) as pool:
        for arr, _ in pool.imap(a1_tokenizer.encode_array_worker, texts_s(), chunksize=8):
            arr.tofile(f); np.array([EOT_ID], np.uint16).tofile(f); n_docs += 1; n_tok += int(len(arr))
    files[s] = {"train_bin": path}
    stats[s] = {"unique_docs_written": n_docs, "unique_tokens_written": n_tok,
                "times_seen_in_phase1_train_bin": round(sum(c for (x, _), c in times_in_train.items() if x == s) / max(1, len(picked[s])), 2)}
    log(f"unique {s}: done", **stats[s])

# ======================================================================= D. manifest
orig = ns["manifest"]
man = {
    "replay": rep,
    "reconstructed_original_manifest": orig,
    "original_selection": {
        s: {"docs_kept_after_cleaning": int(enc[s]["stats"].get("docs_kept", 0)), "unique_tokens": int(enc[s]["stats"].get("unique_tokens", 0)),
            "docs_selected": len(picked[s]), "doc_copies_in_train_bin": int(sum(c for (x, _), c in times_in_train.items() if x == s)),
            "tokens_in_train_bin": int(orig["sources"].get(s, {}).get("tokens_used", 0)),
            "dropped_short_in_scanned_part": int(ns["stream_stats"].get(s, {}).get("dropped_short", 0)),
            "dropped_fertility_in_scanned_part": int(enc[s]["stats"].get("dropped_fertility", 0))} for s in enc},
    "sc_validation_slice": {"docs": len(val_texts), "tokens": int(orig["val_tokens"])},
    "new_data_scan": {s: dict(stats[s]) for s in ("high_courts", "tribunals")},
    "sc_acts_unique_files": {s: stats[s] for s in ("supreme_court", "acts")},
    "overlap_histograms": ovl_hist,
    "near_dup_settings": NDC,
    "protected_sources": dict(protected_sources),
    "heldout_percent": held,
    "files": files,
    "tokenizer": {"path": cfg["tokenizer"], "sha256_16": hashlib.sha256(open(cfg["tokenizer"], "rb").read()).hexdigest()[:16],
                  "vocab_size": V, "eot_id": EOT_ID, "retrained": False},
    "available_new_training_tokens": int(sum(stats[s]["train_tokens"] for s in ("high_courts", "tribunals"))),
    "repeat_only_unique_tokens": {s: stats[s]["unique_tokens_written"] for s in ("supreme_court", "acts")},
    "minutes": round((time.time() - T0) / 60, 1),
}
json.dump(man, open(os.path.join(OUT, "manifest_ctxext.json"), "w"), indent=2)
log("manifest written", available_new_training_tokens_B=round(man["available_new_training_tokens"] / 1e9, 3))
shutil.rmtree(os.path.join(SCR, "replay"), ignore_errors=True)
