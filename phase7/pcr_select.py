#!/usr/bin/env python3
"""Score fusion of finished PCR runs (and optionally BM25) -> dev Micro-F1@7 + test submission (plumbing; declare it).

  python pcr_select.py --name ENS1 --out DIR --data-dir DATA --members RUN_DIR [RUN_DIR ...] [--weights w1 w2 ...]
                       [--bm25-weight W] [--norm zscore|minmax|none]

Each member's dev_S.npz / test_S.npz (written by train_pcr.py) is normalised per query row (--norm), multiplied by its
weight and summed; --bm25-weight W adds BM25 scores of the same split (same normalisation). Weights are YOUR values
(course rule 9); nothing is tuned here. Only one member and --bm25-weight 0 = that run unchanged.
  python pcr_select.py --bm25-only --data-dir DATA --out DIR      # local BM25 reference score on dev (not a submission)
Outputs in DIR/<name>/: dev_scores.json, prediction.json, submission_<name>.zip.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcr_data  # noqa: E402


def norm_rows(S, how):
    S = S.astype(np.float64)
    if how == "zscore":
        return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    if how == "minmax":
        lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
        return (S - lo) / (hi - lo + 1e-9)
    return S


def load_S(run_dir, split, qids, cids):
    z = np.load(os.path.join(run_dir, f"{split}_S.npz"))
    q, c = list(z["qids"]), list(z["cids"])
    qi, ci = {x: i for i, x in enumerate(q)}, {x: j for j, x in enumerate(c)}
    assert set(q) == set(qids) and set(c) == set(cids), f"{run_dir}: {split} ids differ from the data files"
    return z["S"][np.ix_([qi[x] for x in qids], [ci[x] for x in cids])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="BM25")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--members", nargs="*", default=[])
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--bm25-weight", type=float, default=0.0)
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--norm", default="zscore", choices=["zscore", "minmax", "none"])
    a = ap.parse_args()
    assert a.members or a.bm25_only or a.bm25_weight, "give --members and/or --bm25-weight, or --bm25-only"
    w = a.weights or [1.0] * len(a.members)
    assert len(w) == len(a.members), "one weight per member"
    od = os.path.join(a.out, a.name)
    os.makedirs(od, exist_ok=True)
    res = {"members": a.members, "weights": w, "bm25_weight": 1.0 if a.bm25_only else a.bm25_weight, "norm": a.norm}
    for split in (("dev",) if a.bm25_only else ("dev", "test")):
        queries, pool = pcr_data.load_split(a.data_dir, split)
        qids, cids = [q["id"] for q in queries], [d["id"] for d in pool]
        S = np.zeros((len(qids), len(cids)))
        for m, wm in zip(a.members, w):
            S += wm * norm_rows(load_S(m, split, qids, cids), a.norm)
        if a.bm25_only or a.bm25_weight:
            S += res["bm25_weight"] * norm_rows(pcr_data.BM25(pool).score_matrix(queries), a.norm)
        if split == "dev":
            res["dev"] = pcr_data.report(S, qids, cids, {q["id"]: q["relevant"] for q in queries})
            print(json.dumps({"name": a.name, **res["dev"]["at7"], "curve": res["dev"]["curve"],
                              "recall@100": res["dev"]["recall@100"]}))
        else:
            pp = os.path.join(od, "prediction.json")
            pcr_data.write_prediction(S, qids, cids, pp)
            with zipfile.ZipFile(os.path.join(od, f"submission_{a.name}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
                z.write(pp, arcname="prediction.json")
    json.dump(res, open(os.path.join(od, "dev_scores.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
