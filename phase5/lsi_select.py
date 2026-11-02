#!/usr/bin/env python3
"""Thresholds + probability ensembles for LSI runs (plumbing; declare it).

  python lsi_select.py --name NAME --out ensembles --members runs/A [runs/B ...] [--weights 1 1]
                       [--mode global|per_label] [--min-support 5] [--min-one] [--select-on dev_excl_test|dev_all]

P(statute) per document = weighted mean of the members' dev_probs.json / test_probs.json (all members must share
labels.json). Thresholds are chosen on DEV only (the selection set; test labels are never read):
  global     one threshold for every statute, grid 0.05..0.95 step 0.01, maximising macro-F1 (ties -> closest to 0.5)
  per_label  start from the global threshold; every statute with >= --min-support positives in the selection set gets
             its own threshold (same grid) maximising ITS F1; rarer statutes keep the global one
--min-one: a document with no statute above threshold gets its single most probable statute (your choice; off by default).
Reported: macro-F1 (labels present in gold), macro-F1 over all labels, micro-F1, subset accuracy on dev_all /
dev_excl_test / dev_in_test (dev rows also in test.json, if any). Writes <out>/<name>/{prediction.json, zip,
summary.json, thresholds.json}.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lsi_data                                      # noqa: E402

GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)


def predict(P, th, label_space, min_one):
    out = []
    for row in P:
        on = [label_space[j] for j in np.where(row >= th)[0]]
        if not on and min_one:
            on = [label_space[int(row.argmax())]]
        out.append(on)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--mode", default="global", choices=("global", "per_label"))
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-one", action="store_true")
    ap.add_argument("--select-on", default="dev_excl_test", choices=("dev_excl_test", "dev_all"))
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.members)); w = w / w.sum()
    label_space = json.load(open(os.path.join(a.members[0], "labels.json")))
    for m in a.members[1:]:
        assert json.load(open(os.path.join(m, "labels.json"))) == label_space, f"{m}: different labels.json"
    dev, test = lsi_data.load_split(a.data_dir, "dev"), lsi_data.load_split(a.data_dir, "test")
    lsi_data.mark_dev_excl_test(dev, test)
    dp = [json.load(open(os.path.join(m, "dev_probs.json"))) for m in a.members]
    tp = [json.load(open(os.path.join(m, "test_probs.json"))) for m in a.members]
    sets = {"dev_all": dev, "dev_excl_test": [d for d in dev if not d["in_test"]], "dev_in_test": [d for d in dev if d["in_test"]]}
    for m, d_, t_ in zip(a.members, dp, tp):
        assert {d["id"] for d in dev} <= set(d_), f"{m}: dev_probs misses dev ids"
        assert set(t_) == {d["id"] for d in test}, f"{m}: test_probs ids differ from test.json"
    Pd = {k: np.array([sum(wi * np.array(d_[d["id"]]) for wi, d_ in zip(w, dp)) for d in v]) if v else None for k, v in sets.items()}
    Pt = np.array([sum(wi * np.array(t_[d["id"]]) for wi, t_ in zip(w, tp)) for d in test])
    sel = sets[a.select_on] or sets["dev_all"]
    sel_key = a.select_on if sets[a.select_on] else "dev_all"
    gold_sel = [d["labels"] for d in sel]
    score = lambda P, th, gold: lsi_data.multilabel_scores(gold, predict(P, th, label_space, a.min_one), label_space)
    curve = [(float(t), score(Pd[sel_key], np.full(len(label_space), t), gold_sel)["macro_f1"]) for t in GRID]
    best = max(f for _, f in curve)
    tg = min((t for t, f in curve if f == best), key=lambda t: abs(t - 0.5))
    th = np.full(len(label_space), tg)
    if a.mode == "per_label":
        G = np.array([[l in g for l in label_space] for g in gold_sel])
        for j in range(len(label_space)):
            if G[:, j].sum() < a.min_support:
                continue
            p = Pd[sel_key][:, j]
            f1 = []
            for t in GRID:
                pr = p >= t; tp_ = (pr & G[:, j]).sum(); fp = (pr & ~G[:, j]).sum(); fn = (~pr & G[:, j]).sum()
                f1.append(2 * tp_ / max(2 * tp_ + fp + fn, 1))
            f1 = np.array(f1); th[j] = float(min(GRID[f1 == f1.max()], key=lambda t: abs(t - tg)))
    rep = {k: {kk: vv for kk, vv in score(Pd[k], th, [d["labels"] for d in v]).items() if kk != "per_label"}
           for k, v in sets.items() if v}
    pred = [{"id": d["id"], "pred": [int(x) for x in p]} for d, p in zip(test, predict(Pt, th, label_space, a.min_one))]
    assert len(pred) == len(test) and len({x["id"] for x in pred}) == len(pred)
    assert all(len(set(x["pred"])) == len(x["pred"]) and set(x["pred"]) <= set(label_space) for x in pred)
    out = os.path.join(a.out, a.name); os.makedirs(out, exist_ok=True)
    json.dump(pred, open(os.path.join(out, "prediction.json"), "w"))
    with zipfile.ZipFile(os.path.join(out, f"submission_{a.name}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out, "prediction.json"), arcname="prediction.json")
    json.dump({str(l): float(t) for l, t in zip(label_space, th)}, open(os.path.join(out, "thresholds.json"), "w"))
    summ = {"name": a.name, "members": a.members, "weights": w.tolist(), "mode": a.mode, "min_support": a.min_support,
            "min_one": a.min_one, "select_on": sel_key, "global_threshold": tg, "dev": rep,
            "test_mean_pred_labels": round(float(np.mean([len(x["pred"]) for x in pred])), 3),
            "test_docs_with_no_label": int(sum(1 for x in pred if not x["pred"])), "curve": curve}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
