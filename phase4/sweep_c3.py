#!/usr/bin/env python3
"""Threshold sweep + probability-ensemble sweep for CJPE (plumbing; declare it). Your experiment spec, 2026-10-19.

  python sweep_c3.py [--runs runs] [--data data/Phase_4_CJPE] [--out ensembles/sweep_c3]

Inputs per run (downloaded from the Kaggle outputs): runs/<name>/dev_probs.json, test_probs.json ({id: P(label 1)}),
summary.json (optional: training minutes / parameters for the compute column).
Selection set: dev_excl_test = dev rows whose text is not in test.json (1,988 rows), labels from dev.json.
Thresholds 0.05..0.95 step 0.01; the threshold with the highest macro-F1 is chosen (ties -> closest to 0.5).
Nothing here reads test labels; test probabilities are only combined with the chosen weights / threshold.

Candidates: each model alone at 0.5 and tuned (C3, W1, E8 on the same documents) + the weight grid (C3, E8, W1).
Writes <out>/results.json, <out>/results.md and, for every candidate, <out>/<candidate>/{prediction.json, zip,
dev_probs.json, test_probs.json, config.json} so the selected one is reproducible without rerunning anything.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cjpe_data                                    # noqa: E402

MODELS = ["C3_B2_overlap", "E8_KD_B2s12", "W1_B2_full_weighted"]
GRID = [(0.50, 0.25, 0.25), (0.60, 0.20, 0.20), (0.70, 0.15, 0.15),
        (0.50, 0.00, 0.50), (0.75, 0.00, 0.25), (0.75, 0.25, 0.00)]          # (C3, E8, W1), your spec
THRESHOLDS = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)


def sweep(p, y):
    curve = [(float(t), cjpe_data.binary_scores(y, (p >= t).astype(int))["macro_f1"]) for t in THRESHOLDS]
    best = max(f for _, f in curve)
    th = min((t for t, f in curve if f == best), key=lambda t: abs(t - 0.5))
    return th, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "Phase_4_CJPE"))
    ap.add_argument("--out", default=os.path.join(HERE, "ensembles", "sweep_c3"))
    a = ap.parse_args()
    dev, test = cjpe_data.load_split(a.data, "dev"), cjpe_data.load_split(a.data, "test")
    cjpe_data.mark_dev_excl_test(dev, test)
    sel = [d for d in dev if not d["in_test"]]                 # selection documents, fixed order
    y = np.array([d["label"] for d in sel])
    test_ids = [d["id"] for d in test]
    dp, tp, cost = {}, {}, {}
    for m in MODELS:
        d = os.path.join(a.runs, m)
        dp[m] = json.load(open(os.path.join(d, "dev_probs.json")))
        tp[m] = json.load(open(os.path.join(d, "test_probs.json")))
        assert {x["id"] for x in dev} <= set(dp[m]), f"{m}: dev_probs.json misses dev ids"
        assert set(test_ids) == set(tp[m]), f"{m}: test_probs.json ids differ from test.json"
        sp = os.path.join(d, "summary.json")
        if os.path.exists(sp):
            s = json.load(open(sp))
            cost[m] = {"train_min": round(s["train_seconds"] / 60, 1), "trainable": s["params"]["trainable"],
                       "total_params": s["params"]["total"], "test_infer_s": s.get("test_infer_seconds")}
    P = {m: np.array([dp[m][d["id"]] for d in sel]) for m in MODELS}          # same documents, same order
    T = {m: np.array([tp[m][i] for i in test_ids]) for m in MODELS}

    cands = []
    for m in MODELS:
        cands.append((f"{m}@0.5", {m: 1.0}, 0.5))
        cands.append((f"{m}@tuned", {m: 1.0}, None))
    for w in GRID:
        name = "ENS_C3{:.2f}_E8{:.2f}_W1{:.2f}".format(*w)
        cands.append((name, {mm: ww for mm, ww in zip(MODELS, w) if ww > 0}, None))

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for name, w, fixed in cands:
        s = sum(w.values())
        p = sum(ww / s * P[m] for m, ww in w.items())
        pt = sum(ww / s * T[m] for m, ww in w.items())
        th, curve = (fixed, None) if fixed is not None else sweep(p, y)
        sc = cjpe_data.binary_scores(y, (p >= th).astype(int))
        at05 = cjpe_data.binary_scores(y, (p >= 0.5).astype(int))
        rows.append({"candidate": name, "weights": w, "threshold": th, "dev_excl_test_macro_f1": sc["macro_f1"],
                     "dev_excl_test_macro_f1_at_0.5": at05["macro_f1"], "per_class": sc["per_class"],
                     "confusion_gold_x_pred": sc["confusion_gold_x_pred"], "n": sc["n"],
                     "train_min_total": round(sum(cost.get(m, {}).get("train_min", 0) for m in w), 1) if cost else None,
                     "test_pred_share_1": round(float((pt >= th).mean()), 4), "curve": curve})
        cd = os.path.join(a.out, name.replace("@", "_"))
        os.makedirs(cd, exist_ok=True)
        pred = [{"id": i, "pred": int(v >= th)} for i, v in zip(test_ids, pt)]
        assert len(pred) == len(test) and len({x["id"] for x in pred}) == len(pred)
        json.dump(pred, open(os.path.join(cd, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(cd, f"submission_{name.replace('@', '_')}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(cd, "prediction.json"), arcname="prediction.json")
        json.dump({d["id"]: float(v) for d, v in zip(sel, p)}, open(os.path.join(cd, "dev_excl_test_probs.json"), "w"))
        json.dump({i: float(v) for i, v in zip(test_ids, pt)}, open(os.path.join(cd, "test_probs.json"), "w"))
        json.dump({"weights": w, "threshold": th, "select_on": "dev_excl_test", "thresholds": [0.05, 0.95, 0.01],
                   "members": {m: os.path.join(a.runs, m) for m in w}}, open(os.path.join(cd, "config.json"), "w"), indent=1)
    agree = {f"{MODELS[i]}~{MODELS[j]}": round(float(((P[MODELS[i]] >= 0.5) == (P[MODELS[j]] >= 0.5)).mean()), 4)
             for i in range(3) for j in range(i + 1, 3)}
    json.dump({"selection_set": "dev_excl_test", "n": int(len(y)), "agreement_at_0.5": agree, "cost": cost,
               "candidates": rows}, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    lines = ["| Candidate | Weights (C3/E8/W1) | Threshold | dev_excl_test F1 | F1 @0.5 | P0 | R0 | P1 | R1 | Confusion | Train min |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["dev_excl_test_macro_f1"]):
        pc = r["per_class"]
        wv = "/".join(f"{r['weights'].get(m, 0):.2f}" for m in MODELS)
        lines.append(f"| {r['candidate']} | {wv} | {r['threshold']:.2f} | {r['dev_excl_test_macro_f1']:.4f} | "
                     f"{r['dev_excl_test_macro_f1_at_0.5']:.4f} | {pc['0']['precision']:.3f} | {pc['0']['recall']:.3f} | "
                     f"{pc['1']['precision']:.3f} | {pc['1']['recall']:.3f} | {r['confusion_gold_x_pred']} | {r['train_min_total']} |")
    md = "\n".join(lines) + f"\n\nSelection set dev_excl_test, n = {len(y)}. Agreement at 0.5: {agree}\n"
    open(os.path.join(a.out, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
