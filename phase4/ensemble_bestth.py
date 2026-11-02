#!/usr/bin/env python3
"""Submissions from the individually tuned thresholds (plumbing; declare it).
Individual: model m predicts 1 iff p_m >= t_m.  Ensemble (your 6 weight ratios): predict 1 iff sum_m w_m (p_m - t_m) >= 0,
i.e. every model is centred at its own best threshold; no further threshold search (nothing more fitted on dev).
Thresholds = the finest-sweep picks on dev_excl_test.  Writes ensembles/bestth/<name>/{prediction.json, zip, config.json}
and ensembles/bestth/results.md (dev_excl_test scores: thresholds were chosen on these documents -> in-sample)."""
import json, os, sys, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cjpe_data
TH = {"C3_B2_overlap": 0.295, "E8_KD_B2s12": 0.3605, "W1_B2_full_weighted": 0.38}
GRID = [(0.50, 0.25, 0.25), (0.60, 0.20, 0.20), (0.70, 0.15, 0.15), (0.50, 0.00, 0.50), (0.75, 0.00, 0.25), (0.75, 0.25, 0.00)]
M = list(TH)
D = os.path.join(HERE, "data", "Phase_4_CJPE")
dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test"); cjpe_data.mark_dev_excl_test(dev, test)
sel = [d for d in dev if not d["in_test"]]; y = np.array([d["label"] for d in sel]); tid = [d["id"] for d in test]
P = {m: np.array([json.load(open(f"{HERE}/runs/{m}/dev_probs.json"))[d["id"]] for d in sel]) for m in M}
T = {m: np.array([json.load(open(f"{HERE}/runs/{m}/test_probs.json"))[i] for i in tid]) for m in M}
OUT = os.path.join(HERE, "ensembles", "bestth"); os.makedirs(OUT, exist_ok=True)
cands = [(f"{m}_th{TH[m]}", {m: 1.0}) for m in M] + [("ENS_bestth_C3{:.2f}_E8{:.2f}_W1{:.2f}".format(*w), {m: x for m, x in zip(M, w) if x}) for w in GRID]
rows = []
for name, w in cands:
    s = sum(w.values())
    dv = sum(x / s * (P[m] - TH[m]) for m, x in w.items()); tv = sum(x / s * (T[m] - TH[m]) for m, x in w.items())
    sc = cjpe_data.binary_scores(y, (dv >= 0).astype(int)); pred = [{"id": i, "pred": int(v >= 0)} for i, v in zip(tid, tv)]
    assert len(pred) == len(test) and len({p["id"] for p in pred}) == len(pred)
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    json.dump(pred, open(f"{d}/prediction.json", "w"))
    with zipfile.ZipFile(f"{d}/submission_{name}.zip", "w", zipfile.ZIP_DEFLATED) as z: z.write(f"{d}/prediction.json", arcname="prediction.json")
    eff = sum(x / s * TH[m] for m, x in w.items())
    json.dump({"weights": w, "thresholds": {m: TH[m] for m in w}, "rule": "sum w_m (p_m - t_m) >= 0", "equivalent_avg_threshold": round(eff, 4)}, open(f"{d}/config.json", "w"), indent=1)
    rows.append((name, w, round(eff, 4), sc, round(float(np.mean([p["pred"] for p in pred])), 4), f"{d}/submission_{name}.zip"))
L = ["| Submission | Weights C3/E8/W1 | Effective threshold | dev_excl_test F1 (in-sample) | P1 | R1 | Test share pred 1 |", "|---|---|---|---|---|---|---|"]
for name, w, eff, sc, sh, _ in rows:
    L.append(f"| {name} | {'/'.join(f'{w.get(m, 0):.2f}' for m in M)} | {eff} | {sc['macro_f1']:.4f} | {sc['per_class']['1']['precision']:.3f} | {sc['per_class']['1']['recall']:.3f} | {sh} |")
open(f"{OUT}/results.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(); [print(r[5]) for r in rows]
