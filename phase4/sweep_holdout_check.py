#!/usr/bin/env python3
"""Honesty check for sweep_c3.py (plumbing): choose the threshold on a random half of dev_excl_test, score it on the
other half; 500 random splits. Weights are fixed (your grid), only the threshold is re-chosen per split.
  python sweep_holdout_check.py   ->  ensembles/sweep_c3/holdout_check.json / .md"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cjpe_data
from sweep_c3 import MODELS, sweep
R = json.load(open(os.path.join(HERE, "ensembles", "sweep_c3", "results.json")))
dev, test = cjpe_data.load_split(os.path.join(HERE, "data", "Phase_4_CJPE"), "dev"), cjpe_data.load_split(os.path.join(HERE, "data", "Phase_4_CJPE"), "test")
cjpe_data.mark_dev_excl_test(dev, test)
sel = [d for d in dev if not d["in_test"]]; y = np.array([d["label"] for d in sel])
P = {m: np.array([json.load(open(os.path.join(HERE, "runs", m, "dev_probs.json")))[d["id"]] for d in sel]) for m in MODELS}
rng = np.random.default_rng(0); n = len(y); out = []
for c in R["candidates"]:
    w = c["weights"]; s = sum(w.values()); p = sum(ww / s * P[m] for m, ww in w.items())
    tuned = c["candidate"].endswith("@0.5") is False
    held, at05, ths = [], [], []
    for _ in range(500):
        idx = rng.permutation(n); a, b = idx[: n // 2], idx[n // 2:]
        th = sweep(p[a], y[a])[0] if tuned else 0.5
        held.append(cjpe_data.binary_scores(y[b], (p[b] >= th).astype(int))["macro_f1"])
        at05.append(cjpe_data.binary_scores(y[b], (p[b] >= 0.5).astype(int))["macro_f1"]); ths.append(th)
    out.append({"candidate": c["candidate"], "in_sample_f1": c["dev_excl_test_macro_f1"], "heldout_f1_mean": round(float(np.mean(held)), 4),
                "heldout_f1_sd": round(float(np.std(held)), 4), "heldout_f1_at_0.5_mean": round(float(np.mean(at05)), 4),
                "gain_over_0.5_heldout": round(float(np.mean(np.array(held) - np.array(at05))), 4),
                "threshold_chosen_mean": round(float(np.mean(ths)), 3), "threshold_chosen_sd": round(float(np.std(ths)), 3)})
out.sort(key=lambda r: -r["heldout_f1_mean"])
json.dump(out, open(os.path.join(HERE, "ensembles", "sweep_c3", "holdout_check.json"), "w"), indent=1)
lines = ["| Candidate | in-sample F1 | held-out F1 (mean ± sd, 500 half-splits) | held-out F1 @0.5 | gain vs 0.5 (held-out) | threshold chosen (mean ± sd) |", "|---|---|---|---|---|---|"]
for r in out:
    lines.append(f"| {r['candidate']} | {r['in_sample_f1']:.4f} | {r['heldout_f1_mean']:.4f} ± {r['heldout_f1_sd']:.4f} | {r['heldout_f1_at_0.5_mean']:.4f} | {r['gain_over_0.5_heldout']:+.4f} | {r['threshold_chosen_mean']:.2f} ± {r['threshold_chosen_sd']:.2f} |")
md = "\n".join(lines); open(os.path.join(HERE, "ensembles", "sweep_c3", "holdout_check.md"), "w").write(md + "\n"); print(md)
