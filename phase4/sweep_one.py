#!/usr/bin/env python3
"""Coarse + fine threshold sweep for one run's checkpoints (plumbing; declare it).
  python sweep_one.py C3_long        -> ensembles/sweep_long/<run>_<ckpt>_{coarse,fine}/ (prediction.json + zip)
Selection on dev_excl_test (coarse 0.05..0.95 step 0.01; fine = coarse best +/- 0.02, step 0.001; ties -> closest to the
coarse pick / 0.5). Reported at 0.5, coarse and fine thresholds: dev_all (3,505 rows), dev_excl_test (1,988 rows),
dev_in_test (1,517 test documents, labels from dev.json = the Codabench test score; reported only, never used to select)."""
import json, os, sys, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cjpe_data
run = sys.argv[1]
D = os.path.join(HERE, "data", "Phase_4_CJPE")
dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test"); cjpe_data.mark_dev_excl_test(dev, test)
lab = {cjpe_data.norm_hash(d["text"]): d["label"] for d in dev}
S = {"dev_all": [(d["id"], d["label"]) for d in dev], "dev_excl_test": [(d["id"], d["label"]) for d in dev if not d["in_test"]],
     "dev_in_test": [(d["id"], lab[cjpe_data.norm_hash(d["text"])]) for d in test]}
Y = {k: np.array([l for _, l in v]) for k, v in S.items()}
tid = [i for i, _ in S["dev_in_test"]]
out = {}
for ck in ("best_dev_excl_test", "best_dev_all", "best_dev_in_test"):
    d = os.path.join(HERE, "runs", run, ck)
    if not os.path.exists(os.path.join(d, "dev_probs.json")):
        continue
    dp, tp = json.load(open(os.path.join(d, "dev_probs.json"))), json.load(open(os.path.join(d, "test_probs.json")))
    P = {k: np.array([(tp if k == "dev_in_test" else dp)[i] for i, _ in S[k]]) for k in S}
    f1 = lambda k, t: cjpe_data.binary_scores(Y[k], (P[k] >= t).astype(int))["macro_f1"]
    coarse = [(float(t), f1("dev_excl_test", t)) for t in np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)]
    b = max(x for _, x in coarse); tc = min((t for t, x in coarse if x == b), key=lambda t: abs(t - 0.5))
    fine = [(float(t), f1("dev_excl_test", t)) for t in np.round(np.arange(tc - 0.02, tc + 0.02 + 1e-9, 0.001), 3)]
    bf = max(x for _, x in fine); tf = min((t for t, x in fine if x == bf), key=lambda t: abs(t - tc))
    plateau = [t for t, x in fine if x >= bf - 0.001]
    res = {}
    for tag, t in (("0.5", 0.5), ("coarse", tc), ("fine", tf)):
        res[tag] = {"threshold": t, **{k: round(f1(k, t), 4) for k in S}}
        if tag != "0.5":
            od = os.path.join(HERE, "ensembles", "sweep_long", f"{run}_{ck}_{tag}_{t}"); os.makedirs(od, exist_ok=True)
            pred = [{"id": i, "pred": int(tp[i] >= t)} for i in tid]
            json.dump(pred, open(f"{od}/prediction.json", "w"))
            with zipfile.ZipFile(f"{od}/submission_{run}_{ck}_{tag}_{t}.zip", "w", zipfile.ZIP_DEFLATED) as z: z.write(f"{od}/prediction.json", arcname="prediction.json")
            res[tag]["zip"] = f"{od}/submission_{run}_{ck}_{tag}_{t}.zip"
    res["fine_plateau_within_0.001"] = [plateau[0], plateau[-1]]
    res["coarse_curve_near_best"] = [(t, round(x, 4)) for t, x in coarse if abs(t - tc) <= 0.05]
    out[ck] = res
same = {}
json.dump(out, open(os.path.join(HERE, "ensembles", "sweep_long", f"{run}_threshold_sweep.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
