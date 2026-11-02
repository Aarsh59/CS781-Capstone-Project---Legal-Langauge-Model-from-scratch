#!/usr/bin/env python3
"""Systematic ensemble search (plumbing; declare it). All 2- and 3-member subsets of the dev-selected models, equal
weights, mean P(1), threshold chosen on dev_excl_test (0.25..0.95 step 0.01, ties -> closest to 0.5). Ranked by the
held-out score: threshold chosen on a random half of dev_excl_test, scored on the other half (300 splits).
Reports dev_all / dev_excl_test / dev_in_test (= test, never used) at the chosen threshold; zip per subset.
  python ens_search.py   -> ensembles/search/results.md"""
import itertools, json, os, sys, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cjpe_data
M = {"C3_1ep": "runs/C3_B2_overlap", "E8_1ep": "runs/E8_KD_B2s12", "W1_1ep": "runs/W1_B2_full_weighted",
     "C3_long": "runs/C3_long/best_dev_excl_test", "E8_long": "runs/E8_long/best_dev_excl_test"}
TH = np.round(np.arange(0.25, 0.95 + 1e-9, 0.01), 2)
D = os.path.join(HERE, "data", "Phase_4_CJPE")
dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test"); cjpe_data.mark_dev_excl_test(dev, test)
lab = {cjpe_data.norm_hash(d["text"]): d["label"] for d in dev}
S = {"dev_all": [(d["id"], d["label"]) for d in dev], "dev_excl_test": [(d["id"], d["label"]) for d in dev if not d["in_test"]],
     "dev_in_test": [(d["id"], lab[cjpe_data.norm_hash(d["text"])]) for d in test]}
Y = {k: np.array([l for _, l in v]) for k, v in S.items()}
P = {}
for m, d in M.items():
    dp, tp = json.load(open(f"{HERE}/{d}/dev_probs.json")), json.load(open(f"{HERE}/{d}/test_probs.json"))
    P[m] = {k: np.array([(tp if k == "dev_in_test" else dp)[i] for i, _ in S[k]]) for k in S}
f1 = lambda y, p, t: cjpe_data.binary_scores(y, (p >= t).astype(int))["macro_f1"]
def pick(p, y):
    c = [(t, f1(y, p, t)) for t in TH]; b = max(x for _, x in c)
    return min((t for t, x in c if x == b), key=lambda t: abs(t - 0.5))
rng = np.random.default_rng(0); ye = Y["dev_excl_test"]; n = len(ye)
splits = [rng.permutation(n) for _ in range(300)]
rows = []
for r in (1, 2, 3):
    for sub in itertools.combinations(M, r):
        p = {k: np.mean([P[m][k] for m in sub], axis=0) for k in S}
        th = pick(p["dev_excl_test"], ye)
        held = [f1(ye[i[n // 2:]], p["dev_excl_test"][i[n // 2:]], pick(p["dev_excl_test"][i[: n // 2]], ye[i[: n // 2]])) for i in splits]
        rows.append({"members": "+".join(sub), "threshold": th, "heldout": float(np.mean(held)), "heldout_sd": float(np.std(held)),
                     **{k: f1(Y[k], p[k], th) for k in S}})
        d = os.path.join(HERE, "ensembles", "search", "+".join(sub)); os.makedirs(d, exist_ok=True)
        tid = [i for i, _ in S["dev_in_test"]]
        json.dump([{"id": i, "pred": int(v >= th)} for i, v in zip(tid, p["dev_in_test"])], open(f"{d}/prediction.json", "w"))
        with zipfile.ZipFile(f"{d}/submission_ENS_{'+'.join(sub)}.zip", "w", zipfile.ZIP_DEFLATED) as z: z.write(f"{d}/prediction.json", arcname="prediction.json")
rows.sort(key=lambda r: -r["heldout"])
L = ["| # | Members (equal weights) | Thr | held-out excl (mean ± sd) | dev_excl_test | dev_all | dev_in_test (= test) |", "|---|---|---|---|---|---|---|"]
for k, r in enumerate(rows, 1):
    L.append(f"| {k} | {r['members']} | {r['threshold']:.2f} | {r['heldout']:.4f} ± {r['heldout_sd']:.4f} | {r['dev_excl_test']:.4f} | {r['dev_all']:.4f} | {r['dev_in_test']:.4f} |")
os.makedirs(os.path.join(HERE, "ensembles", "search"), exist_ok=True)
open(os.path.join(HERE, "ensembles", "search", "results.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
