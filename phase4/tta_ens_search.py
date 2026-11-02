#!/usr/bin/env python3
"""Equal-weight ensembles of re-chunked variants (plumbing; declare it). Threshold tuned on dev_excl_test (994 distinct
docs, 0.25..0.95 then +/-0.02 step 0.001); ranked by stratified 5-fold x 3 held-out; dev_all / dev_in_test (= test)
reported only.  python tta_ens_search.py -> ensembles/tta/ens_results.md + zips for the top 5."""
import itertools, json, os, sys, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cjpe_data
from tta_analysis import f1_many, tune
z = np.load(os.path.join(HERE, "ensembles", "tta", "tta_probs.npz"))
CAND = {"C3_avg8": "C3_B2_overlap__avg8", "C3_own": "C3_B2_overlap__overlap_4_640", "E8L_ov8": "E8_long__overlap_8_640",
        "E8L_avg8": "E8_long__avg8", "C3L_own": "C3_long__overlap_4_640", "C3L_ov8": "C3_long__overlap_8_640"}
S = ("dev_all", "dev_excl_test", "dev_in_test")
D = os.path.join(HERE, "data", "Phase_4_CJPE")
dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test"); cjpe_data.mark_dev_excl_test(dev, test)
lab = {cjpe_data.norm_hash(d["text"]): d["label"] for d in dev}
excl, seen = [], set()
for d in dev:
    if not d["in_test"] and d["id"] not in seen:
        seen.add(d["id"]); excl.append(d["label"])
Y = {"dev_all": np.array([d["label"] for d in dev]), "dev_excl_test": np.array(excl),
     "dev_in_test": np.array([lab[cjpe_data.norm_hash(d["text"])] for d in test])}
assert all(len(Y[k]) == len(z[f"{CAND['C3_own']}__{k}"]) for k in S)
P = {c: {k: z[f"{v}__{k}"] for k in S} for c, v in CAND.items()}
# the one-epoch E8 (not re-chunked) from its original run
e8d, e8t = json.load(open("runs/E8_KD_B2s12/dev_probs.json")), json.load(open("runs/E8_KD_B2s12/test_probs.json"))
P["E8_1ep"] = {"dev_all": np.array([e8d[d["id"]] for d in dev]), "dev_excl_test": np.array([e8d[i] for i in z["ids__dev_excl_test"]]),
               "dev_in_test": np.array([e8t[d["id"]] for d in test])}
y = Y["dev_excl_test"]; rng = np.random.default_rng(0); folds_all = []
for _ in range(3):
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; rng.shuffle(i0); rng.shuffle(i1)
    folds_all.append([np.concatenate([i0[f::5], i1[f::5]]) for f in range(5)])
rows = []
for r in (1, 2, 3):
    for sub in itertools.combinations(P, r):
        p = {k: np.mean([P[m][k] for m in sub], axis=0) for k in S}
        t = tune(p["dev_excl_test"], y); held = []
        for folds in folds_all:
            for f in range(5):
                te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
                held.append(float(f1_many(p["dev_excl_test"][te], y[te], np.array([tune(p["dev_excl_test"][tr], y[tr])]))[0]))
        rows.append({"members": "+".join(sub), "thr": t, "held": float(np.mean(held)), "sd": float(np.std(held)),
                     **{k: float(f1_many(p[k], Y[k], np.array([t]))[0]) for k in S}, "p_test": p["dev_in_test"]})
rows.sort(key=lambda r: -r["held"])
L = ["| # | Members | Thr | held-out excl | dev_excl_test | dev_all | dev_in_test (= test) |", "|---|---|---|---|---|---|---|"]
for k, r in enumerate(rows[:20], 1):
    L.append(f"| {k} | {r['members']} | {r['thr']:.3f} | {r['held']:.4f} ± {r['sd']:.4f} | {r['dev_excl_test']:.4f} | {r['dev_all']:.4f} | {r['dev_in_test']:.4f} |")
open(os.path.join(HERE, "ensembles", "tta", "ens_results.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
tid = [d["id"] for d in test]
for r in rows[:5]:
    d = os.path.join(HERE, "ensembles", "tta", "ENS_" + r["members"]); os.makedirs(d, exist_ok=True)
    json.dump([{"id": i, "pred": int(v >= r["thr"])} for i, v in zip(tid, r["p_test"])], open(f"{d}/prediction.json", "w"))
    with zipfile.ZipFile(f"{d}/submission_ENS_{r['members']}.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(f"{d}/prediction.json", arcname="prediction.json")
    json.dump({"members": r["members"].split("+"), "threshold": r["thr"], "weights": "equal"}, open(f"{d}/config.json", "w"))
