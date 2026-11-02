#!/usr/bin/env python3
"""Inference-only re-chunking results (Stage 9) + test-time averaging (plumbing; declare it).

  python tta_analysis.py [predict_root]    (default kaggle_dl/phase4/predict)  -> ensembles/tta/{results.md, results.json, zips}

Per run: each chunking variant and the averages avg4 (even/headtail/overlap with 4 chunks), avg8 (same with 8), avg_all
(all 6). Threshold = 0.5, and tuned on dev_excl_test (994 distinct documents; 0.25..0.95 step 0.01, then +/-0.02 step
0.001; ties -> closest to 0.5) with a stratified 5-fold x 3 held-out check. Reported: dev_all (3,505 rows), dev_excl_test,
dev_in_test (1,517 test documents, labels from dev.json = the test score; reported only, never used to select).
"""
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cjpe_data                                       # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_dl", "phase4", "predict")
OWN = {"C3_B2_overlap": "overlap_4_640", "C3_long": "overlap_4_640", "E8_long": "even_4"}   # each run's training chunking
V4, V8 = ["even_4", "headtail_4", "overlap_4_640"], ["even_8", "headtail_8", "overlap_8_640"]
OUT = os.path.join(HERE, "ensembles", "tta")


def f1_many(p, y, ths):
    pred = p[None, :] >= ths[:, None]; y1 = (y == 1)[None, :]
    tp1 = (pred & y1).sum(1); fp1 = (pred & ~y1).sum(1); fn1 = (~pred & y1).sum(1); tp0 = (~pred & ~y1).sum(1)
    f = lambda tp, fp, fn: np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return (f(tp1, fp1, fn1) + f(tp0, fn1, fp1)) / 2


def tune(p, y):
    c = np.round(np.arange(0.25, 0.95 + 1e-9, 0.01), 2); fc = f1_many(p, y, c)
    tc = min(c[fc == fc.max()], key=lambda t: abs(t - 0.5))
    fn = np.round(np.arange(tc - 0.02, tc + 0.02 + 1e-9, 0.001), 3); ff = f1_many(p, y, fn)
    return float(min(fn[ff == ff.max()], key=lambda t: abs(t - tc)))


def main():
    D = os.path.join(HERE, "data", "Phase_4_CJPE")
    dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test"); cjpe_data.mark_dev_excl_test(dev, test)
    lab = {cjpe_data.norm_hash(d["text"]): d["label"] for d in dev}
    excl, seen = [], set()
    for d in dev:
        if not d["in_test"] and d["id"] not in seen:
            seen.add(d["id"]); excl.append((d["id"], d["label"]))
    S = {"dev_all": [(d["id"], d["label"]) for d in dev], "dev_excl_test": excl,
         "dev_in_test": [(d["id"], lab[cjpe_data.norm_hash(d["text"])]) for d in test]}
    Y = {k: np.array([l for _, l in v]) for k, v in S.items()}
    rng = np.random.default_rng(0)
    y = Y["dev_excl_test"]
    folds_all = []
    for _ in range(3):
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; rng.shuffle(i0); rng.shuffle(i1)
        folds_all.append([np.concatenate([i0[f::5], i1[f::5]]) for f in range(5)])
    os.makedirs(OUT, exist_ok=True)
    rows, probs = [], {}
    for run in sorted(os.listdir(ROOT)):
        V = {}
        for v in sorted(os.listdir(os.path.join(ROOT, run))):
            d = os.path.join(ROOT, run, v)
            dp, tp = json.load(open(os.path.join(d, "dev_probs.json"))), json.load(open(os.path.join(d, "test_probs.json")))
            # dev_probs may cover only the dev documents NOT in test (the Kaggle dev.json was updated); test documents
            # inside the old dev set take their test_probs value (same document, same model: agree to ~1e-3)
            pick = lambda i, k: tp[i] if (k == "dev_in_test" or i not in dp) else dp[i]
            V[v] = {k: np.array([pick(i, k) for i, _ in S[k]]) for k in S}
        combos = {v: [v] for v in V}
        combos.update({"avg4": [v for v in V4 if v in V], "avg8": [v for v in V8 if v in V], "avg_all": list(V)})
        for name, members in combos.items():
            P = {k: np.mean([V[m][k] for m in members], axis=0) for k in S}
            probs[(run, name)] = P
            t = tune(P["dev_excl_test"], y)
            held = []
            for folds in folds_all:
                for f in range(5):
                    te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
                    held.append(float(f1_many(P["dev_excl_test"][te], y[te], np.array([tune(P["dev_excl_test"][tr], y[tr])]))[0]))
            r = {"run": run, "variant": name, "own_chunking": name == OWN.get(run), "tuned_threshold": t,
                 "heldout_mean": round(float(np.mean(held)), 4), "heldout_sd": round(float(np.std(held)), 4)}
            for tag, th in (("at0.5", 0.5), ("tuned", t)):
                for k in S:
                    r[f"{k}_{tag}"] = round(float(f1_many(P[k], Y[k], np.array([th]))[0]), 4)
            rows.append(r)
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    L = []
    for run in sorted({r["run"] for r in rows}):
        L += [f"### {run} (own training chunking: {OWN.get(run)})", "",
              "| Variant | dev_all @0.5 | dev_excl @0.5 | dev_in_test @0.5 | tuned thr (excl) | dev_all | dev_excl_test | dev_in_test (= test) | held-out excl |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted([r for r in rows if r["run"] == run], key=lambda r: -r["heldout_mean"]):
            mark = " *(own)*" if r["own_chunking"] else ""
            L.append(f"| {r['variant']}{mark} | {r['dev_all_at0.5']:.4f} | {r['dev_excl_test_at0.5']:.4f} | {r['dev_in_test_at0.5']:.4f} | "
                     f"{r['tuned_threshold']:.3f} | {r['dev_all_tuned']:.4f} | {r['dev_excl_test_tuned']:.4f} | {r['dev_in_test_tuned']:.4f} | "
                     f"{r['heldout_mean']:.4f} ± {r['heldout_sd']:.4f} |")
        L.append("")
    open(os.path.join(OUT, "results.md"), "w").write("\n".join(L) + "\n")
    np.savez(os.path.join(OUT, "tta_probs.npz"), **{f"{r}__{v}__{k}": P[k] for (r, v), P in probs.items() for k in S},
             **{f"ids__{k}": np.array([i for i, _ in S[k]]) for k in S})
    print("\n".join(L))


if __name__ == "__main__":
    main()
