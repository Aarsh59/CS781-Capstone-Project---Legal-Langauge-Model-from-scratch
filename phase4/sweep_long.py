#!/usr/bin/env python3
"""Threshold sweep + ensembles for the long runs, with three dev scores per candidate (plumbing; declare it).

  python sweep_long.py            -> ensembles/sweep_long/{results.md, results.json, <candidate>/...}

Members (dev_probs.json / test_probs.json): C3_long, E8_long (their best_dev_excl_test checkpoints) and, for the
ensembles, W1_B2_full_weighted. Weight grid = your earlier (C3, E8, W1) ratios with C3_long / E8_long in place of C3 / E8.
Two combination rules, as before:
  avg   : weighted mean of P(1), one threshold re-tuned on the mean
  bestth: every model centred at its own tuned threshold, predict 1 iff sum w_m (p_m - t_m) >= 0 (nothing re-tuned)
Selection (thresholds) on dev_excl_test only, grid 0.05..0.95 step 0.01, ties -> closest to 0.5; held-out check =
choose on a random half of dev_excl_test, score on the other half (500 splits).
Reported at the chosen threshold: dev_all (3,505 released rows), dev_excl_test (1,988 rows), dev_in_test (the 1,517 test
documents, labels from their copies in dev.json = the Codabench test score; reported, never used to select).
"""
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cjpe_data                                     # noqa: E402

RUNS = {"C3_long": "runs/C3_long/best_dev_excl_test", "E8_long": "runs/E8_long/best_dev_excl_test",
        "W1": "runs/W1_B2_full_weighted"}
GRID = [(0.50, 0.25, 0.25), (0.60, 0.20, 0.20), (0.70, 0.15, 0.15),
        (0.50, 0.00, 0.50), (0.75, 0.00, 0.25), (0.75, 0.25, 0.00)]                # (C3_long, E8_long, W1)
TH = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)


def load_json(path):
    for p in (path, path[:-5] + ".txt"):
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(path)


def best_th(p, y):
    f = [(float(t), cjpe_data.binary_scores(y, (p >= t).astype(int))["macro_f1"]) for t in TH]
    b = max(x for _, x in f)
    return min((t for t, x in f if x == b), key=lambda t: abs(t - 0.5))


def main():
    D = os.path.join(HERE, "data", "Phase_4_CJPE")
    dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test")
    cjpe_data.mark_dev_excl_test(dev, test)
    by_hash = {cjpe_data.norm_hash(d["text"]): d["label"] for d in dev}
    sets = {"dev_all": [(d["id"], d["label"]) for d in dev],
            "dev_excl_test": [(d["id"], d["label"]) for d in dev if not d["in_test"]],
            "dev_in_test": [(d["id"], by_hash[cjpe_data.norm_hash(d["text"])]) for d in test]}
    dev_p = {m: load_json(os.path.join(HERE, r, "dev_probs.json")) for m, r in RUNS.items()}
    test_p = {m: load_json(os.path.join(HERE, r, "test_probs.json")) for m, r in RUNS.items()}
    M = list(RUNS)

    def probs(m, key):                                  # P(1) of model m on a scoring set, in set order
        src = test_p[m] if key == "dev_in_test" else dev_p[m]
        return np.array([src[i] for i, _ in sets[key]])

    Y = {k: np.array([l for _, l in v]) for k, v in sets.items()}
    T_ind = {m: best_th(probs(m, "dev_excl_test"), Y["dev_excl_test"]) for m in M}

    cands = []
    for m in ("C3_long", "E8_long"):
        cands += [(f"{m}@0.5", {m: 1}, "fixed", 0.5), (f"{m}@tuned", {m: 1}, "avg", None)]
    for w in GRID:
        ww = {m: x for m, x in zip(M, w) if x}
        tag = "C3L{:.2f}_E8L{:.2f}_W1{:.2f}".format(*w)
        cands += [(f"ENSavg_{tag}", ww, "avg", None), (f"ENSbestth_{tag}", ww, "bestth", None)]

    rng = np.random.default_rng(0)
    out_root = os.path.join(HERE, "ensembles", "sweep_long"); os.makedirs(out_root, exist_ok=True)
    rows = []
    for name, w, rule, fixed in cands:
        s = sum(w.values())
        if rule == "bestth":
            score = lambda key: sum(x / s * (probs(m, key) - T_ind[m]) for m, x in w.items())
            th = 0.0
        else:
            score = lambda key: sum(x / s * probs(m, key) for m, x in w.items())
            th = fixed if fixed is not None else best_th(score("dev_excl_test"), Y["dev_excl_test"])
        res = {k: cjpe_data.binary_scores(Y[k], (score(k) >= th).astype(int)) for k in sets}
        # held-out check on dev_excl_test (re-choose the threshold on one half, score the other)
        pe, ye, n, held = score("dev_excl_test"), Y["dev_excl_test"], len(Y["dev_excl_test"]), []
        if rule == "avg" and fixed is None:
            for _ in range(500):
                idx = rng.permutation(n); a, b = idx[: n // 2], idx[n // 2:]
                held.append(cjpe_data.binary_scores(ye[b], (pe[b] >= best_th(pe[a], ye[a])).astype(int))["macro_f1"])
        tid = [i for i, _ in sets["dev_in_test"]]
        ts = sum(x / s * ((np.array([test_p[m][i] for i in tid]) - T_ind[m]) if rule == "bestth"
                          else np.array([test_p[m][i] for i in tid])) for m, x in w.items())
        pred = [{"id": i, "pred": int(v >= th)} for i, v in zip(tid, ts)]
        d = os.path.join(out_root, name.replace("@", "_")); os.makedirs(d, exist_ok=True)
        json.dump(pred, open(os.path.join(d, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(d, f"submission_{name.replace('@', '_')}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(d, "prediction.json"), arcname="prediction.json")
        cfg = {"weights": w, "rule": rule, "threshold": th, "member_thresholds": {m: T_ind[m] for m in w} if rule == "bestth" else None}
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
        rows.append({"candidate": name, **cfg, "heldout_excl_mean": round(float(np.mean(held)), 4) if held else None,
                     "test_share_pred_1": round(float(np.mean([p["pred"] for p in pred])), 4),
                     **{f"{k}_f1": r["macro_f1"] for k, r in res.items()},
                     "excl_P1": res["dev_excl_test"]["per_class"]["1"]["precision"], "excl_R1": res["dev_excl_test"]["per_class"]["1"]["recall"]})
    rows.sort(key=lambda r: -r["dev_excl_test_f1"])
    json.dump({"member_thresholds": T_ind, "candidates": rows}, open(os.path.join(out_root, "results.json"), "w"), indent=1)
    L = [f"Individual tuned thresholds (dev_excl_test): {T_ind}", "",
         "| Candidate | Rule | Threshold | dev_all F1 | dev_excl_test F1 (selection) | held-out excl | dev_in_test F1 (= test) | P1 / R1 (excl) | share pred 1 (test) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['candidate']} | {r['rule']} | {r['threshold']} | {r['dev_all_f1']:.4f} | {r['dev_excl_test_f1']:.4f} | "
                 f"{r['heldout_excl_mean'] if r['heldout_excl_mean'] is not None else '-'} | {r['dev_in_test_f1']:.4f} | "
                 f"{r['excl_P1']:.3f} / {r['excl_R1']:.3f} | {r['test_share_pred_1']} |")
    open(os.path.join(out_root, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
