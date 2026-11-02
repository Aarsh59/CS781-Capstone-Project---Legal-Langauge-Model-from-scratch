#!/usr/bin/env python3
"""Phase 4 CJPE — dual-split ensemble optimisation (plumbing; declare it). Your spec of 2026-10-20.

  python dual_split_sweep.py            (seed 0)  ->  ensembles/dual_split_sweep/

Members (saved P(class 1); no retraining): C3_B2_overlap, E8_KD_B2s12, W1, C3_long, E8_long_A (dev-selected checkpoint).
Splits, matched by document id, duplicates removed (dev.json repeats each of the 994 dev_excl_test documents twice):
  dev_excl_test : 994 distinct dev documents whose text is not in test.json                      (Track A objective)
  dev_in_test   : the 1,517 test documents, labels from their copies in dev.json, scores from test_probs.json
                  (OVERLAP-INFORMED: this is the test set; Track B objective)
  dev_all       : the 3,505 released dev rows (supplementary diagnostic only)
Ensemble: p = sum_i w_i p_i (w >= 0, sum 1); predict 1 iff p >= threshold.
Search per (configuration, track): the spec's initial weights + N_DIRICHLET random Dirichlet(1) weight vectors
(seed fixed) + coordinate refinement (move mass between member pairs, steps 0.05 / 0.02 / 0.01 until no gain);
threshold for every weight vector: 0.20..0.60 step 0.01, then the best +/- 0.01 step 0.001 (ties -> closest to 0.5).
Held-out (Track A): stratified 5-fold x 3 repeats over the 994 dev_excl_test documents; weights + threshold tuned on
the 4 training folds only (smaller budget), scored on the held-out fold.
"""
import csv
import itertools
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cjpe_data                                      # noqa: E402

SEED = 0
N_DIRICHLET, N_DIRICHLET_HELDOUT = 400, 120
MEMBERS = {"C3": "runs/C3_B2_overlap", "E8": "runs/E8_KD_B2s12", "W1": "runs/W1_B2_full_weighted",
           "C3_long": "runs/C3_long/best_dev_excl_test", "E8_long_A": "runs/E8_long/best_dev_excl_test"}
CONFIGS = {
    "ENS01": {"C3": 0.50, "E8": 0.25, "W1": 0.25}, "ENS02": {"C3": 0.75, "E8": 0.25}, "ENS03": {"C3": 0.50, "W1": 0.50},
    "ENS04": {"C3": 0.75, "C3_long": 0.25}, "ENS05": {"C3": 0.75, "E8_long_A": 0.25},
    "ENS06": {"C3": 0.50, "E8": 0.25, "C3_long": 0.25}, "ENS07": {"C3": 0.50, "E8": 0.25, "E8_long_A": 0.25},
    "ENS08": {"C3": 0.50, "W1": 0.25, "C3_long": 0.25}, "ENS09": {"C3": 0.40, "E8": 0.25, "W1": 0.15, "C3_long": 0.20},
    "ENS10": {"C3": 0.40, "E8": 0.25, "W1": 0.15, "E8_long_A": 0.20},
}
COARSE = np.round(np.arange(0.20, 0.60 + 1e-9, 0.01), 2)
OUT = os.path.join(HERE, "ensembles", "dual_split_sweep")


# ------------------------------------------------------------------ metrics (vectorised over thresholds)
def macro_f1_many(p, y, ths):
    """p (N,), y (N,) in {0,1}, ths (T,) -> macro-F1 (T,)."""
    pred = p[None, :] >= ths[:, None]
    y1 = (y == 1)[None, :]
    tp1 = (pred & y1).sum(1); fp1 = (pred & ~y1).sum(1); fn1 = (~pred & y1).sum(1)
    tp0 = (~pred & ~y1).sum(1); fp0 = fn1; fn0 = fp1
    f = lambda tp, fp, fn: np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return (f(tp1, fp1, fn1) + f(tp0, fp0, fn0)) / 2


def best_threshold(p, y):
    f = macro_f1_many(p, y, COARSE)
    b = f.max(); tc = min(COARSE[f == b], key=lambda t: abs(t - 0.5))
    fine = np.round(np.arange(tc - 0.01, tc + 0.01 + 1e-9, 0.001), 3)
    ff = macro_f1_many(p, y, fine)
    bf = ff.max(); t = float(min(fine[ff == bf], key=lambda t: abs(t - tc)))
    return t, float(bf)


def full_scores(p, y, t):
    s = cjpe_data.binary_scores(y, (p >= t).astype(int))
    return {"macro_f1": s["macro_f1"], "per_class": s["per_class"], "confusion_gold_x_pred": s["confusion_gold_x_pred"],
            "pred_share_1": round(float((p >= t).mean()), 4), "n": s["n"]}


# ------------------------------------------------------------------ weight search
def search(P, y, names, init, rng, n_dir):
    """P: {name: probs}; returns (weights dict, threshold, in-split macro-F1, n_evaluated)."""
    k = len(names)
    X = np.stack([P[m] for m in names])                       # (k, N)
    cache = {}

    def score(w):
        key = tuple(np.round(w, 6))
        if key not in cache:
            cache[key] = best_threshold(w @ X, y)
        return cache[key]

    cands = [np.array([init[m] for m in names], float)] if init else []
    cands += [np.eye(k)[i] for i in range(k)] if k > 1 else []
    cands += list(rng.dirichlet(np.ones(k), n_dir)) if k > 1 else [np.ones(1)]
    best_w, (best_t, best_f) = None, (None, -1.0)
    for w in cands:
        w = w / w.sum(); t, f = score(w)
        if f > best_f:
            best_w, best_t, best_f = w, t, f
    if k > 1:                                                 # coordinate refinement: move mass i -> j
        for step in (0.05, 0.02, 0.01):
            improved = True
            while improved:
                improved = False
                for i, j in itertools.permutations(range(k), 2):
                    if best_w[i] < step - 1e-12:
                        continue
                    w = best_w.copy(); w[i] -= step; w[j] += step
                    t, f = score(w)
                    if f > best_f + 1e-12:
                        best_w, best_t, best_f, improved = w, t, f, True
    return {m: round(float(x), 6) for m, x in zip(names, best_w)}, best_t, best_f, len(cache)


def main():
    os.makedirs(OUT, exist_ok=True)
    D = os.path.join(HERE, "data", "Phase_4_CJPE")
    dev, test = cjpe_data.load_split(D, "dev"), cjpe_data.load_split(D, "test")
    cjpe_data.mark_dev_excl_test(dev, test)
    lab_by_hash = {cjpe_data.norm_hash(d["text"]): d["label"] for d in dev}
    # ---- splits by id, duplicates removed
    excl, seen = [], set()
    for d in dev:
        if not d["in_test"] and d["id"] not in seen:
            seen.add(d["id"]); excl.append((d["id"], d["label"]))
    intest = [(d["id"], lab_by_hash[cjpe_data.norm_hash(d["text"])]) for d in test]
    allrows = [(d["id"], d["label"]) for d in dev]
    SPL = {"dev_excl_test": excl, "dev_in_test": intest, "dev_all": allrows}
    Y = {k: np.array([l for _, l in v]) for k, v in SPL.items()}
    # ---- preflight
    pre = {"splits": {k: {"n": len(v), "unique_ids": len({i for i, _ in v}), "label_share_1": round(float(Y[k].mean()), 4)} for k, v in SPL.items()}}
    assert pre["splits"]["dev_excl_test"]["n"] == pre["splits"]["dev_excl_test"]["unique_ids"]
    assert pre["splits"]["dev_in_test"]["n"] == pre["splits"]["dev_in_test"]["unique_ids"] == len(test)
    assert not ({i for i, _ in excl} & {i for i, _ in intest}), "dev_excl_test and dev_in_test overlap"
    P, pre["members"] = {}, {}
    for m, rel in MEMBERS.items():
        d = os.path.join(HERE, rel)
        dp, tp = json.load(open(os.path.join(d, "dev_probs.json"))), json.load(open(os.path.join(d, "test_probs.json")))
        P[m] = {}
        for k, rows in SPL.items():
            src = tp if k == "dev_in_test" else dp
            miss = [i for i, _ in rows if i not in src]
            assert not miss, f"{m}: {len(miss)} ids missing in {'test' if k == 'dev_in_test' else 'dev'}_probs.json for {k}"
            a = np.array([src[i] for i, _ in rows], float)
            assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all(), f"{m}: invalid probabilities in {k}"
            P[m][k] = a
        assert set(tp) == {d["id"] for d in test}, f"{m}: test_probs ids differ from test.json"
        consist = max(abs(dp[i] - tp[i]) for i, _ in intest)
        pre["members"][m] = {"path": rel, "dev_ids": len(dp), "test_ids": len(tp), "max_abs_dev_vs_test_prob_on_overlap": round(consist, 6)}
    json.dump(pre, open(os.path.join(OUT, "preflight.json"), "w"), indent=1)
    print("PREFLIGHT OK:", json.dumps(pre, indent=1))

    rng = np.random.default_rng(SEED)
    results = []
    # ---- individual baselines + ensembles, both tracks
    jobs = [(f"SINGLE_{m}", {m: 1.0}) for m in MEMBERS] + list(CONFIGS.items())
    for cid, init in jobs:
        names = list(init)
        rec = {"id": cid, "members": names, "initial_weights": init, "tracks": {}}
        for track, split in (("A", "dev_excl_test"), ("B", "dev_in_test")):
            w, t, f, nev = search({m: P[m][split] for m in names}, Y[split], names, init, rng, N_DIRICHLET)
            ev = {}
            for k in SPL:
                p = sum(w[m] * P[m][k] for m in names)
                ev[k] = full_scores(p, Y[k], t)
            rec["tracks"][track] = {"optimised_on": split, "weights": w, "threshold": t, "in_split_macro_f1": f,
                                    "weight_vectors_evaluated": nev, "eval": ev}
        results.append(rec)
        print(f"{cid:18s} A: w={rec['tracks']['A']['weights']} t={rec['tracks']['A']['threshold']} "
              f"excl={rec['tracks']['A']['eval']['dev_excl_test']['macro_f1']:.4f} in={rec['tracks']['A']['eval']['dev_in_test']['macro_f1']:.4f} | "
              f"B: t={rec['tracks']['B']['threshold']} in={rec['tracks']['B']['eval']['dev_in_test']['macro_f1']:.4f} "
              f"excl={rec['tracks']['B']['eval']['dev_excl_test']['macro_f1']:.4f}", flush=True)

    # ---- held-out checks (Track A): stratified 5-fold x 3 repeats on the 994 documents
    y = Y["dev_excl_test"]
    held = {}
    hrng = np.random.default_rng(SEED + 1)
    for rec in results:
        names = rec["members"]; scores = []
        for rep in range(3):
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            hrng.shuffle(idx0); hrng.shuffle(idx1)
            folds = [np.concatenate([idx0[f::5], idx1[f::5]]) for f in range(5)]
            for f in range(5):
                te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
                w, t, _, _ = search({m: P[m]["dev_excl_test"][tr] for m in names}, y[tr], names, rec["initial_weights"], hrng, N_DIRICHLET_HELDOUT)
                p = sum(w[m] * P[m]["dev_excl_test"][te] for m in names)
                scores.append(float(macro_f1_many(p, y[te], np.array([t]))[0]))
        held[rec["id"]] = {"mean": round(float(np.mean(scores)), 4), "sd": round(float(np.std(scores)), 4), "folds": len(scores)}
        rec["heldout_track_A"] = held[rec["id"]]
        print(f"held-out {rec['id']:18s} {held[rec['id']]['mean']:.4f} ± {held[rec['id']]['sd']:.4f}", flush=True)

    # ---- outputs
    def row(rec, tr):
        T = rec["tracks"][tr]
        return {"id": rec["id"], "track": tr, "optimised_on": T["optimised_on"], "weights": json.dumps(T["weights"]),
                "threshold": T["threshold"], "in_split_macro_f1": round(T["in_split_macro_f1"], 4),
                "dev_excl_test_f1": T["eval"]["dev_excl_test"]["macro_f1"], "dev_in_test_f1": T["eval"]["dev_in_test"]["macro_f1"],
                "dev_all_f1": T["eval"]["dev_all"]["macro_f1"],
                "excl_P1": T["eval"]["dev_excl_test"]["per_class"]["1"]["precision"], "excl_R1": T["eval"]["dev_excl_test"]["per_class"]["1"]["recall"],
                "in_P1": T["eval"]["dev_in_test"]["per_class"]["1"]["precision"], "in_R1": T["eval"]["dev_in_test"]["per_class"]["1"]["recall"],
                "test_pred_share_1": T["eval"]["dev_in_test"]["pred_share_1"],
                "heldout_A_mean": rec["heldout_track_A"]["mean"], "heldout_A_sd": rec["heldout_track_A"]["sd"]}
    rows = [row(r, t) for r in results for t in ("A", "B")]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(OUT, "cross_evaluation.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["id", "config_from_track", "weights", "threshold", "eval_dev_excl_test", "eval_dev_in_test", "eval_dev_all"])
        for r in results:
            for t in ("A", "B"):
                T = r["tracks"][t]
                wr.writerow([r["id"], t, json.dumps(T["weights"]), T["threshold"], T["eval"]["dev_excl_test"]["macro_f1"],
                             T["eval"]["dev_in_test"]["macro_f1"], T["eval"]["dev_all"]["macro_f1"]])
    json.dump(held, open(os.path.join(OUT, "heldout_checks.json"), "w"), indent=1)
    hm = ["| ID | members | held-out Track A macro-F1 (mean ± sd, 15 folds) | in-split Track A |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["heldout_track_A"]["mean"]):
        hm.append(f"| {r['id']} | {'+'.join(r['members'])} | {r['heldout_track_A']['mean']:.4f} ± {r['heldout_track_A']['sd']:.4f} | {r['tracks']['A']['in_split_macro_f1']:.4f} |")
    open(os.path.join(OUT, "heldout_checks.md"), "w").write("\n".join(hm) + "\n")
    bestA = max(results, key=lambda r: r["tracks"]["A"]["in_split_macro_f1"])
    bestB = max(results, key=lambda r: r["tracks"]["B"]["in_split_macro_f1"])
    robust = max(results, key=lambda r: r["heldout_track_A"]["mean"])
    singles = [r for r in results if r["id"].startswith("SINGLE_")]
    best_single = max(singles, key=lambda r: r["heldout_track_A"]["mean"])
    json.dump({"id": bestA["id"], **bestA["tracks"]["A"], "heldout": bestA["heldout_track_A"]}, open(os.path.join(OUT, "best_track_A_config.json"), "w"), indent=1)
    json.dump({"id": bestB["id"], "label": "OVERLAP-INFORMED (optimised on the test documents)", **bestB["tracks"]["B"]},
              open(os.path.join(OUT, "best_track_B_config.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "ensemble_probabilities.npz"),
             **{f"{m}__{k}": P[m][k] for m in P for k in SPL}, **{f"ids__{k}": np.array([i for i, _ in v]) for k, v in SPL.items()},
             **{f"labels__{k}": Y[k] for k in SPL})
    # submissions for the shortlist
    test_ids = [i for i, _ in intest]
    short = {"best_single_A": (best_single, "A"), "best_track_A": (bestA, "A"), "best_robust_A": (robust, "A"), "best_track_B_overlap": (bestB, "B")}
    subs = {}
    for tag, (r, tr) in short.items():
        T = r["tracks"][tr]
        p = sum(T["weights"][m] * P[m]["dev_in_test"] for m in r["members"])
        pred = [{"id": i, "pred": int(v >= T["threshold"])} for i, v in zip(test_ids, p)]
        assert len(pred) == len(test) and len({x["id"] for x in pred}) == len(pred) and all(x["pred"] in (0, 1) for x in pred)
        d = os.path.join(OUT, f"{tag}_{r['id']}"); os.makedirs(d, exist_ok=True)
        json.dump(pred, open(os.path.join(d, "prediction.json"), "w"))
        z = os.path.join(d, f"submission_{tag}_{r['id']}.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(d, "prediction.json"), arcname="prediction.json")
        json.dump({"id": r["id"], "track": tr, "weights": T["weights"], "threshold": T["threshold"]}, open(os.path.join(d, "config.json"), "w"), indent=1)
        subs[tag] = {"id": r["id"], "track": tr, "zip": z}
    json.dump({"seed": SEED, "n_dirichlet": N_DIRICHLET, "n_dirichlet_heldout": N_DIRICHLET_HELDOUT, "members": MEMBERS,
               "preflight": pre, "results": results, "shortlist": subs}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, "command.txt"), "w").write("cd ~/capstone/phase4 && python3 dual_split_sweep.py   # seed 0, deterministic\n")
    # REPORT
    def line(r, tr):
        T = r["tracks"][tr]; e = T["eval"]
        return (f"| {r['id']} | {'+'.join(r['members'])} | {tr} | {json.dumps(T['weights'])} | {T['threshold']} | "
                f"{e['dev_excl_test']['macro_f1']:.4f} | {e['dev_in_test']['macro_f1']:.4f} | {e['dev_all']['macro_f1']:.4f} | "
                f"{r['heldout_track_A']['mean']:.4f} ± {r['heldout_track_A']['sd']:.4f} |")
    R = ["# Phase 4 — dual-split ensemble optimisation", "",
         "Validation numbers only. dev_in_test = the 1,517 test documents (labels from dev.json): Track B and every dev_in_test "
         "column are OVERLAP-INFORMED, i.e. equal to / optimised on the test score; not an independent estimate.",
         f"dev_excl_test = 994 distinct documents (dev.json lists each twice; earlier sweeps' held-out checks split rows and were therefore optimistic).",
         f"Search: seed {SEED}, {N_DIRICHLET} Dirichlet weight vectors + initial + one-hot + coordinate refinement per (config, track); "
         "thresholds 0.20-0.60 step 0.01, refined +/-0.01 step 0.001.", "",
         "| ID | Members | Track | Weights | Threshold | dev_excl_test | dev_in_test (= test) | dev_all | held-out Track A |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["heldout_track_A"]["mean"]):
        R += [line(r, "A"), line(r, "B")]
    R += ["", "## Shortlist (by the spec's criteria)",
          f"- Best single model (held-out Track A): {best_single['id']} — held-out {best_single['heldout_track_A']['mean']:.4f}",
          f"- Best Track A in-split: {bestA['id']} — dev_excl_test {bestA['tracks']['A']['in_split_macro_f1']:.4f}, held-out {bestA['heldout_track_A']['mean']:.4f}",
          f"- Best robust (held-out Track A): {robust['id']} — {robust['heldout_track_A']['mean']:.4f} ± {robust['heldout_track_A']['sd']:.4f}",
          f"- Best Track B (OVERLAP-INFORMED): {bestB['id']} — dev_in_test {bestB['tracks']['B']['in_split_macro_f1']:.4f}",
          "", "Submissions: " + json.dumps(subs, indent=1),
          "", "Interpretation / final choice: yours (course rule 9)."]
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(R) + "\n")
    print("\n".join(R[-12:]))


if __name__ == "__main__":
    main()
