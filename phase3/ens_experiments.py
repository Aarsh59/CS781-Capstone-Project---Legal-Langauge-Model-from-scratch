#!/usr/bin/env python3
"""Phase 3 ensemble experiments on the seven campaign-2 runs (plumbing/evaluation; declare it).

Experiment design (routing rules, bias classes, grid, validation protocol) is the user's; this script only
executes it. Nothing existing is modified: inputs are read from the downloaded Kaggle output, results go to
--out (default ens_exp_out/).

  python ens_experiments.py --runs ~/kaggle_outputs/notebooka3fd7ebab3/phase3/runs \
                            --ens  ~/kaggle_outputs/notebooka3fd7ebab3/phase3/ensembles --data data

Definitions
  hard vote        rr_ensemble.vote (weighted majority, ties -> earliest member), members in the order listed.
  routing rule     "label L <- model M": wherever M predicts L, the prediction becomes L; elsewhere the base
                   prediction is kept. Several rules are applied in the listed order (a later rule wins a conflict).
  bias             argmax_c( log p(c) + b_c ) with b_c from the grid for the four bias classes, 0 for the rest.
                   Probability sets: each single model's own probabilities, and the mean log-probability of the
                   four runs trained with the same inv_sqrt cross-entropy (B_lora_crf is never used for probs).
  cross-domain     a configuration is selected on CL dev and scored on IT dev, and vice versa; "robust" = the
                   selected configuration beats its own unmodified baseline on the held-out domain in BOTH
                   directions.
Metric: rr_metrics.rr_scores (unchanged), as used by train_rr.score_domains.
"""
import argparse
import csv
import itertools
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_data                                   # noqa: E402
from rr_ensemble import vote                     # noqa: E402
from rr_metrics import rr_scores                 # noqa: E402

RUNS = ["B_lora_crf", "B_lora_ctx_wide", "B_lora_seed123", "B_lora_unweighted", "B_rslora_r16_lr2e4",
        "B_rslora_r16_lr5e5", "KD_seed123"]
CE_WEIGHTING = {                                  # how each run's training loss weighted the classes
    "B_lora_ctx_wide": "inv_sqrt CE", "B_lora_seed123": "inv_sqrt CE", "B_rslora_r16_lr2e4": "inv_sqrt CE",
    "B_rslora_r16_lr5e5": "inv_sqrt CE", "B_lora_unweighted": "unweighted CE",
    "KD_seed123": "KD (teacher inv_sqrt) + unweighted CE", "B_lora_crf": "CRF NLL (no class weights)"}
ROUTE_CANDIDATES = [                              # user's candidates: (label, [models])
    ("ArgumentPetitioner", ["B_lora_ctx_wide"]),
    ("ArgumentRespondent", ["B_rslora_r16_lr2e4", "KD_seed123"]),
    ("RulingByLowerCourt", ["B_rslora_r16_lr5e5", "B_rslora_r16_lr2e4"]),
    ("PrecedentNotReliedUpon", ["B_lora_crf", "B_lora_ctx_wide"]),
]
BIAS_CLASSES = ["ArgumentPetitioner", "ArgumentRespondent", "RulingByLowerCourt", "RatioOfTheDecision"]
GRID = [-0.6, -0.3, 0.0, 0.3, 0.6]
NO_TUNE = {"Dissent", "PrecedentOverruled"}       # no gold dev examples: never biased


def fail(msg):
    sys.exit(f"ALIGNMENT/METRIC CHECK FAILED: {msg}")


def split_docs(flat, docs, nsent):
    out, k = {}, 0
    for d in docs:
        out[d] = flat[k:k + nsent[d]]
        k += nsent[d]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/kaggle_outputs/notebooka3fd7ebab3/phase3/runs"))
    ap.add_argument("--ens", default=os.path.expanduser("~/kaggle_outputs/notebooka3fd7ebab3/phase3/ensembles"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ens_exp_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------- 1. load + verify
    labels = rr_data.load_labels(a.data)
    L, li = len(labels), {l: i for i, l in enumerate(labels)}
    dev = {D: rr_data.load_domain(a.data, D, "dev") for D in ("CL", "IT")}
    tst = {D: rr_data.load_domain(a.data, D, "test") for D in ("CL", "IT")}
    dom_of = {d["id"]: D for D in dev for d in dev[D]}
    gold = {d["id"]: d["labels"] for D in dev for d in dev[D]}
    docs = list(gold)                                                   # fixed document order
    tdocs = [d["id"] for D in tst for d in tst[D]]
    nsent = {d: len(gold[d]) for d in docs}
    tnsent = {d["id"]: len(d["sentences"]) for D in tst for d in tst[D]}
    gold_D = {D: {d["id"]: d["labels"] for d in dev[D]} for D in dev}

    pred, tpred, logp, tlogp = {}, {}, {}, {}
    for n in RUNS:
        rd = os.path.join(a.runs, n)
        if json.load(open(os.path.join(rd, "labels.json"))) != labels:
            fail(f"{n}: labels.json order differs from label_vocab")
        p = json.load(open(os.path.join(rd, "dev_predictions.json")))
        t = json.load(open(os.path.join(rd, "prediction.json")))
        for name, P, ref in (("dev_predictions", p, nsent), ("prediction", t, tnsent)):
            if set(P) != set(ref) or any(len(P[d]) != ref[d] for d in ref):
                fail(f"{n}/{name}.json: document ids or sentence counts differ from the data")
            if not set(l for v in P.values() for l in v) <= set(labels):
                fail(f"{n}/{name}.json: unknown label")
        pred[n], tpred[n] = p, t
        for fn, ref, store in (("dev_probs.json", nsent, logp), ("test_probs.json", tnsent, tlogp)):
            pr = json.load(open(os.path.join(rd, fn)))
            if pr["labels"] != labels:
                fail(f"{n}/{fn}: class order differs")
            if set(pr["probs"]) != set(ref) or any(len(pr["probs"][d]) != ref[d] for d in ref):
                fail(f"{n}/{fn}: document ids or sentence counts differ")
            order = docs if ref is nsent else tdocs
            arr = np.array([row for d in order for row in pr["probs"][d]], dtype=np.float64)
            if arr.shape[1] != L or not np.isfinite(arr).all() or np.abs(arr.sum(1) - 1).max() > 1e-4:
                fail(f"{n}/{fn}: probabilities not finite / not 13 columns / rows not summing to 1")
            store[n] = np.log(np.clip(arr, 1e-12, 1.0))
        if pr["decode"] == "argmax":                                    # probs must reproduce the predictions
            flat = [l for d in docs for l in pred[n][d]]
            if any(labels[i] != l for i, l in zip(logp[n].argmax(1), flat)):
                fail(f"{n}: argmax of dev_probs does not reproduce dev_predictions")
    ens = {}
    for e in sorted(os.listdir(a.ens)) if os.path.isdir(a.ens) else []:
        f = os.path.join(a.ens, e, "dev_predictions.json")
        if os.path.exists(f):
            p = json.load(open(f))
            if set(p) != set(nsent) or any(len(p[d]) != nsent[d] for d in nsent):
                fail(f"ensemble {e}: dev ids / counts differ")
            tp = json.load(open(os.path.join(a.ens, e, "prediction.json")))
            ens[e] = (p, tp, json.load(open(os.path.join(a.ens, e, "summary.json"))))
    print(f"verified: {len(RUNS)} runs (dev + test predictions and probabilities), {len(ens)} ensembles; "
          f"dev {len(docs)} docs / {sum(nsent.values())} sentences; class order = label_vocab")

    def score(P):
        s = rr_scores(gold, P, labels)
        r = {"all": s["macro_f1"], "CL": rr_scores(gold_D["CL"], {d: P[d] for d in gold_D["CL"]}, labels)["macro_f1"],
             "IT": rr_scores(gold_D["IT"], {d: P[d] for d in gold_D["IT"]}, labels)["macro_f1"], "per": s["per_label"]}
        return r

    rows = []

    def record(exp, base, config, P, extra=None):
        s = score(P)
        row = {"experiment": exp, "base": base, "config": config, "dev_all": s["all"], "dev_CL": s["CL"], "dev_IT": s["IT"]}
        for l in labels:
            row[f"F1_{l}"] = s["per"].get(l, {}).get("f1", "")
        row.update(extra or {})
        rows.append(row)
        return row

    # ---------------- 2. baselines (+ reproduction check against the saved summaries)
    print("\nbaselines (dev macro-F1 all / CL / IT; saved value in brackets)")
    for n in ["KD_seed123", "B_lora_ctx_wide", "B_rslora_r16_lr2e4"] + [r for r in RUNS if r not in
                                                                      ("KD_seed123", "B_lora_ctx_wide", "B_rslora_r16_lr2e4")]:
        r = record("baseline", n, "single run", pred[n])
        saved = json.load(open(os.path.join(a.runs, n, "summary.json")))
        sv = (saved["dev_macro_f1"], saved["dev_CL_macro_f1"], saved["dev_IT_macro_f1"])
        if (round(r["dev_all"], 4), round(r["dev_CL"], 4), round(r["dev_IT"], 4)) != tuple(round(x, 4) for x in sv):
            fail(f"{n}: recomputed dev scores {r['dev_all'], r['dev_CL'], r['dev_IT']} != saved {sv}")
        print(f"  {n:22s} {r['dev_all']:.4f} / {r['dev_CL']:.4f} / {r['dev_IT']:.4f}   [{sv[0]} / {sv[1]} / {sv[2]}]")
    v7 = vote([pred[n] for n in RUNS], [1.0] * len(RUNS))
    tv7 = vote([tpred[n] for n in RUNS], [1.0] * len(RUNS))
    r = record("baseline", "VOTE7", "hard vote, 7 runs, members in RUNS order", v7)
    print(f"  {'VOTE7 (hard vote)':22s} {r['dev_all']:.4f} / {r['dev_CL']:.4f} / {r['dev_IT']:.4f}")
    bases = {"KD_seed123": (pred["KD_seed123"], tpred["KD_seed123"]), "VOTE7": (v7, tv7)}
    if "ENS_top5" in ens:
        p, tp, sm = ens["ENS_top5"]
        r = record("baseline", "ENS_top5", "existing ensemble (campaign-1 members)", p)
        if round(r["dev_all"], 4) != round(sm["dev_macro_f1"], 4):
            fail(f"ENS_top5: recomputed {r['dev_all']} != saved {sm['dev_macro_f1']}")
        print(f"  {'ENS_top5':22s} {r['dev_all']:.4f} / {r['dev_CL']:.4f} / {r['dev_IT']:.4f}   [{sm['dev_macro_f1']}]")
        bases["ENS_top5"] = (p, tp)
    base_score = {b: next(x for x in rows if x["experiment"] == "baseline" and x["base"] == b) for b in bases}

    # ---------------- 3. experiment A: routing on hard predictions
    def route(base_p, rules, src):
        out = {d: list(v) for d, v in base_p.items()}
        for lab, m in rules:
            for d in out:
                for i, x in enumerate(src[m][d]):
                    if x == lab:
                        out[d][i] = lab
        return out

    single = [((lab, m),) for lab, ms in ROUTE_CANDIDATES for m in ms]
    combos = [tuple(zip([c[0] for c in ROUTE_CANDIDATES], ch)) for ch in itertools.product(*[c[1] for c in ROUTE_CANDIDATES])]
    rule_sets = single + combos
    route_rows = {b: [] for b in bases}
    for b, (bp, _) in bases.items():
        for rs in rule_sets:
            cfg = "; ".join(f"{lab}<-{m}" for lab, m in rs)
            route_rows[b].append((rs, record("A_routing", b, cfg, route(bp, rs, pred))))

    # ---------------- 4. experiment B: class-specific log-probability biases
    calib = {}
    flatg = np.array([li[l] for d in docs for l in gold[d]])
    for n in RUNS:
        if n == "B_lora_crf":
            continue
        p = np.exp(logp[n]); conf = p.max(1); ok = p.argmax(1) == flatg
        bins = np.minimum((conf * 10).astype(int), 9)
        ece = sum(abs(conf[bins == k].mean() - ok[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
        calib[n] = dict(weighting=CE_WEIGHTING[n], mean_conf=conf.mean(), accuracy=ok.mean(), ECE=ece,
                        conf_correct=conf[ok].mean(), conf_wrong=conf[~ok].mean())
    inv = [n for n in RUNS if CE_WEIGHTING[n] == "inv_sqrt CE"]
    psets = {n: (logp[n], tlogp[n]) for n in ["KD_seed123", "B_lora_ctx_wide", "B_rslora_r16_lr2e4"]}
    psets["MEANLOG_inv_sqrt4"] = (np.mean([logp[n] for n in inv], 0), np.mean([tlogp[n] for n in inv], 0))
    bidx = [li[c] for c in BIAS_CLASSES]
    assert not (set(BIAS_CLASSES) & NO_TUNE)
    bias_rows = {s: [] for s in psets}
    for s, (lp, _) in psets.items():
        for bvals in itertools.product(GRID, repeat=len(BIAS_CLASSES)):
            b = np.zeros(L); b[bidx] = bvals
            P = split_docs([labels[i] for i in (lp + b).argmax(1)], docs, nsent)
            cfg = ", ".join(f"{c}={v:+.1f}" for c, v in zip(BIAS_CLASSES, bvals))
            extra = {f"bias_{c}": v for c, v in zip(BIAS_CLASSES, bvals)}
            bias_rows[s].append((bvals, record("B_bias", s, cfg, P, extra)))
        base_score[s] = next(r for bv, r in bias_rows[s] if not any(bv))       # zero bias = the unmodified model

    # ---------------- 5. cross-domain validation
    def xval(cands, bname):
        b = base_score[bname]
        best_CL = max(cands, key=lambda c: c[1]["dev_CL"]); best_IT = max(cands, key=lambda c: c[1]["dev_IT"])
        return {"base": bname, "base_all": b["dev_all"], "base_CL": b["dev_CL"], "base_IT": b["dev_IT"],
                "CL->IT config": best_CL[1]["config"], "CL->IT held-out IT": best_CL[1]["dev_IT"], "IT gain": best_CL[1]["dev_IT"] - b["dev_IT"],
                "IT->CL config": best_IT[1]["config"], "IT->CL held-out CL": best_IT[1]["dev_CL"], "CL gain": best_IT[1]["dev_CL"] - b["dev_CL"],
                "robust (both gains > 0)": (best_CL[1]["dev_IT"] > b["dev_IT"]) and (best_IT[1]["dev_CL"] > b["dev_CL"]),
                "configs improving BOTH CL and IT": sum(c[1]["dev_CL"] > b["dev_CL"] and c[1]["dev_IT"] > b["dev_IT"] for c in cands),
                "configs tested": len(cands)}
    xv = [dict(experiment="A_routing", **xval(route_rows[b], b)) for b in route_rows] + \
         [dict(experiment="B_bias", **xval(bias_rows[s], s)) for s in bias_rows]

    # ---------------- 6. outputs
    keys = list(rows[0]) + [k for k in {k for r in rows for k in r} if k not in rows[0]]
    with open(os.path.join(a.out, "all_configs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "cross_domain.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(xv[0])); w.writeheader(); w.writerows(xv)
    with open(os.path.join(a.out, "calibration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run"] + list(next(iter(calib.values())))); w.writeheader()
        for n, c in calib.items():
            w.writerow({"run": n, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}})

    print("\ncalibration of the six probability-producing runs (dev)")
    for n, c in calib.items():
        print(f"  {n:22s} {c['weighting']:38s} mean conf {c['mean_conf']:.3f} acc {c['accuracy']:.3f} ECE {c['ECE']:.3f}")
    print("\ncross-domain validation (selected on one domain, scored on the other)")
    for x in xv:
        print(f"  {x['experiment']:9s} base {x['base']:18s} base CL/IT {x['base_CL']:.4f}/{x['base_IT']:.4f} | "
              f"CL->IT {x['CL->IT held-out IT']:.4f} ({x['IT gain']:+.4f}) | IT->CL {x['IT->CL held-out CL']:.4f} ({x['CL gain']:+.4f}) | "
              f"robust {x['robust (both gains > 0)']} | improve both: {x['configs improving BOTH CL and IT']}/{x['configs tested']}")

    # best candidate: among configs that beat their own baseline on BOTH CL and IT, highest combined dev
    cand = []
    for b, lst in route_rows.items():
        cand += [("A_routing", b, rs, r) for rs, r in lst if r["dev_CL"] > base_score[b]["dev_CL"] and r["dev_IT"] > base_score[b]["dev_IT"]]
    for s, lst in bias_rows.items():
        cand += [("B_bias", s, bv, r) for bv, r in lst if r["dev_CL"] > base_score[s]["dev_CL"] and r["dev_IT"] > base_score[s]["dev_IT"]]
    best_baseline = max((r for r in rows if r["experiment"] == "baseline"), key=lambda r: r["dev_all"])
    print(f"\nbest unmodified baseline on combined dev: {best_baseline['base']} {best_baseline['dev_all']:.4f}")
    if not cand:
        print("no configuration improves its own baseline on both CL and IT dev -> no robust improvement found; no candidate written")
        return
    exp, b, spec, r = max(cand, key=lambda c: c[3]["dev_all"])
    print(f"best both-domain candidate: {exp} | base {b} | {r['config']}\n   dev all/CL/IT {r['dev_all']:.4f}/{r['dev_CL']:.4f}/{r['dev_IT']:.4f} "
          f"vs its baseline {base_score[b]['dev_all']:.4f}/{base_score[b]['dev_CL']:.4f}/{base_score[b]['dev_IT']:.4f} "
          f"(gain {r['dev_all'] - base_score[b]['dev_all']:+.4f}) | vs best baseline {best_baseline['dev_all']:.4f} ({r['dev_all'] - best_baseline['dev_all']:+.4f})")
    if exp == "A_routing":
        dpred, tp = route(bases[b][0], spec, pred), route(bases[b][1], spec, tpred)
    else:
        bv = np.zeros(L); bv[bidx] = spec
        dpred = split_docs([labels[i] for i in (psets[b][0] + bv).argmax(1)], docs, nsent)
        tp = split_docs([labels[i] for i in (psets[b][1] + bv).argmax(1)], tdocs, tnsent)
    assert all(len(tp[d]) == tnsent[d] for d in tdocs) and set(tp) == set(tnsent)
    json.dump({"experiment": exp, "base": b, "config": r["config"], "dev_all": r["dev_all"], "dev_CL": r["dev_CL"],
               "dev_IT": r["dev_IT"], "baseline": {k: base_score[b][k] for k in ("dev_all", "dev_CL", "dev_IT")}},
              open(os.path.join(a.out, "best_candidate.json"), "w"), indent=2)
    json.dump(dpred, open(os.path.join(a.out, "best_candidate_dev_predictions.json"), "w"))
    json.dump(tp, open(os.path.join(a.out, "best_candidate_prediction.json"), "w"))
    x = next(x for x in xv if x["base"] == b and x["experiment"] == exp)
    if r["dev_all"] > best_baseline["dev_all"] and x["robust (both gains > 0)"]:
        zp = os.path.join(a.out, "submission_ENS_candidate.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(a.out, "best_candidate_prediction.json"), arcname="prediction.json")
        print(f"candidate zip written: {zp} (beats the best baseline on combined dev AND its family is robust in both directions)")
    else:
        print("no zip: the candidate does not beat the best baseline on combined dev, or its experiment family is not robust in both directions")


if __name__ == "__main__":
    main()
