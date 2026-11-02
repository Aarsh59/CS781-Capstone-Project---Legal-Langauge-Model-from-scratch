#!/usr/bin/env python3
"""Second-stage offline analysis on artifact_audit/bundle (plumbing / evaluation; declare it).

Reads only ../bundle and ../experiment_inventory.csv (+ the run configs it points to). Writes CSVs into this folder.
No training, inference, downloads or submissions. Test labels do not exist and are never used.

Methods (fixed in advance; any selection uses only the 9 training documents of each leave-one-document-out fold):
  BASE_anchor        B_lora_ctx_wide_allep alone (no optimisation)
  FIXED_H / G2 / C1  the submitted ensembles with their fixed members (members were chosen on ALL dev docs earlier,
                     so their LOO numbers are not clean held-out estimates; reported for comparison)
  LOO_anchor_plus2   anchor + best pair among the 6 runs with the best net fixes-vs-anchor-errors on the fold's
                     training docs (15 pairs), 3-member hard vote, ties -> anchor
  LOO_anchor_w_top4  anchor weight w in {1, 2} + the 4 best-net runs (training docs) x1; w chosen on training docs
  LOO_topk_hard      uniform hard vote of the top-k runs by training-docs macro-F1, k in {3, 5, 7} chosen on training docs
  LOO_topk_soft      mean probabilities of the top-k runs of ONE calibration group (inv_sqrt cross-entropy, non-CRF),
                     ranked on training docs, k in {3, 5} chosen on training docs
  LOO_anchor_bias    anchor probabilities + log-probability biases motivated by the confusion matrices (arguments and
                     RulingByLowerCourt absorbed by RatioOfTheDecision): b_Ratio in {0, -0.3, -0.6} and one shared
                     b in {0, +0.3, +0.6} for ArgumentPetitioner / ArgumentRespondent / RulingByLowerCourt (9 combos),
                     chosen on training docs
Scoring: rr_metrics.rr_scores (macro over labels present in gold OR predictions). Per-document scores are reported,
and so are POOLED scores: each document predicted by the model selected without it, all 10 pooled and scored once.
"""
import csv
import itertools
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
BD = os.path.join(AUD, "bundle")
sys.path.insert(0, os.path.join(BD, "evaluation"))
from rr_ensemble import vote                     # noqa: E402
from rr_metrics import rr_scores                 # noqa: E402

labels = json.load(open(os.path.join(BD, "label_vocab.json")))["labels"]
L, li = len(labels), {l: i for i, l in enumerate(labels)}
ANCHOR, FIXED = "B_lora_ctx_wide_allep", ["H_anchor_w2.0+top4net", "G2_anchor+B_lorma_allep+KD_seed123_allep", "C1_one_rep_per_family_bestdev"]

# ---------------- load bundle
gold, dom = defaultdict(list), {}
for r in csv.DictReader(open(os.path.join(BD, "dev_gold.csv"))):
    gold[r["doc_id"]].append(r["gold_label"]); dom[r["doc_id"]] = r["domain"]
docs = list(gold)
HP = defaultdict(lambda: defaultdict(list))
rd = csv.DictReader(open(os.path.join(BD, "hard_predictions_dev.csv")))
names = [c for c in rd.fieldnames if c not in ("split", "doc_id", "sentence_index")]
for r in rd:
    for n in names:
        HP[n][r["doc_id"]].append(r[n])
inv = {r["experiment_name"]: r for r in csv.DictReader(open(os.path.join(AUD, "experiment_inventory.csv")))}
runs = [n for n in names if n in inv and not inv[n]["experiment_id"].startswith("ensemble:")]
PR = {}
for n in runs:
    f = os.path.join(BD, "probs", f"{n}_dev.csv")
    if os.path.exists(f):
        d = defaultdict(list); dec = None
        for r in csv.DictReader(open(f)):
            d[r["doc_id"]].append([float(r[f"p_{l}"]) for l in labels]); dec = r["decode"]
        PR[n] = ({k: np.array(v) for k, v in d.items()}, dec)
cw = {}
for n in runs:
    cp = inv[n]["config_path"]
    c = json.load(open(cp)) if os.path.exists(cp) else {}
    cw[n] = ("CRF" if c.get("head", {}).get("name") == "mlp_crf" else "KD" if c.get("kd") else
             str(c.get("train", {}).get("class_weights")))
calib_group = [n for n in PR if PR[n][1] == "argmax" and cw.get(n) == "inv_sqrt"]
assert ANCHOR in calib_group


def sc(P, ds):
    g = {k: gold[k] for k in ds}
    return rr_scores(g, {k: P[k] for k in ds}, labels)


def mf1(P, ds):
    return sc(P, ds)["macro_f1"]


def hv(ms, ds, w=None):
    out = vote([{k: HP[m][k] for k in ds} for m in ms], w or [1.0] * len(ms))
    return out


def net_vs_anchor(ds):
    res = {}
    for n in runs:
        if n == ANCHOR or HP[n] == HP[ANCHOR]:
            continue
        f = b = 0
        for k in ds:
            for a, x, g in zip(HP[ANCHOR][k], HP[n][k], gold[k]):
                f += a != g and x == g; b += a == g and x != g
        res[n] = f - b
    return res


def soft(ms, ds, bias=None):
    out = {}
    for k in ds:
        acc = np.mean([PR[m][0][k] for m in ms], 0)
        lp = np.log(np.clip(acc, 1e-12, 1)) + (bias if bias is not None else 0)
        out[k] = [labels[i] for i in lp.argmax(1)]
    return out


def bias_vec(br, bo):
    b = np.zeros(L); b[li["RatioOfTheDecision"]] = br
    for c in ("ArgumentPetitioner", "ArgumentRespondent", "RulingByLowerCourt"):
        b[li[c]] = bo
    return b


# ---------------- 1. complementarity vs H
H = FIXED[0]
comp, comp_doc, comp_cls = [], [], []
for n in [x for x in names if x != H]:
    tot = defaultdict(int); per_doc = defaultdict(lambda: [0, 0]); per_cls = defaultdict(lambda: [0, 0])
    for k in docs:
        for h, x, g in zip(HP[H][k], HP[n][k], gold[k]):
            if h != g and x == g:
                tot["fixes"] += 1; per_doc[k][0] += 1; per_cls[g][0] += 1
            if h == g and x != g:
                tot["breaks"] += 1; per_doc[k][1] += 1; per_cls[g][1] += 1
    comp.append({"run": n, "fixes_H_errors": tot["fixes"], "breaks_H_correct": tot["breaks"], "net": tot["fixes"] - tot["breaks"],
                 "CL_net": sum(per_doc[k][0] - per_doc[k][1] for k in docs if dom[k] == "CL"),
                 "IT_net": sum(per_doc[k][0] - per_doc[k][1] for k in docs if dom[k] == "IT")})
    for k in docs:
        comp_doc.append({"run": n, "document": k, "domain": dom[k], "fixes": per_doc[k][0], "breaks": per_doc[k][1], "net": per_doc[k][0] - per_doc[k][1]})
    for c in labels:
        if per_cls[c] != [0, 0]:
            comp_cls.append({"run": n, "gold_label": c, "fixes": per_cls[c][0], "breaks": per_cls[c][1], "net": per_cls[c][0] - per_cls[c][1]})
comp.sort(key=lambda r: -r["net"])
h_err = sum(h != g for k in docs for h, g in zip(HP[H][k], gold[k]))

# ---------------- 2-4. leave-one-document-out
methods = ["BASE_anchor", "FIXED_H", "FIXED_G2", "FIXED_C1", "LOO_anchor_plus2", "LOO_anchor_w_top4", "LOO_topk_hard", "LOO_topk_soft", "LOO_anchor_bias"]
pooled = {m: {} for m in methods}
perdoc, choices = [], []
for held in docs:
    tr = [k for k in docs if k != held]
    nets = net_vs_anchor(tr)
    cand6 = sorted(nets, key=lambda n: -nets[n])[:6]
    best_pair = max(itertools.combinations(cand6, 2), key=lambda p: mf1(hv([ANCHOR, *p], tr), tr))
    top4 = sorted(nets, key=lambda n: -nets[n])[:4]
    best_w = max((1.0, 2.0), key=lambda w: mf1(hv([ANCHOR] + top4, tr, [w, 1, 1, 1, 1]), tr))
    rank_tr = sorted(runs, key=lambda n: -mf1(HP[n], tr))
    best_k = max((3, 5, 7), key=lambda k: mf1(hv(rank_tr[:k], tr), tr))
    rank_cal = [n for n in rank_tr if n in calib_group]
    best_ks = max((3, 5), key=lambda k: mf1(soft(rank_cal[:k], tr), tr))
    best_b = max(itertools.product((0.0, -0.3, -0.6), (0.0, 0.3, 0.6)), key=lambda b: mf1(soft([ANCHOR], tr, bias_vec(*b)), tr))
    pred = {
        "BASE_anchor": HP[ANCHOR][held], "FIXED_H": HP[FIXED[0]][held], "FIXED_G2": HP[FIXED[1]][held], "FIXED_C1": HP[FIXED[2]][held],
        "LOO_anchor_plus2": hv([ANCHOR, *best_pair], [held])[held],
        "LOO_anchor_w_top4": hv([ANCHOR] + top4, [held], [best_w, 1, 1, 1, 1])[held],
        "LOO_topk_hard": hv(rank_tr[:best_k], [held])[held],
        "LOO_topk_soft": soft(rank_cal[:best_ks], [held])[held],
        "LOO_anchor_bias": soft([ANCHOR], [held], bias_vec(*best_b))[held],
    }
    choices.append({"held_out": held, "domain": dom[held], "anchor_plus2_pair": " + ".join(best_pair), "anchor_w": best_w,
                    "top4_by_net": " ".join(top4), "topk_hard_k": best_k, "topk_hard_members": " ".join(rank_tr[:best_k]),
                    "topk_soft_k": best_ks, "topk_soft_members": " ".join(rank_cal[:best_ks]), "bias_ratio": best_b[0], "bias_args_rlc": best_b[1]})
    for m in methods:
        pooled[m][held] = pred[m]
        perdoc.append({"method": m, "document": held, "domain": dom[held], "doc_macro_f1": round(mf1({held: pred[m]}, [held]), 4)})

summary, pclass = [], []
base_pd = {r["document"]: r["doc_macro_f1"] for r in perdoc if r["method"] == "BASE_anchor"}
for m in methods:
    v = [r["doc_macro_f1"] for r in perdoc if r["method"] == m]
    s = sc(pooled[m], docs)
    wins = sum(r["doc_macro_f1"] > base_pd[r["document"]] for r in perdoc if r["method"] == m)
    summary.append({"method": m, "pooled_macro_f1": round(s["macro_f1"], 4), "pooled_CL": round(mf1(pooled[m], [k for k in docs if dom[k] == "CL"]), 4),
                    "pooled_IT": round(mf1(pooled[m], [k for k in docs if dom[k] == "IT"]), 4), "pooled_P": round(s["macro_p"], 4), "pooled_R": round(s["macro_r"], 4),
                    "doc_mean": round(st.mean(v), 4), "doc_median": round(st.median(v), 4), "doc_std": round(st.stdev(v), 4),
                    "docs_better_than_anchor": wins, "docs_worse_than_anchor": sum(r["doc_macro_f1"] < base_pd[r["document"]] for r in perdoc if r["method"] == m),
                    "labels_in_pooled_macro": len(s["per_label"])})
    for c in labels:
        pl = s["per_label"].get(c)
        pclass.append({"method": m, "label": c, "precision": pl["p"] if pl else "", "recall": pl["r"] if pl else "", "f1": pl["f1"] if pl else "",
                       "support": pl["support"] if pl else 0})

# in-sample hard vs soft on the full dev set (same members), for item 3
hs = []
for k in (3, 5, 7):
    rk = sorted(calib_group, key=lambda n: -mf1(HP[n], docs))[:k]
    hs.append({"members(calibration group, ranked on full dev)": " ".join(rk), "k": k,
               "hard_vote_dev": round(mf1(hv(rk, docs), docs), 4), "soft_vote_dev": round(mf1(soft(rk, docs), docs), 4)})
ece = []
flatg = np.array([li[g] for k in docs for g in gold[k]])
for n, (p, dec) in PR.items():
    a = np.concatenate([p[k] for k in docs]); conf = a.max(1); ok = a.argmax(1) == flatg; b = np.minimum((conf * 10).astype(int), 9)
    ece.append({"run": n, "decode": dec, "loss_type": cw.get(n), "mean_confidence": round(conf.mean(), 3), "accuracy_of_argmax": round(ok.mean(), 3),
                "ECE": round(sum(abs(conf[b == i].mean() - ok[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any()), 3),
                "in_soft_vote_group": n in calib_group})


def dump(fn, rows):
    with open(os.path.join(HERE, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)


dump("complementarity_vs_H.csv", comp); dump("complementarity_vs_H_by_document.csv", comp_doc); dump("complementarity_vs_H_by_class.csv", comp_cls)
dump("loo_per_document.csv", perdoc); dump("loo_summary.csv", summary); dump("loo_per_class_pooled.csv", pclass); dump("loo_fold_choices.csv", choices)
dump("hard_vs_soft_in_sample.csv", hs); dump("calibration.csv", ece)
print(f"H errors on dev: {h_err}; best complementarity (net vs H): {[(r['run'], r['net']) for r in comp[:5]]}")
print(f"calibration group for soft voting (inv_sqrt CE, non-CRF): {len(calib_group)} runs")
for r in summary:
    print(f"{r['method']:20s} pooled {r['pooled_macro_f1']:.4f} (CL {r['pooled_CL']:.4f} IT {r['pooled_IT']:.4f}) | per-doc mean {r['doc_mean']:.4f} "
          f"median {r['doc_median']:.4f} sd {r['doc_std']:.4f} | docs better/worse than anchor {r['docs_better_than_anchor']}/{r['docs_worse_than_anchor']}")
print("hard vs soft (in-sample):", [(r["k"], r["hard_vote_dev"], r["soft_vote_dev"]) for r in hs])
