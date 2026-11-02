#!/usr/bin/env python3
"""Leave-one-document-out evaluation of methods 1-6, driven by advanced_config.json (your values).

  python optimize.py [--config advanced_config.json] [--out .]

Every fitted or selected quantity (pool, partner, k, temperatures, biases, lambda, threshold, beta, transition matrix)
uses only the 9 training documents of the fold; the held-out document is then predicted. Pooled = the 10 held-out
predictions scored together. Baselines: anchor (no optimisation) and the fixed submitted ensembles H, G2, C1 (their
members were chosen on all 10 documents, so they are not held-out estimates).
Outputs: method_comparison.csv, loo_per_document.csv, loo_per_class.csv, selected_configurations_by_fold.json,
advanced_optimization_report.md (numbers + the decision-criteria labels from your thresholds).
"""
import argparse
import csv
import itertools
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict

import numpy as np

from common import ANCHOR, FIXED, GROUP, HP, PR, L, cl_it, dom, docs, gold, labels, li, mf1, runs, score, vote

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "advanced_config.json"))
ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()
CFG = json.load(open(a.config))
missing = [f"{sec}.{k}" for sec, v in CFG.items() if isinstance(v, dict) for k, x in v.items() if not k.startswith("_") and x is None]
assert not missing, "fill these values in advanced_config.json first (your choices): " + ", ".join(missing)
os.makedirs(a.out, exist_ok=True)
EPS = 1e-12


def lab(P):            # argmax -> labels
    return [labels[i] for i in P.argmax(1)]


def hard(ms, ds, w=None):
    return vote([{k: HP[m][k] for k in ds} for m in ms], w or [1.0] * len(ms))


def fit_T(m, ds):
    """per-model temperature by NLL on training docs (golden-section search on log T in [log 0.25, log 4])."""
    p = np.concatenate([PR[m][k] for k in ds]); y = np.array([li[g] for k in ds for g in gold[k]])
    lp = np.log(np.clip(p, EPS, 1))

    def nll(t):
        z = lp / t; z = z - z.max(1, keepdims=True); z = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -z[np.arange(len(y)), y].mean()
    lo, hi = math.log(0.25), math.log(4.0); g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(math.exp(c)) < nll(math.exp(d)):
            hi = d
        else:
            lo = c
    return math.exp((lo + hi) / 2)


def probs(m, k, T=None):
    p = PR[m][k]
    if T is None or abs(T - 1) < 1e-9:
        return p
    z = np.log(np.clip(p, EPS, 1)) / T; z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def soft_probs(ms, k, Ts=None):
    return np.mean([probs(m, k, (Ts or {}).get(m)) for m in ms], 0)


def soft(ms, ds, Ts=None):
    return {k: lab(soft_probs(ms, k, Ts)) for k in ds}


def nets(ds):
    out = {}
    for n in runs:
        if n == ANCHOR or all(HP[n][k] == HP[ANCHOR][k] for k in docs):
            continue
        out[n] = sum((x == g) - (y == g) for k in ds for x, y, g in zip(HP[n][k], HP[ANCHOR][k], gold[k]))
    return out


def viterbi(lp, logT, beta):
    n = len(lp); dp = lp[0].copy(); bp = np.zeros((n, L), int)
    for i in range(1, n):
        s = dp[:, None] + beta * logT
        bp[i] = s.argmax(0); dp = s.max(0) + lp[i]
    path = [int(dp.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(bp[i][path[-1]]))
    return [labels[j] for j in path[::-1]]


def best(options, f):
    """options in config order; first one wins ties."""
    sc = [(f(o), -i, o) for i, o in enumerate(options)]
    return max(sc)[2]


M1, M2, M3, M4, M5, M6 = (CFG[k] for k in ("M1_complementary", "M2_soft_voting", "M3_class_bias", "M4_domain_aware",
                                           "M5_confidence_gate", "M6_transition"))
methods = ["anchor", "FIXED_H", "FIXED_G2", "FIXED_C1", "M1a_anchor+1_soft", "M1b_anchor+2_hard", "M1c_anchor+2_soft", "M2_soft_topk",
           "M3a_bias_prior_correction", "M3b_bias_grid", "M4a_interp_global", "M4b_interp_per_domain", "M5_confidence_gate", "M6_transition_viterbi"]
if not M3["estimated_prior_correction"]:
    methods.remove("M3a_bias_prior_correction")
pooled = {m: {} for m in methods}
fold_cfg = {}
for held in docs:
    tr = [k for k in docs if k != held]
    D = dom[held]; trD = [k for k in tr if dom[k] == D]
    cfg = {}
    # M1
    nt = nets(tr); pool = sorted(nt, key=lambda n: -nt[n])[:M1["candidate_pool_size"]]
    poolg = [n for n in pool if n in GROUP]
    p1 = best(poolg, lambda n: mf1(soft([ANCHOR, n], tr), tr)) if poolg else None
    pairs = list(itertools.combinations(pool, 2))
    p2 = best(pairs, lambda p: mf1(hard([ANCHOR, *p], tr), tr))
    pairsg = list(itertools.combinations(poolg, 2))
    p3 = best(pairsg, lambda p: mf1(soft([ANCHOR, *p], tr), tr)) if pairsg else None
    cfg.update(M1_pool=pool, M1a_partner=p1, M1b_pair=list(p2), M1c_pair=list(p3) if p3 else None)
    # M2
    rank = sorted(GROUP, key=lambda n: -mf1(HP[n], tr))
    Ts = {n: fit_T(n, tr) for n in rank[:max(M2["k_values"])]} if M2["temperature_scaling"] else None
    k2 = best(M2["k_values"], lambda k: mf1(soft(rank[:k], tr, Ts), tr))
    ens = rank[:k2]
    cfg.update(M2_k=k2, M2_members=ens, M2_temperatures={n: round(t, 3) for n, t in (Ts or {}).items() if n in ens})
    ensP = {k: soft_probs(ens, k, Ts) for k in docs}
    # M3 (on anchor or on the M2 ensemble)
    baseP = {k: (PR[ANCHOR][k] if M3["apply_to"] == "anchor" else ensP[k]) for k in docs}
    if M3["estimated_prior_correction"]:
        gshare = Counter(g for k in tr for g in gold[k]); n_tr = sum(gshare.values())
        mp = np.concatenate([baseP[k] for k in tr]).mean(0)
        bprior = np.array([math.log((gshare[c] + 1) / (n_tr + L)) - math.log(mp[li[c]] + EPS) for c in labels])
        cfg["M3a_bias"] = {c: round(float(bprior[li[c]]), 3) for c in labels}
    b = np.zeros(L)
    for c in M3["classes"]:
        def f(v, c=c):
            bb = b.copy(); bb[li[c]] = v
            return mf1({k: lab(np.log(np.clip(baseP[k], EPS, 1)) + bb) for k in tr}, tr)
        b[li[c]] = best(M3["bias_grid"], f)
    cfg["M3b_bias"] = {c: float(b[li[c]]) for c in M3["classes"]}
    # M4
    def interp(lmb, ds):
        return {k: lab((1 - lmb) * PR[ANCHOR][k] + lmb * ensP[k]) for k in ds}
    lg = best(M4["lambda_grid"], lambda v: mf1(interp(v, tr), tr))
    ld = best(M4["lambda_grid"], lambda v: mf1(interp(v, trD), trD))
    cfg.update(M4a_lambda=lg, M4b_lambda_for_domain=ld, M4b_domain=D)
    # M5
    def gate(t, ds):
        out = {}
        for k in ds:
            pa = PR[ANCHOR][k]; srt = np.sort(pa, 1); margin = srt[:, -1] - srt[:, -2]
            la, le = lab(pa), lab(ensP[k])
            out[k] = [e if mg < t else x for x, e, mg in zip(la, le, margin)]
        return out
    tg = best(M5["margin_thresholds"], lambda t: mf1(gate(t, tr), tr))
    cfg["M5_threshold"] = tg
    # M6
    cnt = np.ones((L, L))
    for k in tr:
        for x, y in zip(gold[k][:-1], gold[k][1:]):
            cnt[li[x], li[y]] += 1
    logT = np.log(cnt / cnt.sum(1, keepdims=True))
    def vit(beta, ds):
        return {k: viterbi(np.log(np.clip(PR[ANCHOR][k], EPS, 1)), logT, beta) for k in ds}
    bt = best(M6["beta_grid"], lambda v: mf1(vit(v, tr), tr))
    cfg["M6_beta"] = bt
    fold_cfg[held] = cfg
    # held-out predictions
    pred = {"anchor": HP[ANCHOR][held], "FIXED_H": HP[FIXED["H"]][held], "FIXED_G2": HP[FIXED["G2"]][held], "FIXED_C1": HP[FIXED["C1"]][held],
            "M1a_anchor+1_soft": soft([ANCHOR, p1], [held])[held] if p1 else HP[ANCHOR][held],
            "M1b_anchor+2_hard": hard([ANCHOR, *p2], [held])[held],
            "M1c_anchor+2_soft": soft([ANCHOR, *p3], [held])[held] if p3 else HP[ANCHOR][held],
            "M2_soft_topk": lab(ensP[held]),
            "M3b_bias_grid": lab(np.log(np.clip(baseP[held], EPS, 1)) + b),
            "M4a_interp_global": interp(lg, [held])[held], "M4b_interp_per_domain": interp(ld, [held])[held],
            "M5_confidence_gate": gate(tg, [held])[held], "M6_transition_viterbi": vit(bt, [held])[held]}
    if M3["estimated_prior_correction"]:
        pred["M3a_bias_prior_correction"] = lab(np.log(np.clip(baseP[held], EPS, 1)) + bprior)
    for m in methods:
        pooled[m][held] = pred[m]

# ---------------- results
perdoc, comp, pcls = [], [], []
base_doc = {k: mf1({k: pooled["anchor"][k]}, [k]) for k in docs}
H_pooled = mf1(pooled["FIXED_H"], docs)
for m in methods:
    v = {k: mf1({k: pooled[m][k]}, [k]) for k in docs}
    for k in docs:
        perdoc.append({"method": m, "document": k, "domain": dom[k], "doc_macro_f1": round(v[k], 4), "delta_vs_anchor": round(v[k] - base_doc[k], 4)})
    s = score(pooled[m], docs); ci = cl_it(pooled[m])
    gains = sorted(((v[k] - base_doc[k]), k) for k in docs)
    pos = [g for g, _ in gains if g > 0]
    comp.append({"method": m, "pooled_macro_f1": round(s["macro_f1"], 4), "pooled_all13": round(mf1(pooled[m], docs, "all13"), 4),
                 "CL": round(ci["CL"], 4), "IT": round(ci["IT"], 4), "macro_P": round(s["macro_p"], 4), "macro_R": round(s["macro_r"], 4),
                 "gain_vs_anchor": round(s["macro_f1"] - mf1(pooled["anchor"], docs), 4), "gain_vs_H_pooled": round(s["macro_f1"] - H_pooled, 4),
                 "CL_change_vs_anchor": round(ci["CL"] - cl_it(pooled["anchor"])["CL"], 4),
                 "doc_mean": round(st.mean(v.values()), 4), "doc_median": round(st.median(v.values()), 4), "doc_sd": round(st.stdev(v.values()), 4),
                 "docs_improved_vs_anchor": len(pos), "docs_worse_vs_anchor": sum(g < 0 for g, _ in gains),
                 "largest_doc_share_of_gains": round(max(pos) / sum(pos), 3) if pos else "",
                 "held_out": not m.startswith("FIXED")})
    for c in labels:
        pl = s["per_label"].get(c)
        pcls.append({"method": m, "label": c, "precision": pl["p"] if pl else "", "recall": pl["r"] if pl else "", "f1": pl["f1"] if pl else "",
                     "support": pl["support"] if pl else 0})


def stable(m):
    keymap = {"M1a_anchor+1_soft": "M1a_partner", "M1b_anchor+2_hard": "M1b_pair", "M1c_anchor+2_soft": "M1c_pair", "M2_soft_topk": "M2_members",
              "M3a_bias_prior_correction": "M3a_bias", "M3b_bias_grid": "M3b_bias", "M4a_interp_global": "M4a_lambda",
              "M4b_interp_per_domain": "M4b_lambda_for_domain", "M5_confidence_gate": "M5_threshold", "M6_transition_viterbi": "M6_beta"}
    if m not in keymap:
        return ""
    c = Counter(json.dumps(fold_cfg[k][keymap[m]], sort_keys=True) for k in docs)
    return c.most_common(1)[0][1]


DC = CFG["decision_criteria"]
for r in comp:
    r["folds_with_most_common_choice"] = stable(r["method"])
    if not r["held_out"] or r["method"] == "anchor":
        r["label"] = "baseline"; continue
    ok = (r["gain_vs_H_pooled"] >= DC["min_pooled_gain_over_H"] and r["CL_change_vs_anchor"] >= -DC["max_CL_drop_vs_anchor"]
          and r["docs_improved_vs_anchor"] >= DC["min_documents_improved_vs_anchor"]
          and (r["folds_with_most_common_choice"] == "" or r["folds_with_most_common_choice"] >= DC["min_folds_with_same_choice"]))
    r["label"] = "Promising" if ok else ("Inconclusive" if r["gain_vs_H_pooled"] > 0 else "No improvement")


def dump(fn, rows):
    with open(os.path.join(a.out, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)


dump("method_comparison.csv", comp); dump("loo_per_document.csv", perdoc); dump("loo_per_class.csv", pcls)
json.dump({"config_used": CFG, "folds": fold_cfg}, open(os.path.join(a.out, "selected_configurations_by_fold.json"), "w"), indent=1, default=str)
for r in comp:
    print(f"{r['method']:28s} pooled {r['pooled_macro_f1']:.4f} (vs H {r['gain_vs_H_pooled']:+.4f}) CL {r['CL']:.4f} IT {r['IT']:.4f} | "
          f"docs +{r['docs_improved_vs_anchor']}/-{r['docs_worse_vs_anchor']} | stable {r['folds_with_most_common_choice']} | {r['label']}")
