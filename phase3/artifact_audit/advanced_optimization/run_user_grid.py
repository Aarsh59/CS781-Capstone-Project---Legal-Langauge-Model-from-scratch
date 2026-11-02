#!/usr/bin/env python3
"""Leave-one-document-out evaluation of the user's decision strategies (user_hyperparameters.json). Plumbing; declare it.

Every selection (members, k, weights, temperature, biases, alphas, thresholds, transition strength/smoothing, transition
matrix) uses only the 9 training documents of the fold (the per-domain alphas: only that domain's training documents).
Selection score = training-documents macro-F1 (local scorer); ties -> the first configuration in the listed order.
Outputs (prefix ug_): method_comparison.csv, loo_per_document.csv, loo_per_class.csv, selected_configurations_by_fold.json,
gate_switches.csv, config_counts.csv
"""
import csv
import itertools
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict

import numpy as np

from common import ANCHOR, FIXED, GROUP, HERE, HP, PR, L, cl_it, dom, docs, gold, labels, li, mf1, runs, score, vote

U = json.load(open(os.path.join(HERE, "user_hyperparameters.json")))
EPS = 1e-12
COUNT = Counter()                     # configurations evaluated per strategy (per fold; must stay <= budget)


def lab(P):
    return [labels[i] for i in P.argmax(1)]


def logp(P):
    return np.log(np.clip(P, EPS, 1))


def pick(strategy, options, f):
    COUNT[strategy] = max(COUNT[strategy], len(options))
    assert len(options) <= U["budget_per_strategy"], f"{strategy}: {len(options)} configurations > budget"
    best, bi = None, None
    for i, o in enumerate(options):
        s = f(o)
        if best is None or s > best:
            best, bi = s, i
    return options[bi]


# ---------------- strategy implementations (all given the fold's training docs `tr`)
def net_rank(pool, tr):
    return sorted(pool, key=lambda n: -sum((x == g) - (y == g) for k in tr for x, y, g in zip(HP[n][k], HP[ANCHOR][k], gold[k])))


def s1(tr):
    pool = net_rank(U["S1_complementarity"]["pool"], tr)
    cfgs = []
    for size in U["S1_complementarity"]["ensemble_sizes"]:
        mem = [ANCHOR] + pool[:size - 1]
        cfgs.append(("always", "uniform", size, 1.0, mem))
        for w in U["S1_complementarity"]["anchor_weights"]:
            if w != 1.0:
                cfgs.append(("always", "anchor-weighted", size, w, mem))
    rank_f1 = sorted([ANCHOR] + U["S1_complementarity"]["pool"], key=lambda n: -mf1(HP[n], tr))
    for size in U["S1_complementarity"]["ensemble_sizes"]:      # anchor optional: top-size by training F1, uniform
        cfgs.append(("optional", "uniform", size, 1.0, rank_f1[:size]))

    def run(c, ds):
        mem, w = c[4], c[3]
        ws = [w if m == ANCHOR else 1.0 for m in mem]
        if ANCHOR in mem:                                    # tie -> anchor: list the anchor first
            order = [ANCHOR] + [m for m in mem if m != ANCHOR]; ws = [w] + [1.0] * (len(order) - 1)
        else:
            order = mem
        return vote([{k: HP[m][k] for k in ds} for m in order], ws)
    c = pick("S1", cfgs, lambda c: mf1(run(c, tr), tr))
    return c, run


def s2(tr):
    others = sorted([n for n in GROUP if n != ANCHOR], key=lambda n: -mf1(HP[n], tr))
    def ens(k, w, T, d):
        mem = [ANCHOR] + others[:k - 1]
        ps = []
        for m in mem:
            z = logp(PR[m][d]) / T; z -= z.max(1, keepdims=True); e = np.exp(z); ps.append(e / e.sum(1, keepdims=True))
        ws = np.array([w] + [1.0] * (len(mem) - 1))
        return np.tensordot(ws / ws.sum(), np.array(ps), 1), mem
    stageA = [(k, w, 1.0) for k in U["S2_soft"]["k"] for w in U["S2_soft"]["anchor_weights"]]
    kA, wA, _ = pick("S2_stageA", stageA, lambda c: mf1({d: lab(ens(*c, d)[0]) for d in tr}, tr))
    stageB = [(kA, wA, T) for T in U["S2_soft"]["temperatures"]]
    c = pick("S2_stageB", stageB, lambda c: mf1({d: lab(ens(*c, d)[0]) for d in tr}, tr))
    return c, ens


def s3_single(cls, tr):
    grid = U["S3_bias"]["grids"][cls]
    def run(v, ds):
        b = np.zeros(L); b[li[cls]] = v
        return {d: lab(logp(PR[ANCHOR][d]) + b) for d in ds}
    return pick(f"S3_{cls}", grid, lambda v: mf1(run(v, tr), tr)), run


def interp(ensP, a, ds):
    return {d: lab((1 - a) * PR[ANCHOR][d] + a * ensP[d]) for d in ds}


def gate_pred(ensP, signal, t, ds, stats=None):
    out = {}
    for d in ds:
        pa = PR[ANCHOR][d]; srt = np.sort(pa, 1)
        conf = {"max_prob": srt[:, -1], "margin": srt[:, -1] - srt[:, -2],
                "entropy": -(pa * logp(pa)).sum(1) / math.log(L)}[signal]
        use = conf > t if signal == "entropy" else conf < t           # low confidence = high entropy
        la, le = lab(pa), lab(ensP[d])
        out[d] = [e if u else x for x, e, u in zip(la, le, use)]
        if stats is not None:
            for x, e, u, g in zip(la, le, use, gold[d]):
                if u and x != e:
                    stats["switches"] += 1; stats["fixed"] += (e == g) and (x != g); stats["broke"] += (x == g) and (e != g)
    return out


def transmat(tr, smooth):
    cnt = np.full((L, L), smooth)
    for d in tr:
        for x, y in zip(gold[d][:-1], gold[d][1:]):
            cnt[li[x], li[y]] += 1
    return np.log(cnt / cnt.sum(1, keepdims=True))


def local_fix(lam, logT, ds):
    out = {}
    for d in ds:
        lp = logp(PR[ANCHOR][d]); a = lp.argmax(1); n = len(a); new = []
        for i in range(n):
            s = lp[i].copy()
            if i > 0:
                s += lam * logT[a[i - 1], :]
            if i + 1 < n:
                s += lam * logT[:, a[i + 1]]
            new.append(labels[int(s.argmax())])
        out[d] = new
    return out


def viterbi(lam, logT, ds):
    out = {}
    for d in ds:
        lp = logp(PR[ANCHOR][d]); n = len(lp); dp = lp[0].copy(); bp = np.zeros((n, L), int)
        for i in range(1, n):
            s = dp[:, None] + lam * logT; bp[i] = s.argmax(0); dp = s.max(0) + lp[i]
        path = [int(dp.argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(bp[i][path[-1]]))
        out[d] = [labels[j] for j in path[::-1]]
    return out


# ---------------- leave-one-document-out
M = ["anchor", "FIXED_H", "FIXED_G2", "FIXED_C1", "S1_complementarity_hard", "S2_soft_compatible",
     "S3_bias_ArgumentPetitioner", "S3_bias_ArgumentRespondent", "S3_bias_RulingByLowerCourt", "S3_bias_RatioOfTheDecision",
     "S3_bias_PrecedentReliedUpon", "S3_bias_PrecedentNotReliedUpon", "S3_bias_Statute", "S3_bias_combo_args_lowercourt",
     "S4_always_ensemble", "S4_alpha_global", "S4_alpha_per_domain", "S5_gate_max_prob", "S5_gate_margin", "S5_gate_entropy",
     "S6_transition_local", "S6_transition_viterbi"]
pooled = {m: {} for m in M}
folds = {}
gstats = {s: Counter() for s in ("max_prob", "margin", "entropy")}
for held in docs:
    tr = [d for d in docs if d != held]; D = dom[held]; trD = [d for d in tr if dom[d] == D]
    cfg = {}
    c1, run1 = s1(tr); cfg["S1"] = {"anchor": c1[0], "voting": c1[1], "size": c1[2], "anchor_weight": c1[3], "members": c1[4]}
    c2, ens = s2(tr); ensP = {d: ens(*c2, d)[0] for d in docs}
    cfg["S2"] = {"k": c2[0], "anchor_weight": c2[1], "temperature": c2[2], "members": ens(*c2, held)[1]}
    pred = {"anchor": HP[ANCHOR][held], "FIXED_H": HP[FIXED["H"]][held], "FIXED_G2": HP[FIXED["G2"]][held], "FIXED_C1": HP[FIXED["C1"]][held],
            "S1_complementarity_hard": run1(c1, [held])[held], "S2_soft_compatible": lab(ensP[held])}
    chosen = {}
    for cls in U["S3_bias"]["grids"]:
        v, run3 = s3_single(cls, tr); chosen[cls] = v
        pred[f"S3_bias_{cls}"] = run3(v, [held])[held]
    cfg["S3_independent"] = chosen
    bcombo = np.zeros(L)
    for cls in U["S3_bias"]["combination"]:
        bcombo[li[cls]] = chosen[cls]
    pred["S3_bias_combo_args_lowercourt"] = lab(logp(PR[ANCHOR][held]) + bcombo)
    ag = pick("S4_global", U["S4_domain"]["alpha_global"], lambda a: mf1(interp(ensP, a, tr), tr))
    adom = pick(f"S4_{D}", U["S4_domain"][f"alpha_{D}"], lambda a: mf1(interp(ensP, a, trD), trD))
    cfg["S4"] = {"alpha_global": ag, f"alpha_{D}": adom, "domain_training_docs": len(trD)}
    pred["S4_always_ensemble"] = lab(ensP[held]); pred["S4_alpha_global"] = interp(ensP, ag, [held])[held]
    pred["S4_alpha_per_domain"] = interp(ensP, adom, [held])[held]
    cfg["S5"] = {}
    for sig in ("max_prob", "margin", "entropy"):
        t = pick(f"S5_{sig}", U["S5_gate"][sig], lambda t: mf1(gate_pred(ensP, sig, t, tr), tr))
        cfg["S5"][sig] = t
        pred[f"S5_gate_{sig}"] = gate_pred(ensP, sig, t, [held], gstats[sig])[held]
    tcfg = [(lam, sm) for lam in U["S6_transition"]["strength"] for sm in U["S6_transition"]["smoothing"]]
    lt = pick("S6_local", tcfg, lambda c: mf1(local_fix(c[0], transmat(tr, c[1]), tr), tr))
    vt = pick("S6_viterbi", tcfg, lambda c: mf1(viterbi(c[0], transmat(tr, c[1]), tr), tr))
    cfg["S6"] = {"local_strength_smoothing": lt, "viterbi_strength_smoothing": vt}
    pred["S6_transition_local"] = local_fix(lt[0], transmat(tr, lt[1]), [held])[held]
    pred["S6_transition_viterbi"] = viterbi(vt[0], transmat(tr, vt[1]), [held])[held]
    folds[held] = cfg
    for m in M:
        pooled[m][held] = pred[m]

# ---------------- report tables
base_doc = {d: mf1({d: pooled["anchor"][d]}, [d]) for d in docs}
base = mf1(pooled["anchor"], docs); base_cl = cl_it(pooled["anchor"])["CL"]; Hp = mf1(pooled["FIXED_H"], docs)
keymap = {"S1_complementarity_hard": lambda c: c["S1"], "S2_soft_compatible": lambda c: c["S2"],
          "S3_bias_combo_args_lowercourt": lambda c: {k: c["S3_independent"][k] for k in U["S3_bias"]["combination"]},
          "S4_alpha_global": lambda c: c["S4"]["alpha_global"], "S4_alpha_per_domain": lambda c: {k: v for k, v in c["S4"].items() if k.startswith("alpha_") and k != "alpha_global"},
          "S6_transition_local": lambda c: c["S6"]["local_strength_smoothing"], "S6_transition_viterbi": lambda c: c["S6"]["viterbi_strength_smoothing"]}
for cls in U["S3_bias"]["grids"]:
    keymap[f"S3_bias_{cls}"] = lambda c, cls=cls: c["S3_independent"][cls]
for sig in ("max_prob", "margin", "entropy"):
    keymap[f"S5_gate_{sig}"] = lambda c, sig=sig: c["S5"][sig]
comp, perdoc, pcls = [], [], []
for m in M:
    v = {d: mf1({d: pooled[m][d]}, [d]) for d in docs}
    s = score(pooled[m], docs); ci = cl_it(pooled[m])
    deltas = [v[d] - base_doc[d] for d in docs]; pos = [x for x in deltas if x > 0]
    stab = ""
    if m in keymap:
        cc = Counter(json.dumps(keymap[m](folds[d]), sort_keys=True, default=str) for d in docs)
        stab = cc.most_common(1)[0][1]
    comp.append({"method": m, "held_out": not m.startswith("FIXED"), "pooled_macro_f1": round(s["macro_f1"], 4),
                 "pooled_gold_only": round(mf1(pooled[m], docs, "gold_only"), 4), "pooled_all13": round(mf1(pooled[m], docs, "all13"), 4),
                 "CL": round(ci["CL"], 4), "IT": round(ci["IT"], 4), "macro_P": round(s["macro_p"], 4), "macro_R": round(s["macro_r"], 4),
                 "gain_vs_anchor": round(s["macro_f1"] - base, 4), "gain_vs_H": round(s["macro_f1"] - Hp, 4), "CL_change_vs_anchor": round(ci["CL"] - base_cl, 4),
                 "docs_improved": len(pos), "docs_worse": sum(x < 0 for x in deltas),
                 "largest_doc_share_of_gains": round(max(pos) / sum(pos), 3) if pos else "",
                 "doc_mean": round(st.mean(v.values()), 4), "doc_median": round(st.median(v.values()), 4), "doc_sd": round(st.stdev(v.values()), 4),
                 "folds_with_same_choice": stab})
    for d in docs:
        perdoc.append({"method": m, "document": d, "domain": dom[d], "doc_macro_f1": round(v[d], 4), "delta_vs_anchor": round(v[d] - base_doc[d], 4)})
    for c in labels:
        pl = s["per_label"].get(c)
        pcls.append({"method": m, "label": c, "precision": pl["p"] if pl else "", "recall": pl["r"] if pl else "", "f1": pl["f1"] if pl else "", "support": pl["support"] if pl else 0})


def dump(fn, rows):
    with open(os.path.join(HERE, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)


dump("ug_method_comparison.csv", comp); dump("ug_loo_per_document.csv", perdoc); dump("ug_loo_per_class.csv", pcls)
dump("ug_gate_switches.csv", [{"signal": s, **v} for s, v in gstats.items()])
dump("ug_config_counts.csv", [{"strategy": k, "configurations_per_fold": v, "budget": U["budget_per_strategy"]} for k, v in sorted(COUNT.items())])
json.dump({"hyperparameters": U, "folds": folds}, open(os.path.join(HERE, "ug_selected_configurations_by_fold.json"), "w"), indent=1, default=str)
for r in comp:
    print(f"{r['method']:32s} pooled {r['pooled_macro_f1']:.4f} (anchor {r['gain_vs_anchor']:+.4f}, H {r['gain_vs_H']:+.4f}) CL {r['CL']:.4f} "
          f"({r['CL_change_vs_anchor']:+.4f}) IT {r['IT']:.4f} | docs +{r['docs_improved']}/-{r['docs_worse']} share {r['largest_doc_share_of_gains']} | stable {r['folds_with_same_choice']}")
print("gate switches:", dict(gstats)); print("config counts:", dict(COUNT))
