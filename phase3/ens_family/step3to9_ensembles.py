#!/usr/bin/env python3
"""Steps 3-9: family-balanced, anchored, weighted and probability ensembles; stability; shortlist; candidate zips.

All rules are fixed before evaluation (no weight grids, no exhaustive member search). Families (user's choice,
by technique type): F1 LoRA-type adapters on B, F2 prompt/prefix/IA3 on B, F3 KD students, F4 other backbones A/C/D.

Tie-breaking (documented, deterministic): rr_ensemble.vote = weighted majority, ties -> the earliest-listed member.
Members are always listed anchor first (when present), then by dev macro-F1 descending. For family-level votes,
families are listed in the order of their best member's dev macro-F1 (so F1, which contains the anchor, is first).
Probability averaging: mean of probabilities (soft vote), argmax; B_lora_crf_allep is excluded (its saved
probabilities are pre-Viterbi emissions, not its decoded output).
"""
import csv
import itertools
import json
import os
import zipfile

import numpy as np

from common import ANCHOR, flat, load, rr_scores, scores, vote

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
ctx = load()
labels, runs, M, gold = ctx["labels"], ctx["runs"], ctx["M"], ctx["gold"]
docs, tdocs = list(gold), [d["id"] for D in ctx["tst"] for d in ctx["tst"][D]]
S = {n: scores(ctx, M[n]["dev"]) for n in runs}


def family(n):
    r = M[n]
    if r["kd"]:
        return "F3_KD"
    if r["backbone"] != "run_B2":
        return "F4_other_backbones"
    if r["technique"] in ("prompt_tuning", "prefix_tuning", "ia3"):
        return "F2_prompt_prefix_ia3"
    return "F1_lora_type_adapters"


FAM = {}
for n in runs:
    FAM.setdefault(family(n), []).append(n)
for f in FAM:
    FAM[f].sort(key=lambda n: -S[n]["dev"])
fam_order = sorted(FAM, key=lambda f: -S[FAM[f][0]]["dev"])
assert sorted(len(v) for v in FAM.values()) == [2, 3, 4, 18], {f: len(v) for f, v in FAM.items()}

gold_flat = flat(ctx, gold)
A_dev = flat(ctx, M[ANCHOR]["dev"])
fixes = {n: sum(A_dev[i] != gold_flat[i] and x == gold_flat[i] for i, x in enumerate(flat(ctx, M[n]["dev"]))) for n in runs}
breaks = {n: sum(A_dev[i] == gold_flat[i] and x != gold_flat[i] for i, x in enumerate(flat(ctx, M[n]["dev"]))) for n in runs}
net = {n: fixes[n] - breaks[n] for n in runs}

# ---------------- model_family_audit.csv
dev_dup = {}
with open(os.path.join(OUT, "model_family_audit.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "run", "backbone", "technique", "kd", "head", "dev", "CL", "IT", "rank_in_family",
                "fixes_anchor_errors", "breaks_anchor_correct", "net_vs_anchor", "near_duplicate_of"])
    F = {n: flat(ctx, M[n]["dev"]) for n in runs}
    T = {n: flat(ctx, M[n]["test"], "test") for n in runs}
    for fam in fam_order:
        for i, n in enumerate(FAM[fam]):
            dup = [o for o in runs if o != n and np.mean([a == b for a, b in zip(F[n], F[o])]) >= 0.99
                   and np.mean([a == b for a, b in zip(T[n], T[o])]) >= 0.99]
            dev_dup[n] = dup
            r = M[n]
            w.writerow([fam, n, r["backbone"], r["technique"], r["kd"], r["head"], f"{S[n]['dev']:.4f}", f"{S[n]['CL']:.4f}",
                        f"{S[n]['IT']:.4f}", i + 1, fixes[n], breaks[n], net[n], " ".join(dup)])


def order_members(ms):
    ms = list(dict.fromkeys(ms))
    head = [ANCHOR] if ANCHOR in ms else []
    return head + sorted([m for m in ms if m != ANCHOR], key=lambda n: -S[n]["dev"])


def hv(ms, w=None, split="dev"):
    ms = order_members(ms) if w is None else ms
    return vote([M[m][split] for m in ms], w or [1.0] * len(ms))


def family_vote(fams, split="dev", fam_w=None, rep=None):
    """within-family majority (or a representative), then a weighted vote across families."""
    fams = [f for f in fam_order if f in fams]
    preds = [(M[rep[f]][split] if rep and f in rep else hv(FAM[f], split=split)) for f in fams]
    return vote(preds, [fam_w.get(f, 1.0) if fam_w else 1.0 for f in fams])


def soft(ms, w=None, split="dev"):
    key = "dev_probs" if split == "dev" else "test_probs"
    order = docs if split == "dev" else tdocs
    w = w or [1.0] * len(ms)
    out = {}
    for d in order:
        acc = sum(wi * np.array(M[m][key][d]) for m, wi in zip(ms, w)) / sum(w)
        out[d] = [labels[i] for i in acc.argmax(1)]
    return out


# ---------------- predetermined experiments
E = {}                       # name -> (description, members/weights text, dev_pred, test_pred)
rank = sorted(runs, key=lambda n: -S[n]["dev"])


def add(name, desc, mtxt, dev_fn):
    E[name] = (desc, mtxt, dev_fn("dev"), dev_fn("test"))


add("base_anchor_single", "single model (anchor)", ANCHOR, lambda s: M[ANCHOR][s])
add("A_all27_uniform", "uniform hard vote, all 27", "all 27", lambda s: vote([M[m][s] for m in rank], [1.0] * 27))
for k in (5, 7):
    add(f"B_top{k}_uniform", f"uniform hard vote, top {k} by dev", " ".join(rank[:k]),
        lambda s, k=k: vote([M[m][s] for m in rank[:k]], [1.0] * k))
rep_dev = {f: FAM[f][0] for f in FAM}                                     # representative = best dev in family
rep_net = {f: (ANCHOR if ANCHOR in FAM[f] else max(FAM[f], key=lambda n: net[n])) for f in FAM}   # best net vs anchor
add("C1_one_rep_per_family_bestdev", "one representative per family (best dev), equal votes",
    " ".join(rep_dev[f] for f in fam_order), lambda s: family_vote(FAM, s, rep=rep_dev))
add("C2_one_rep_per_family_bestnet", "one representative per family (anchor for F1; best net-vs-anchor elsewhere), equal votes",
    " ".join(rep_net[f] for f in fam_order), lambda s: family_vote(FAM, s, rep=rep_net))
add("D_within_family_majority_equal", "majority inside each family, then equal family votes",
    "; ".join(f"{f}: majority of {len(FAM[f])}" for f in fam_order), lambda s: family_vote(FAM, s))
add("E_family_balanced_F1x2", "within-family majority; F1 family weight 2, others 1 (F1 overruled only if the other 3 families agree)",
    "F1 x2, F2/F3/F4 x1 (family majorities)", lambda s: family_vote(FAM, s, fam_w={"F1_lora_type_adapters": 2.0}))
top2 = [m for f in fam_order for m in FAM[f][:2]]
add("F_top2_per_family_uniform", "best 2 (by dev) from each family, uniform vote (8 members)", " ".join(order_members(top2)),
    lambda s: hv(top2, split=s))

# anchored (step 4): candidates ranked by net error-correction vs the anchor
cands = sorted([n for n in runs if n != ANCHOR and not (set(dev_dup[n]) & {ANCHOR})], key=lambda n: -net[n])
top6 = cands[:6]
for c in top6:                                  # anchor + 1 (2-member vote: ties -> anchor => identical to anchor)
    add(f"G1_anchor+{c}", "anchor + 1 (2-member hard vote, ties -> anchor)", f"{ANCHOR} {c}", lambda s, c=c: hv([ANCHOR, c], split=s))
for a, b in itertools.combinations(top6, 2):     # anchor + 2: anchor overruled only if both others agree on another label
    add(f"G2_anchor+{a}+{b}", "anchor + 2 (3-member majority)", f"{ANCHOR} {a} {b}",
        lambda s, a=a, b=b: hv([ANCHOR, a, b], split=s))
top4 = cands[:4]
for wa in (1.0, 1.5, 2.0):                      # step 5: anchor extra weight, 4 others x1
    ms = [ANCHOR] + order_members(top4)[0:4]
    add(f"H_anchor_w{wa}+top4net", f"anchor weight {wa} + top-4 net-vs-anchor runs x1", " ".join(ms) + f" | weights {wa},1,1,1,1",
        lambda s, ms=ms, wa=wa: vote([M[m][s] for m in ms], [wa] + [1.0] * 4))
for wa in (1.0, 1.5):                           # step 5: anchor + family-balanced votes of the other families
    add(f"I_anchor_w{wa}+family_majorities", f"anchor weight {wa} + one majority vote from each of F2/F3/F4",
        f"{ANCHOR} x{wa}; F2, F3, F4 majorities x1",
        lambda s, wa=wa: vote([M[ANCHOR][s]] + [hv(FAM[f], split=s) for f in fam_order if f != "F1_lora_type_adapters"],
                              [wa, 1.0, 1.0, 1.0]))

# probability averaging (step 6) — non-CRF runs only
noncrf = [n for n in runs if M[n]["dev_probs_decode"] == "argmax"]
assert "B_lora_crf_allep" not in noncrf
top7nc = [n for n in rank if n in noncrf][:7]
add("J1_soft_top7_uniform", "mean probabilities, top 7 non-CRF by dev", " ".join(top7nc), lambda s: soft(top7nc, split=s))
fam_nc = {f: [n for n in FAM[f] if n in noncrf] for f in FAM}


def soft_family(s):
    key, order = ("dev_probs", docs) if s == "dev" else ("test_probs", tdocs)
    out = {}
    for d in order:
        acc = np.mean([np.mean([np.array(M[m][key][d]) for m in fam_nc[f]], 0) for f in fam_order], 0)
        out[d] = [labels[i] for i in acc.argmax(1)]
    return out


add("J2_soft_family_balanced", "mean probabilities inside each family (non-CRF), then mean across the 4 families", "all non-CRF", soft_family)
top2nc = [n for n in cands if n in noncrf][:2]
add("J3_soft_anchor_heavy", "anchor probability weight 2 + top-2 non-CRF net-vs-anchor runs x1", f"{ANCHOR} x2 {top2nc[0]} {top2nc[1]}",
    lambda s: soft([ANCHOR] + top2nc, [2.0, 1.0, 1.0], split=s))

# ---------------- score everything
gD = {D: {k: v for k, v in gold.items() if ctx["domain"][k] == D} for D in ("CL", "IT")}
res = {}
for name, (desc, mtxt, dp, tp) in E.items():
    s = scores(ctx, dp)
    res[name] = s
base = res["base_anchor_single"]
with open(os.path.join(OUT, "ensemble_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "description", "members_weights", "dev", "CL", "IT", "macro_P", "macro_R", "d_dev_vs_anchor",
                "d_CL_vs_anchor", "d_IT_vs_anchor", "dev_sentences_changed_vs_anchor", "test_sentences_changed_vs_anchor"])
    for name, (desc, mtxt, dp, tp) in E.items():
        s = res[name]
        cd = sum(a != b for a, b in zip(flat(ctx, dp), A_dev))
        ct = sum(a != b for a, b in zip(flat(ctx, tp, "test"), flat(ctx, M[ANCHOR]["test"], "test")))
        w.writerow([name, desc, mtxt, f"{s['dev']:.4f}", f"{s['CL']:.4f}", f"{s['IT']:.4f}", f"{s['P']:.4f}", f"{s['R']:.4f}",
                    f"{s['dev'] - base['dev']:+.4f}", f"{s['CL'] - base['CL']:+.4f}", f"{s['IT'] - base['IT']:+.4f}", cd, ct])
with open(os.path.join(OUT, "per_class_results.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method"] + labels)
    for name in E:
        w.writerow([name] + [res[name]["per"].get(l, {}).get("f1", "") for l in labels])

# calibration (step 6 check)
flatg = np.array([labels.index(l) for l in gold_flat])
cal = []
for n in noncrf:
    p = np.array([r for d in docs for r in M[n]["dev_probs"][d]]); conf = p.max(1); ok = p.argmax(1) == flatg
    b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(conf[b == k].mean() - ok[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any())
    cal.append((n, family(n), conf.mean(), ok.mean(), ece))

# ---------------- step 7: stability (per document, leave-one-document-out) for the strongest of each group
groups = {"family_balanced": [k for k in E if k[0] in "ACDEF" and k != "A_all27_uniform"] + ["A_all27_uniform"],
          "anchor_plus_one": [k for k in E if k.startswith("G1_")], "anchor_plus_two": [k for k in E if k.startswith("G2_")],
          "anchor_weighted": [k for k in E if k[0] in "HI"], "probability": [k for k in E if k.startswith("J")],
          "top_k_reference": [k for k in E if k.startswith("B_")]}
best = {g: max(ks, key=lambda k: res[k]["dev"]) for g, ks in groups.items() if ks}
stab_rows = []
for g, k in best.items():
    dp = E[k][2]
    per_doc_wins = 0
    for d in docs:
        sc_c = rr_scores({d: gold[d]}, {d: dp[d]}, labels)["macro_f1"]
        sc_a = rr_scores({d: gold[d]}, {d: M[ANCHOR]["dev"][d]}, labels)["macro_f1"]
        loo_c = rr_scores({x: gold[x] for x in docs if x != d}, {x: dp[x] for x in docs if x != d}, labels)["macro_f1"]
        loo_a = rr_scores({x: gold[x] for x in docs if x != d}, {x: M[ANCHOR]["dev"][x] for x in docs if x != d}, labels)["macro_f1"]
        per_doc_wins += sc_c > sc_a
        stab_rows.append({"group": g, "method": k, "document": d, "domain": ctx["domain"][d], "doc_macro_f1": round(sc_c, 4),
                          "anchor_doc_macro_f1": round(sc_a, 4), "loo_macro_f1(without this doc)": round(loo_c, 4),
                          "anchor_loo_macro_f1": round(loo_a, 4), "loo_gain_vs_anchor": round(loo_c - loo_a, 4)})
with open(os.path.join(OUT, "document_stability.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(stab_rows[0])); wr.writeheader(); wr.writerows(stab_rows)

# ---------------- step 9: candidate zips (best of each group, if it beats the anchor on combined dev)
existing = {}
for zp in [os.path.expanduser(p) for p in ("~/capstone/Phase3 Submissions/*.zip", "~/Downloads/submission_*.zip",
                                           "~/capstone/phase3/ens_exp_out/*.zip")]:
    import glob
    for z in glob.glob(zp):
        try:
            existing[os.path.basename(z)] = json.loads(zipfile.ZipFile(z).read("prediction.json"))
        except Exception:
            pass
for n in runs:
    existing[f"submission_{n}.zip (allep run)"] = M[n]["test"]
cand_dir = os.path.join(OUT, "candidates")
os.makedirs(cand_dir, exist_ok=True)
summary = {}
for g, k in best.items():
    s = res[k]
    tp = E[k][3]
    assert set(tp) == set(ctx["tnsent"]) and all(len(tp[d]) == ctx["tnsent"][d] for d in tp)
    assert {l for v in tp.values() for l in v} <= set(labels)
    ft = flat(ctx, tp, "test")
    nearest = min(existing, key=lambda e: sum(a != b for a, b in zip(ft, flat(ctx, existing[e], "test"))))
    nd = sum(a != b for a, b in zip(ft, flat(ctx, existing[nearest], "test")))
    lw = sum(r["loo_gain_vs_anchor"] > 0 for r in stab_rows if r["method"] == k)
    dw = sum(r["doc_macro_f1"] > r["anchor_doc_macro_f1"] for r in stab_rows if r["method"] == k)
    zpath = None
    if g != "top_k_reference" and s["dev"] > base["dev"] and nd > 0:
        zpath = os.path.join(cand_dir, f"submission_{k}.zip")
        json.dump(tp, open(os.path.join(cand_dir, f"{k}_prediction.json"), "w"))
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("prediction.json", json.dumps(tp))
    summary[g] = dict(method=k, desc=E[k][0], members=E[k][1], dev=s["dev"], CL=s["CL"], IT=s["IT"], P=s["P"], R=s["R"],
                      per={l: s["per"].get(l, {}).get("f1") for l in labels}, loo_wins=lw, doc_wins=dw,
                      nearest_existing=nearest, test_diff_to_nearest=nd, zip=zpath)
json.dump({"anchor": {"dev": base["dev"], "CL": base["CL"], "IT": base["IT"]}, "families": FAM, "calibration": cal,
           "net_vs_anchor": net, "best_per_group": summary}, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)

print(f"{len(E)} predetermined configurations evaluated")
print(f"anchor: dev {base['dev']:.4f} CL {base['CL']:.4f} IT {base['IT']:.4f}")
for g, v in summary.items():
    print(f"[{g}] {v['method']}: dev {v['dev']:.4f} ({v['dev'] - base['dev']:+.4f}) CL {v['CL']:.4f} ({v['CL'] - base['CL']:+.4f}) "
          f"IT {v['IT']:.4f} ({v['IT'] - base['IT']:+.4f}) | LOO wins {v['loo_wins']}/10 | doc wins {v['doc_wins']}/10 | "
          f"nearest existing {v['nearest_existing']} ({v['test_diff_to_nearest']} test sents differ) | zip {'yes' if v['zip'] else 'no'}")
print("calibration (non-CRF): ", [(n, f, round(c, 3), round(a, 3), round(e, 3)) for n, f, c, a, e in cal][:3], "...")
