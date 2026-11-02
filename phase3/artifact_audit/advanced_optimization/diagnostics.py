#!/usr/bin/env python3
"""Measurements only (no tuned parameters): items 1, 6, 7 and the metric-convention sensitivity.

Outputs: candidate_complementarity.csv, candidate_complementarity_by_class.csv, candidate_complementarity_by_document.csv,
transitions.csv, transition_errors.csv, cci_noida_confusions.csv, cci_noida_per_class.csv, cl_documents_compared.csv,
metric_sensitivity.csv
"""
import csv
import os
from collections import Counter, defaultdict

from common import ANCHOR, FIXED, HERE, HP, dom, docs, gold, labels, mf1, names, score


def dump(fn, rows):
    with open(os.path.join(HERE, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)


# ---- 1. error-correction matrix vs the anchor (runs and ensembles)
comp, bycls, bydoc = [], [], []
for n in names:
    if n == ANCHOR:
        continue
    pd_, pc = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for k in docs:
        for a, x, g in zip(HP[ANCHOR][k], HP[n][k], gold[k]):
            if a != g and x == g:
                pd_[k][0] += 1; pc[g][0] += 1
            elif a == g and x != g:
                pd_[k][1] += 1; pc[g][1] += 1
    fx, br = sum(v[0] for v in pd_.values()), sum(v[1] for v in pd_.values())
    nets = {k: pd_[k][0] - pd_[k][1] for k in docs}
    pos_docs = sum(v > 0 for v in nets.values())
    top_share = max((v for v in nets.values() if v > 0), default=0) / max(1, sum(v for v in nets.values() if v > 0))
    comp.append({"candidate": n, "kind": "ensemble" if n.startswith(("ENS", "H_", "G2", "C1")) else "run", "fixes_anchor_errors": fx,
                 "breaks_anchor_correct": br, "net": fx - br,
                 "CL_net": sum(nets[k] for k in docs if dom[k] == "CL"), "IT_net": sum(nets[k] for k in docs if dom[k] == "IT"),
                 "documents_net_positive": pos_docs, "largest_single_document_share_of_positive_net": round(top_share, 3)})
    for k in docs:
        bydoc.append({"candidate": n, "document": k, "domain": dom[k], "fixes": pd_[k][0], "breaks": pd_[k][1], "net": nets[k]})
    for c in labels:
        if pc[c] != [0, 0]:
            bycls.append({"candidate": n, "gold_label": c, "fixes": pc[c][0], "breaks": pc[c][1], "net": pc[c][0] - pc[c][1]})
comp.sort(key=lambda r: -r["net"])
dump("candidate_complementarity.csv", comp); dump("candidate_complementarity_by_class.csv", bycls); dump("candidate_complementarity_by_document.csv", bydoc)

# ---- 6. transitions: gold vs predicted, errors at boundaries
trans = Counter(); ptrans = {m: Counter() for m in [ANCHOR] + list(FIXED.values())}
err = {m: {"at_gold_boundary": [0, 0], "inside_segment": [0, 0]} for m in ptrans}
for k in docs:
    g = gold[k]
    for i in range(1, len(g)):
        trans[(g[i - 1], g[i])] += 1
    for m in ptrans:
        p = HP[m][k]
        for i in range(1, len(p)):
            ptrans[m][(p[i - 1], p[i])] += 1
        for i in range(len(g)):
            boundary = (i > 0 and g[i] != g[i - 1]) or (i + 1 < len(g) and g[i] != g[i + 1])
            key = "at_gold_boundary" if boundary else "inside_segment"
            err[m][key][0] += p[i] != g[i]; err[m][key][1] += 1
rows = []
for (a, b), c in trans.most_common():
    rows.append({"from": a, "to": b, "gold_count": c, **{f"pred_count_{m}": ptrans[m][(a, b)] for m in ptrans}})
for (a, b) in set().union(*[set(v) for v in ptrans.values()]) - set(trans):
    rows.append({"from": a, "to": b, "gold_count": 0, **{f"pred_count_{m}": ptrans[m][(a, b)] for m in ptrans}})
dump("transitions.csv", rows)
seg = []
for m in err:
    seg.append({"method": m, **{f"{k}_error_rate": round(v[0] / v[1], 4) for k, v in err[m].items()},
                **{f"{k}_sentences": v[1] for k, v in err[m].items()},
                "gold_label_changes": sum(c for (a, b), c in trans.items() if a != b),
                "predicted_label_changes": sum(c for (a, b), c in ptrans[m].items() if a != b)})
dump("transition_errors.csv", seg)

# ---- 7. CCI Noida case study
noida = next(k for k in docs if k.startswith("CCI_Noida"))
methods = {"anchor": ANCHOR, **FIXED}
conf, pcls = [], []
for lab, m in methods.items():
    p = HP[m][noida]
    pairs = Counter((g, x) for g, x in zip(gold[noida], p) if g != x)
    for (g, x), c in pairs.most_common():
        idx = [i for i, (a, b) in enumerate(zip(gold[noida], p)) if (a, b) == (g, x)]
        conf.append({"method": lab, "gold": g, "predicted": x, "count": c, "sentence_indices": " ".join(map(str, idx))})
    s = score({noida: p}, [noida])
    for c in labels:
        if c in s["per_label"]:
            v = s["per_label"][c]
            pcls.append({"method": lab, "label": c, "precision": v["p"], "recall": v["r"], "f1": v["f1"], "support": v["support"]})
dump("cci_noida_confusions.csv", conf); dump("cci_noida_per_class.csv", pcls)
cl = []
for k in docs:
    if dom[k] != "CL":
        continue
    gc = Counter(gold[k])
    row = {"document": k, "sentences": len(gold[k]), **{f"gold_share_{c}": round(gc[c] / len(gold[k]), 3) for c in labels if gc[c]}}
    for lab, m in methods.items():
        row[f"macro_f1_{lab}"] = round(mf1({k: HP[m][k]}, [k]), 4)
        row[f"accuracy_{lab}"] = round(sum(a == b for a, b in zip(gold[k], HP[m][k])) / len(gold[k]), 4)
    cl.append(row)
dump("cl_documents_compared.csv", cl)

# ---- metric-convention sensitivity
sens = []
for lab, m in list(methods.items()) + [("ENS_candidate", "ENS_candidate")]:
    if m not in HP:
        continue
    row = {"method": lab}
    for conv in ("gold_or_pred", "gold_only", "all13"):
        row[conv] = round(mf1(HP[m], docs, conv), 4)
    row["labels_gold_or_pred"] = len(score(HP[m], docs)["per_label"])
    sens.append(row)
dump("metric_sensitivity.csv", sens)

print("complementarity (top 6 net vs anchor):", [(r["candidate"], r["net"], r["documents_net_positive"]) for r in comp[:6]])
print("runs only, best net:", [(r["candidate"], r["net"]) for r in comp if r["kind"] == "run"][:3])
print("errors at gold boundaries vs inside segments:", [(r["method"][:12], r["at_gold_boundary_error_rate"], r["inside_segment_error_rate"],
      r["at_gold_boundary_sentences"], r["predicted_label_changes"], r["gold_label_changes"]) for r in seg])
print("Noida top confusions (anchor):", [(r["gold"], r["predicted"], r["count"]) for r in conf if r["method"] == "anchor"][:6])
print("Noida top confusions (H):", [(r["gold"], r["predicted"], r["count"]) for r in conf if r["method"] == "H"][:6])
print("CL documents:", [(r["document"][:20], r["sentences"], r["macro_f1_anchor"], r["macro_f1_H"]) for r in cl])
print("metric sensitivity:", sens)
