"""Strict span metrics for L-NER (plumbing; declare it).

strict match = same start, same end, same label (spec §8.3.5 / §8.4).
Primary: macro-F1 over the 12 labels (per-label F1 averaged; a label with no gold and no predictions is skipped,
a label with gold but no predictions counts as F1 = 0). Also reported: micro P/R/F1 and error counts
(boundary error = right label, overlapping but not identical span; label error = identical span, wrong label).
Counts only: interpreting them is up to you (course rule 9).
"""
from collections import Counter, defaultdict


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def strict_scores(gold, pred, labels):
    """gold, pred: {doc_id: [(start, end, label), ...]} (same end convention). Docs missing from pred = no spans."""
    tp, fp, fn = Counter(), Counter(), Counter()
    boundary, label_err, confusion = Counter(), Counter(), Counter()
    for doc_id, g_spans in gold.items():
        g = set(map(tuple, g_spans))
        p = set(map(tuple, pred.get(doc_id, [])))
        for s in g & p:
            tp[s[2]] += 1
        for s in p - g:
            fp[s[2]] += 1
        for s in g - p:
            fn[s[2]] += 1
        g_by_span = {(s, e): lab for s, e, lab in g}
        for s, e, lab in p - g:
            if (s, e) in g_by_span:
                label_err[lab] += 1
                confusion[(g_by_span[(s, e)], lab)] += 1
            elif any(lab == gl and s < ge and gs < e for gs, ge, gl in g):
                boundary[lab] += 1
    per = {}
    for lab in labels:
        if tp[lab] + fp[lab] + fn[lab] == 0:
            continue
        p, r, f = _prf(tp[lab], fp[lab], fn[lab])
        per[lab] = dict(p=round(p, 4), r=round(r, 4), f1=round(f, 4), tp=tp[lab], fp=fp[lab], fn=fn[lab],
                        boundary_err=boundary[lab], label_err=label_err[lab])
    n = max(1, len(per))
    mp, mr, mf = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {
        "macro_f1": round(sum(v["f1"] for v in per.values()) / n, 4),
        "macro_p": round(sum(v["p"] for v in per.values()) / n, 4),
        "macro_r": round(sum(v["r"] for v in per.values()) / n, 4),
        "micro_p": round(mp, 4), "micro_r": round(mr, 4), "micro_f1": round(mf, 4),
        "per_label": per,
        "confusion_gold_to_pred": {f"{a}->{b}": c for (a, b), c in confusion.most_common()},
    }
