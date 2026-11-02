"""Sentence-level rhetorical-role metrics (plumbing; declare it). Counts only: interpretation is yours (rule 9).

macro over labels: per-label P/R/F1 averaged over the labels that occur in gold or predictions (the usual
scikit-learn behaviour; a label with gold but no predictions counts as F1 0). Also micro-F1 (= accuracy here),
per-label support, and the confusion matrix (gold -> predicted counts).
"""
from collections import Counter


def rr_scores(gold, pred, labels):
    """gold, pred: {doc_id: [label, ...]} with equal lengths per doc."""
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    conf = Counter()
    n = correct = 0
    for d, g in gold.items():
        p = pred[d]
        assert len(p) == len(g), f"doc {d}: {len(p)} predictions for {len(g)} sentences"
        for a, b in zip(g, p):
            n += 1
            support[a] += 1
            conf[(a, b)] += 1
            if a == b:
                tp[a] += 1
                correct += 1
            else:
                fp[b] += 1
                fn[a] += 1
    present = [l for l in labels if support[l] or tp[l] or fp[l]]
    per = {}
    for l in present:
        p = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        r = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[l] = dict(p=round(p, 4), r=round(r, 4), f1=round(f, 4), support=support[l], tp=tp[l], fp=fp[l], fn=fn[l])
    k = max(1, len(per))
    return {
        "macro_f1": round(sum(v["f1"] for v in per.values()) / k, 4),
        "macro_p": round(sum(v["p"] for v in per.values()) / k, 4),
        "macro_r": round(sum(v["r"] for v in per.values()) / k, 4),
        "micro_f1": round(correct / max(1, n), 4),
        "sentences": n,
        "per_label": per,
        "confusion": {f"{a}->{b}": c for (a, b), c in conf.most_common() if a != b},
    }
