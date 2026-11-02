#!/usr/bin/env python3
"""Local scorer for Phase-6 summaries (plumbing; declare it). Facts only: metrics + length / repetition statistics.

  python score_summ.py PRED.json [PRED2.json ...] --refs val_refs.json [--bert-model roberta-large] [--device cuda]

PRED.json: {id: summary} (summ_infer / train_summ output) or the list format [{"id", "summary"}].
--refs: {id: reference summary} (train_summ writes <run>/val_refs.json).
Metrics
  BERTScore F1 (primary)  bert_score, default roberta-large, no baseline rescaling. The official "legal BERTScore"
                          model is unknown: pass --bert-model to match it once known. Inputs longer than the model's
                          512-token limit are truncated by bert_score (most references are longer: compare runs only
                          with the same settings). --no-bert skips it.
  ROUGE-1 / ROUGE-L F1    rouge_score, with the Porter stemmer (--no-stem to disable).
  length                  mean words of predictions vs references, mean pred/ref word ratio.
  repetition              rep4: mean share of repeated word 4-grams per prediction; loop%: predictions where some
                          word 4-gram occurs 3+ times.
"""
import argparse
import json
import statistics
from collections import Counter


def load_pred(path):
    d = json.load(open(path))
    if isinstance(d, list):
        d = {x["id"]: x["summary"] for x in d}
    return {k: (v if isinstance(v, str) else " ".join(v)) for k, v in d.items()}


def rep_stats(text):
    w = text.split()
    g = Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    n = sum(g.values())
    return (sum(c - 1 for c in g.values()) / n if n else 0.0), any(c >= 3 for c in g.values())


def score(preds, refs, bert_model=None, device=None, stem=True, batch_size=16):
    ids = sorted(refs)
    missing = [i for i in ids if i not in preds]
    assert not missing, f"{len(missing)} reference ids have no prediction (e.g. {missing[:3]})"
    P = [preds[i] for i in ids]
    R = [refs[i] for i in ids]
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=stem)
    r1, rl = [], []
    for p, r in zip(P, R):
        s = sc.score(r, p)
        r1.append(s["rouge1"].fmeasure); rl.append(s["rougeL"].fmeasure)
    out = {"n": len(ids), "rouge1_f1": round(statistics.mean(r1), 4), "rougeL_f1": round(statistics.mean(rl), 4)}
    if bert_model:
        from bert_score import score as bscore
        _, _, F = bscore(P, R, model_type=bert_model, device=device, batch_size=batch_size, verbose=False)
        out["bertscore_f1"] = round(float(F.mean()), 4)
        out["bert_model"] = bert_model
    pw = [len(p.split()) for p in P]
    rw = [len(r.split()) for r in R]
    reps = [rep_stats(p) for p in P]
    out.update(pred_words=round(statistics.mean(pw), 1), ref_words=round(statistics.mean(rw), 1),
               len_ratio=round(statistics.mean(a / max(1, b) for a, b in zip(pw, rw)), 3),
               empty=sum(not p.strip() for p in P),
               rep4=round(statistics.mean(r for r, _ in reps), 4),
               loop_pct=round(100 * sum(l for _, l in reps) / len(reps), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--bert-model", default="roberta-large")
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--no-stem", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--out", help="append one JSON line per file")
    a = ap.parse_args()
    refs = json.load(open(a.refs))
    for p in a.preds:
        preds = load_pred(p)
        res = {"file": p, **score(preds, refs, None if a.no_bert else a.bert_model, a.device, not a.no_stem)}
        print(json.dumps(res), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
