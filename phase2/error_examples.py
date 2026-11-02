#!/usr/bin/env python3
"""Concrete strict-NER error examples from a finished run (plumbing for error analysis; declare it).

Reads <run>/val_predictions.json + <run>/config.json (for the validation split), re-scores against the gold spans
(and checks the counts match <run>/val_scores.json), then prints for each requested label:
  BOUNDARY  gold span vs overlapping predicted span with the same label (strict mismatch)
  LABEL     exact span, different label
  SPURIOUS  predicted span with no overlapping gold span of that label
  MISSED    gold span with no overlapping prediction of that label
each with surrounding text; [[...]] marks the gold span, <<...>> the predicted span.
Counts and examples only: drawing conclusions from them is yours (course rule 9).

  python error_examples.py --run runs/B_full --data data --labels PREC WIT --n 5
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_data                         # noqa: E402
from ner_metrics import strict_scores   # noqa: E402


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def context(text, spans, width):
    """Text around the union of `spans`, with gold marked [[ ]] and prediction << >>."""
    lo = max(0, min(s for s, _, _ in spans) - width)
    hi = min(len(text), max(e for _, e, _ in spans) + width)
    marks = []
    for s, e, kind in spans:
        marks += [(s, 0, "[[" if kind == "gold" else "<<"), (e, 1, "]]" if kind == "gold" else ">>")]
    out, pos = [], lo
    for p, _, m in sorted(marks):
        out.append(text[pos:p])
        out.append(m)
        pos = p
    out.append(text[pos:hi])
    return ("…" if lo > 0 else "") + "".join(out).replace("\n", " ") + ("…" if hi < len(text) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--width", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    texts, gold, fold_of, labels = ner_data.load_train(args.data, verbose=False)
    pred = {d: [tuple(x[:3]) for x in v] for d, v in json.load(open(os.path.join(args.run, "val_predictions.json"))).items()}
    _, val_ids = ner_data.split_docs(list(texts), fold_of, cfg["split"])
    assert set(pred) == set(val_ids), "val_predictions.json docs differ from the run's validation split"
    g_val = {d: gold[d] for d in val_ids}

    sc = strict_scores(g_val, pred, labels)
    vs_path = os.path.join(args.run, "val_scores.json")
    if os.path.exists(vs_path):
        ref = json.load(open(vs_path))
        assert abs(ref["macro_f1"] - sc["macro_f1"]) < 1e-6, (ref["macro_f1"], sc["macro_f1"])
        print(f"re-scored {len(val_ids)} validation docs: macro-F1 {sc['macro_f1']} (matches val_scores.json)")

    rng = random.Random(args.seed)
    for lab in args.labels:
        buckets = {"BOUNDARY": [], "LABEL": [], "SPURIOUS": [], "MISSED": []}
        for d in val_ids:
            g = set(s for s in g_val[d])
            p = set(pred.get(d, []))
            g_lab = [s for s in g if s[2] == lab]
            p_lab = [s for s in p if s[2] == lab]
            for ps in p_lab:
                if ps in g:
                    continue
                same_span = [s for s in g if s[:2] == ps[:2]]
                ov = [s for s in g_lab if overlaps(s, ps)]
                if same_span:
                    buckets["LABEL"].append((d, [(*same_span[0][:2], "gold"), (*ps[:2], "pred")],
                                             f"gold {same_span[0][2]} -> pred {lab}"))
                elif ov:
                    buckets["BOUNDARY"].append((d, [(*ov[0][:2], "gold"), (*ps[:2], "pred")],
                                                f"gold {ov[0][:2]} vs pred {ps[:2]}"))
                else:
                    buckets["SPURIOUS"].append((d, [(*ps[:2], "pred")], f"pred {ps[:2]}"))
            for gs in g_lab:
                if gs not in p and not any(overlaps(gs, ps) for ps in p_lab):
                    buckets["MISSED"].append((d, [(*gs[:2], "gold")], f"gold {gs[:2]}"))
        per = sc["per_label"].get(lab, {})
        print(f"\n{'=' * 100}\n{lab}: TP {per.get('tp')} FP {per.get('fp')} FN {per.get('fn')} | "
              + " ".join(f"{k} {len(v)}" for k, v in buckets.items()))
        for kind, items in buckets.items():
            if not items:
                continue
            print(f"\n--- {kind} ({len(items)}), {min(args.n, len(items))} random examples ---")
            for d, spans, note in rng.sample(items, min(args.n, len(items))):
                t = texts[d]
                strs = "  ".join(f"{k}: \"{t[s:e]}\"" for s, e, k in spans)
                print(f"* doc {d} | {note}\n  {strs}\n  {context(t, spans, args.width)}")


if __name__ == "__main__":
    main()
