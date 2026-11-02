#!/usr/bin/env python3
"""Audit predicted spans against the raw text and the gold annotation conventions (plumbing; declare it).

  python audit_spans.py --data data --split-seed 42 --val-frac 0.15 --pred NAME=val_predictions.json [NAME=...]
                        [--test NAME=submission.zip ...]

Validation predictions (scored against gold):
  * format: inside the document, non-empty, no leading/trailing whitespace, label in labels.txt
  * boundary hygiene: span starts inside a word (char before and first char both alphanumeric) / ends inside a word
  * conventions, gold vs predicted, per label: share of spans ending in punctuation, starting with a lowercase
    letter, starting with "In "/"The ", containing "(" or ")", mean length in characters
Test predictions (no gold): format + boundary hygiene only.
Counts only: what to change because of them is your decision (course rule 9).
"""
import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_data   # noqa: E402

PUNCT = set(".,;:)]}'\"")


def read(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("prediction.json"))
    return json.load(open(path))


def hygiene(texts, pred, labels):
    c = defaultdict(int)
    for d, spans in pred.items():
        t = texts[d]
        for s, e, lab in (x[:3] for x in spans):
            c["spans"] += 1
            if not (0 <= s < e <= len(t)):
                c["out_of_bounds_or_empty"] += 1
                continue
            frag = t[s:e]
            c["label_not_in_labels"] += lab not in labels
            c["leading_or_trailing_space"] += frag != frag.strip()
            c["starts_mid_word"] += s > 0 and t[s - 1].isalnum() and t[s].isalnum()
            c["ends_mid_word"] += e < len(t) and t[e - 1].isalnum() and t[e].isalnum()
    return dict(c)


def conventions(texts, spans_by_doc):
    agg = defaultdict(lambda: defaultdict(float))
    for d, spans in spans_by_doc.items():
        t = texts[d]
        for s, e, lab in (x[:3] for x in spans):
            f = t[s:e]
            a = agg[lab]
            a["n"] += 1
            a["ends_punct"] += f[-1:] in PUNCT
            a["starts_lower"] += f[:1].islower()
            a["starts_In_The"] += f.startswith(("In ", "The "))
            a["has_paren"] += ("(" in f) or (")" in f)
            a["chars"] += len(f)
    out = {}
    for lab, a in agg.items():
        n = a["n"]
        out[lab] = {"n": int(n), "ends_punct%": round(100 * a["ends_punct"] / n, 1),
                    "starts_lower%": round(100 * a["starts_lower"] / n, 1),
                    "starts_In/The%": round(100 * a["starts_In_The"] / n, 1),
                    "has_paren%": round(100 * a["has_paren"] / n, 1), "mean_chars": round(a["chars"] / n, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--test", nargs="*", default=[])
    args = ap.parse_args()
    texts, gold, fold_of, labels = ner_data.load_train(args.data, verbose=False)
    test_texts = ner_data.load_test(args.data)
    _, val_ids = ner_data.split_docs(list(texts), fold_of, {"mode": "random", "val_frac": args.val_frac,
                                                            "seed": args.split_seed})
    gold_val = {d: gold[d] for d in val_ids}

    print("== FORMAT + BOUNDARY HYGIENE (validation predictions) ==")
    print(f"{'run':22s}" + "".join(f"{k:>24s}" for k in ("spans", "out_of_bounds_or_empty", "leading_or_trailing_space",
                                                          "starts_mid_word", "ends_mid_word", "label_not_in_labels")))
    hy_gold = hygiene(texts, gold_val, labels)
    rows = [("GOLD (val)", hy_gold)]
    preds = {}
    for item in args.pred:
        name, path = item.split("=", 1)
        preds[name] = read(path)
        assert set(preds[name]) == set(val_ids), f"{name}: documents differ from the validation split"
        rows.append((name, hygiene(texts, preds[name], labels)))
    for name, h in rows:
        print(f"{name:22s}" + "".join(f"{h.get(k, 0):>24d}" for k in ("spans", "out_of_bounds_or_empty",
              "leading_or_trailing_space", "starts_mid_word", "ends_mid_word", "label_not_in_labels")))
    if args.test:
        print("\n== FORMAT + BOUNDARY HYGIENE (test submissions, no gold) ==")
        for item in args.test:
            name, path = item.split("=", 1)
            p = read(path)
            h = hygiene(test_texts, p, labels)
            print(f"{name:22s} docs {len(p)}/35 | " + " | ".join(f"{k} {v}" for k, v in h.items()))

    print("\n== CONVENTIONS per label: GOLD (val) vs each run's predictions ==")
    cg = conventions(texts, gold_val)
    cp = {n: conventions(texts, p) for n, p in preds.items()}
    keys = ["n", "ends_punct%", "starts_lower%", "starts_In/The%", "has_paren%", "mean_chars"]
    for lab in labels:
        print(f"\n{lab}")
        print(f"  {'':20s}" + "".join(f"{k:>16s}" for k in keys))
        print(f"  {'GOLD':20s}" + "".join(f"{cg.get(lab, {}).get(k, 0):>16}" for k in keys))
        for n in cp:
            print(f"  {n:20s}" + "".join(f"{cp[n].get(lab, {}).get(k, 0):>16}" for k in keys))


if __name__ == "__main__":
    main()
