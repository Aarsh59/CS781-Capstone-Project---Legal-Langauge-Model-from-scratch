#!/usr/bin/env python3
"""Apply the optional word-boundary snapping to an EXISTING prediction file or submission zip (plumbing; declare it).

Writes a NEW file next to the output dir; the input is never modified.
  python snap_spans.py --data data --split test --in submission_X.zip --out-dir snapped/   -> snapped/submission_X_snap.zip
  python snap_spans.py --data data --split val  --in X_val_predictions.json --out-dir snapped/ [--report]
--report (val only): strict scores before/after, number of changed spans, label changes (must be 0).
"""
import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_align                        # noqa: E402
import ner_data                         # noqa: E402
from ner_metrics import strict_scores   # noqa: E402


def read(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("prediction.json"))
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    texts, gold, fold_of, labels = ner_data.load_train(args.data, verbose=False)
    if args.split == "test":
        texts = ner_data.load_test(args.data)
    pred = {d: [tuple(x[:3]) for x in v] for d, v in read(args.inp).items()}
    snapped = {d: ner_align.snap_to_words(texts[d], v) for d, v in pred.items()}
    changed = sum(len(set(pred[d]) - set(snapped[d])) for d in pred)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.basename(args.inp)
    stem = base[:-4] if base.endswith(".zip") else base[:-5]
    sub = {d: [list(s) for s in v] for d, v in snapped.items()}
    if args.inp.endswith(".zip"):
        out = os.path.join(args.out_dir, f"{stem}_snap.zip")
        assert os.path.abspath(out) != os.path.abspath(args.inp)
        tmp = os.path.join(args.out_dir, "prediction.json")
        json.dump(sub, open(tmp, "w"))
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(tmp, arcname="prediction.json")
        os.remove(tmp)
    else:
        out = os.path.join(args.out_dir, f"{stem}_snap.json")
        json.dump(sub, open(out, "w"))
    n_in, n_out = sum(map(len, pred.values())), sum(map(len, snapped.values()))
    print(f"{base}: spans {n_in} -> {n_out} | changed/merged {changed} | written {out}")
    if args.report and args.split == "val":
        _, val_ids = ner_data.split_docs(list(texts), fold_of, {"mode": "random", "val_frac": 0.15, "seed": 42})
        gv = {d: gold[d] for d in val_ids}
        a, b = strict_scores(gv, pred, labels), strict_scores(gv, snapped, labels)
        print(f"   val macro-F1 {a['macro_f1']} -> {b['macro_f1']} | micro {a['micro_f1']} -> {b['micro_f1']} | "
              f"P {a['macro_p']} -> {b['macro_p']} | R {a['macro_r']} -> {b['macro_r']}")


if __name__ == "__main__":
    main()
