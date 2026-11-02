#!/usr/bin/env python3
"""Local scorer for Phase 1 continuations (CS781 capstone).

Scores the val_generations_<strategy>.json files written by the training notebooks
({id: {"prediction": str, "reference": str}}) with the leaderboard metrics plus repetition stats,
so decoding strategies / models can be ranked before spending Codabench submissions.

Metrics
  BERTScore F1 (primary)  bert_score, default lang="en" -> roberta-large, no baseline rescaling.
                          The official evaluation script is on the course Drive; if it uses another
                          model / layer / rescaling, pass --bert-model / --num-layers / --rescale.
  ROUGE-L F1              rouge_score (no stemmer by default; --stem to enable)
  BLEU-4                  sacrebleu, reported on a 0-1 scale: corpus BLEU and mean sentence BLEU
  repetition              loop%: share of outputs where some 4-word sequence occurs 3+ times
                          (same definition as the notebooks' looping_share);
                          onset: median word index of the first repeated 4-word sequence;
                          words: mean output length in words (target 128)

Usage
  python score_local.py run_C/val_generations_*.json run_D/val_generations_*.json
  python score_local.py ~/Downloads/output_dir            # finds val_generations_*.json recursively
  python score_local.py DIR --csv scores.csv --device cuda

Install (CPU is fine for a few hundred pairs):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install bert-score rouge-score sacrebleu
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys


def find_files(paths):
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "val_generations_*.json"), recursive=True)
        else:
            files += glob.glob(p)
    return sorted(set(files))


def label_for(path):
    """run_C/val_generations_mild.json -> C/mild"""
    name = os.path.basename(path)[len("val_generations_"):-len(".json")]
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    run = run[len("run_"):] if run.startswith("run_") else run
    return f"{run}/{name}"


def load_pairs(path):
    data = json.load(open(path, encoding="utf-8"))
    keys = sorted(data, key=lambda k: (len(k), k))
    preds = [" ".join(str(data[k]["prediction"]).split()) for k in keys]
    refs = [" ".join(str(data[k]["reference"]).split()) for k in keys]
    return keys, preds, refs


def four_gram_stats(text):
    """(loops, onset): loops = some 4-gram occurs 3+ times; onset = word index of the first repeat (None if none)."""
    w = text.split()
    counts, onset = {}, None
    for i in range(len(w) - 3):
        g = tuple(w[i:i + 4])
        if g in counts and onset is None:
            onset = i
        counts[g] = counts.get(g, 0) + 1
    return (bool(counts) and max(counts.values()) >= 3), onset


def repetition(texts):
    stats = [four_gram_stats(t) for t in texts]
    onsets = [o for _, o in stats if o is not None]
    return {
        "loop_pct": 100.0 * sum(l for l, _ in stats) / max(1, len(texts)),
        "onset_median": statistics.median(onsets) if onsets else None,
        "words_mean": statistics.mean(len(t.split()) for t in texts) if texts else 0.0,
    }


def mean_se(xs):
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, se


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="val_generations_*.json files, globs, or directories")
    ap.add_argument("--bert-model", default=None, help="BERTScore model_type (default: bert_score's choice for --lang)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--rescale", action="store_true", help="rescale_with_baseline=True")
    ap.add_argument("--stem", action="store_true", help="ROUGE-L with Porter stemmer")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-bertscore", action="store_true", help="skip BERTScore (fast check of the other metrics)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = ap.parse_args()

    files = find_files(args.paths)
    if not files:
        sys.exit("no val_generations_*.json files found")

    from rouge_score import rouge_scorer
    import sacrebleu

    runs = []
    for f in files:
        keys, preds, refs = load_pairs(f)
        runs.append({"label": label_for(f), "file": f, "keys": keys, "preds": preds, "refs": refs})
    print(f"{len(runs)} file(s): " + ", ".join(f"{r['label']} ({len(r['preds'])})" for r in runs))

    # BERTScore: one scorer (model loaded once) for every file
    if not args.no_bertscore:
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type=args.bert_model, lang=args.lang, num_layers=args.num_layers,
                            rescale_with_baseline=args.rescale, device=args.device, batch_size=args.batch_size)
        print(f"BERTScore: {scorer.model_type} | layers {scorer.num_layers} | rescale {args.rescale} | "
              f"device {scorer.device}")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=args.stem)

    # the human references themselves, as a repetition baseline (same refs in every file)
    ref_rep = repetition(runs[0]["refs"])

    rows = []
    for r in runs:
        row = {"label": r["label"], "n": len(r["preds"])}
        if not args.no_bertscore:
            _, _, f1 = scorer.score(r["preds"], r["refs"])
            row["bertscore_f1"], row["bertscore_se"] = mean_se(f1.tolist())
        rl = [rouge.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(r["preds"], r["refs"])]
        row["rougeL_f1"], _ = mean_se(rl)
        row["bleu4_corpus"] = sacrebleu.corpus_bleu(r["preds"], [r["refs"]]).score / 100
        row["bleu4_sent_mean"] = statistics.mean(
            sacrebleu.sentence_bleu(p, [ref]).score / 100 for p, ref in zip(r["preds"], r["refs"]))
        row.update(repetition(r["preds"]))
        rows.append(row)

    key = "bertscore_f1" if not args.no_bertscore else "rougeL_f1"
    rows.sort(key=lambda x: -x[key])

    hdr = f"{'model/strategy':<22}{'n':>5}"
    if not args.no_bertscore:
        hdr += f"{'BERTScore F1':>17}"
    hdr += f"{'ROUGE-L':>9}{'BLEU-4':>8}{'sentBLEU':>9}{'loop%':>7}{'onset':>7}{'words':>7}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for x in rows:
        line = f"{x['label']:<22}{x['n']:>5}"
        if not args.no_bertscore:
            line += f"{x['bertscore_f1']:>10.4f} ±{x['bertscore_se']:.4f}"
        onset = "-" if x["onset_median"] is None else f"{x['onset_median']:.0f}"
        line += (f"{x['rougeL_f1']:>9.4f}{x['bleu4_corpus']:>8.4f}{x['bleu4_sent_mean']:>9.4f}"
                 f"{x['loop_pct']:>6.0f}%{onset:>7}{x['words_mean']:>7.0f}")
        print(line)
    onset = "-" if ref_rep["onset_median"] is None else f"{ref_rep['onset_median']:.0f}"
    pad = 5 + (17 if not args.no_bertscore else 0) + 9 + 8 + 9
    print(f"{'(human references)':<22}{'':>{pad}}{ref_rep['loop_pct']:>6.0f}%{onset:>7}{ref_rep['words_mean']:>7.0f}")
    if not args.no_bertscore:
        print("\n± is the standard error of the mean over prompts; gaps smaller than ~2x the larger ± are noise.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
