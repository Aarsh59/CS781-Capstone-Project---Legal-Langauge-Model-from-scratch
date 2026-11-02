#!/usr/bin/env python3
"""Ensemble several finished runs into one submission (plumbing; declare it). Which runs and which mode: yours.

  python ensemble.py --name ENS --runs runs/B_full runs/B_lr3e5 ... --mode prob   --out runs
  python ensemble.py --name ENS --runs runs/B_full runs/B_lr3e5 ... --mode vote --min-votes 2 --out runs

prob : loads every member (config.json + best_model.pt), computes its merged per-token logits on the validation and
       test documents with that member's own window settings, averages the softmax probabilities over members, then
       decodes once (argmax + BIO->spans with the first member's decode.repair). Members must share the tokenizer,
       preprocessing (split_punct, tag scheme) and therefore the token sequence. Needs the checkpoints.
vote : reads each member's val_predictions.json / prediction.json and keeps a span (start, end, label) whose summed
       member weight (--weights, default 1 each) is >= --min-votes; overlapping kept spans are resolved by higher
       vote, then longer span. No models. Members may also be submission_*.zip files (test only, no val score):
         python ensemble.py --name ENS --runs a.zip b.zip c.zip --mode vote --weights 1 1 2 --min-votes 2 --out out

Writes <out>/<name>/: val_scores.json, val_predictions.json, prediction.json, submission_<name>.zip, summary.json.
"""
import argparse
import json
import os
import sys
import time
import zipfile
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_align                        # noqa: E402
import ner_data                         # noqa: E402
import train_ner                        # noqa: E402
from a1_tokenizer import ByteBPE        # noqa: E402
from ner_metrics import strict_scores   # noqa: E402


def load_cfg(run):
    return json.load(open(os.path.join(run, "config.json")))


def check_compatible(cfgs):
    ref = cfgs[0]
    for c in cfgs[1:]:
        assert c["preprocess"] == ref["preprocess"], f"{c['name']}: preprocess differs from {ref['name']}"
        assert c["split"] == ref["split"], f"{c['name']}: validation split differs from {ref['name']}"
        assert c["tokenizer"] == ref["tokenizer"] or os.path.basename(c["tokenizer"]) == os.path.basename(ref["tokenizer"])


def prob_ensemble(runs, cfgs, texts, gold, val_ids, test_texts, labels, device):
    ref = cfgs[0]
    tok = ByteBPE.from_json(ref["tokenizer"])
    scheme = ref["preprocess"]["tag_scheme"]
    tags = ner_align.tag_list(labels, scheme)
    t2i, i2t = {t: i for i, t in enumerate(tags)}, dict(enumerate(tags))
    otok = ner_align.OffsetTokenizer(tok, split_punct=ref["preprocess"]["split_punct"])
    enc_val, _ = train_ner.encode_docs({d: texts[d] for d in val_ids}, gold, otok, t2i, scheme)
    enc_test, _ = train_ner.encode_docs(test_texts, None, otok, t2i, scheme)
    amp = device.type == "cuda"
    sums = {"val": {}, "test": {}}
    for run, cfg in zip(runs, cfgs):
        t0 = time.time()
        model, _, mtags = train_ner.load_run(run, device, tok.get_vocab_size())
        assert mtags == tags, f"{cfg['name']}: tag set differs"
        assert not hasattr(model.head, "decode"), f"{cfg['name']}: CRF heads are not supported in prob mode (use vote)"
        w = cfg["window"]
        micro = cfg["train"].get("infer_micro_batch") or cfg["train"]["micro_batch"] * 2
        for split, enc, ids in (("val", enc_val, val_ids), ("test", enc_test, list(test_texts))):
            lg = train_ner.doc_logits(model, enc, [d for d in ids if len(enc[d]["ids"])], w["infer_len"],
                                      w["infer_stride"], w["merge"], micro, device, amp)
            for d, x in lg.items():
                p = torch.softmax(x, -1)
                sums[split][d] = sums[split][d] + p if d in sums[split] else p
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  member {cfg['name']}: done in {time.time() - t0:.0f}s", flush=True)

    def decode(split, enc, ids):
        out = {}
        for d in ids:
            if d not in sums[split]:
                out[d] = []
                continue
            tag_ids = sums[split][d].argmax(-1).numpy()
            out[d] = ner_align.tags_to_spans(tag_ids, enc[d]["starts"], enc[d]["ends"], i2t, ref["decode"]["repair"])
        return out
    return decode("val", enc_val, val_ids), decode("test", enc_test, list(test_texts))


def read_pred(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("prediction.json"))
    return json.load(open(path))


def vote_spans(preds, weights, min_votes, label_min_votes=None):
    """One document: preds = list of member span lists. Keep (s, e, label) whose summed weight >= the threshold for its
    label (label_min_votes overrides min_votes per label); overlaps resolved by higher vote, then longer span."""
    label_min_votes = label_min_votes or {}
    votes = Counter()
    for p, w in zip(preds, weights):
        for s in set(tuple(x[:3]) for x in p):
            votes[s] += w
    cand = sorted((s for s, v in votes.items() if v >= label_min_votes.get(s[2], min_votes)),
                  key=lambda s: (-votes[s], -(s[1] - s[0]), s[0]))
    kept = []
    for s in cand:
        if all(not (s[0] < k[1] and k[0] < s[1]) for k in kept):
            kept.append(s)
    return sorted(kept)


def vote_ensemble(runs, min_votes, weights=None, label_min_votes=None):
    weights = weights or [1.0] * len(runs)

    def combine(files):
        preds = [read_pred(f) for f in files]
        docs = set().union(*preds)
        out = {}
        for d in docs:
            out[d] = vote_spans([p.get(d, []) for p in preds], weights, min_votes, label_min_votes)
        return out
    if all(r.endswith(".zip") for r in runs):                       # test predictions only
        return None, combine(runs)
    return (combine([os.path.join(r, "val_predictions.json") for r in runs]),
            combine([os.path.join(r, "prediction.json") for r in runs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mode", choices=("prob", "vote"), required=True)
    ap.add_argument("--min-votes", type=float, default=None)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--snap", action="store_true", help="optional: snap the ensemble's spans to word boundaries")
    ap.add_argument("--label-min-votes", nargs="*", default=[], help="per-label thresholds, e.g. WIT=2 PREC=4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="Phase-2 data folder (needed for --snap with zip members)")
    args = ap.parse_args()
    assert len(args.runs) >= 2, "an ensemble needs at least 2 runs"
    lmv = {k: float(v) for k, v in (x.split("=", 1) for x in args.label_min_votes)}
    if args.mode == "vote":
        w = args.weights or [1.0] * len(args.runs)
        assert len(w) == len(args.runs), "--weights needs one value per member"
        assert args.min_votes and 0 < args.min_votes <= sum(w), "--min-votes must be in (0, sum of weights]"
    if all(r.endswith(".zip") for r in args.runs):
        assert args.mode == "vote", "zip members (test predictions only) support --mode vote only"
        _, test_pred = vote_ensemble(args.runs, args.min_votes, args.weights, lmv)
        if args.snap:
            test_texts = ner_data.load_test(args.data) if args.data else None
            assert test_texts, "--snap with zip members needs --data (the Phase-2 data folder) for the document text"
            test_pred = {d: ner_align.snap_to_words(test_texts[d], v) for d, v in test_pred.items()}
        out = os.path.join(args.out, args.name)
        os.makedirs(out, exist_ok=True)
        docs = set().union(*[read_pred(r) for r in args.runs])
        sub = {d: [list(x) for x in test_pred.get(d, [])] for d in sorted(docs)}
        pred_path = os.path.join(out, "prediction.json")
        json.dump(sub, open(pred_path, "w"))
        zip_path = os.path.join(out, f"submission_{args.name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pred_path, arcname="prediction.json")
        summary = dict(name=args.name, mode="vote", members=[os.path.basename(r) for r in args.runs],
                       weights=args.weights or [1.0] * len(args.runs), min_votes=args.min_votes, label_min_votes=lmv,
                       snap=args.snap, docs=len(sub),
                       test_spans=sum(len(v) for v in sub.values()),
                       member_spans={os.path.basename(r): sum(len(v) for v in read_pred(r).values()) for r in args.runs})
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
        print(json.dumps(summary, indent=2))
        print("zip:", zip_path)
        return
    cfgs = [load_cfg(r) for r in args.runs]
    check_compatible(cfgs)
    texts, gold, fold_of, labels = ner_data.load_train(cfgs[0]["data_dir"], verbose=False)
    test_texts = ner_data.load_test(cfgs[0]["data_dir"])
    _, val_ids = ner_data.split_docs(list(texts), fold_of, cfgs[0]["split"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"ensemble {args.name}: mode={args.mode} members={[c['name'] for c in cfgs]}", flush=True)
    if args.mode == "prob":
        val_pred, test_pred = prob_ensemble(args.runs, cfgs, texts, gold, val_ids, test_texts, labels, device)
    else:
        val_pred, test_pred = vote_ensemble(args.runs, args.min_votes, args.weights, lmv)
        # the members wrote their files in the gold-file end convention; scoring uses the same convention
        val_pred = {d: [tuple(s) for s in v] for d, v in val_pred.items()}
    if args.snap:
        val_pred = {d: ner_align.snap_to_words(texts[d], v) for d, v in val_pred.items()}
        test_pred = {d: ner_align.snap_to_words(test_texts[d], v) for d, v in test_pred.items()}
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    if args.mode == "prob":
        val_sub, test_sub = ner_data.to_submission(val_pred), ner_data.to_submission(test_pred)
    else:
        val_sub = {d: [list(s) for s in v] for d, v in val_pred.items()}
        test_sub = {d: [list(s) for s in v] for d, v in test_pred.items()}
    for d in test_texts:
        test_sub.setdefault(d, [])
    gold_sub = {d: [tuple(x) for x in ner_data.to_submission({d: gold[d]})[d]] for d in val_ids}
    sc = strict_scores(gold_sub, {d: [tuple(x) for x in v] for d, v in val_sub.items()}, labels)
    json.dump(sc, open(os.path.join(out, "val_scores.json"), "w"), indent=2)
    json.dump(val_sub, open(os.path.join(out, "val_predictions.json"), "w"))
    pred_path = os.path.join(out, "prediction.json")
    json.dump(test_sub, open(pred_path, "w"))
    zip_path = os.path.join(out, f"submission_{args.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    member_f1 = {}
    for r, c in zip(args.runs, cfgs):
        sp = os.path.join(r, "summary.json")
        member_f1[c["name"]] = json.load(open(sp)).get("val_macro_f1") if os.path.exists(sp) else None
    summary = dict(name=args.name, mode=args.mode, min_votes=args.min_votes, snap=args.snap, members=member_f1,
                   val_macro_f1=sc["macro_f1"], val_macro_p=sc["macro_p"], val_macro_r=sc["macro_r"],
                   val_micro_f1=sc["micro_f1"], test_spans=sum(len(v) for v in test_sub.values()),
                   seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("zip:", zip_path)


if __name__ == "__main__":
    main()
