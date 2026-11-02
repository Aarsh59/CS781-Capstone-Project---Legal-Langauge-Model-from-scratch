#!/usr/bin/env python3
"""Sentence-level voting ensemble of finished RR runs (plumbing; declare it). Members / weights are your choice.

  python rr_ensemble.py --name ENS --members runs/A runs/B runs/C [--weights 1 1 2] --out runs
  python rr_ensemble.py --name ENS --members a.zip b.zip c.zip --out ensembles      # test predictions only

For every sentence the label with the largest summed member weight wins; ties go to the label predicted by the
earliest-listed member among the tied ones. Run directories also give a dev score (from dev_predictions.json).
"""
import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_pred(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("prediction.json"))
    return json.load(open(path))


def vote(preds, weights):
    out = {}
    for d in preds[0]:
        n = len(preds[0][d])
        assert all(len(p[d]) == n for p in preds), f"doc {d}: members disagree on the number of sentences"
        labs = []
        for i in range(n):
            score, first = defaultdict(float), {}
            for m, (p, w) in enumerate(zip(preds, weights)):
                score[p[d][i]] += w
                first.setdefault(p[d][i], m)
            best = max(score.values())
            labs.append(min((l for l, s in score.items() if s == best), key=lambda l: first[l]))
        out[d] = labs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = args.weights or [1.0] * len(args.members)
    assert len(w) == len(args.members) >= 2
    zips = all(m.endswith(".zip") for m in args.members)
    test = vote([read_pred(m if zips else os.path.join(m, "prediction.json")) for m in args.members], w)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    summary = {"name": args.name, "members": args.members, "weights": w}
    if not zips:
        import rr_data
        from rr_metrics import rr_scores
        cfg = json.load(open(os.path.join(args.members[0], "config.json")))
        labels = json.load(open(os.path.join(args.members[0], "labels.json")))
        dev = vote([json.load(open(os.path.join(m, "dev_predictions.json"))) for m in args.members], w)
        docs = [d for dom in ("CL", "IT") for d in rr_data.load_domain(cfg["data_dir"], dom, "dev")]
        if len(dev) < len(docs):                      # smoke runs predict only a few dev docs
            print(f"note: members predicted {len(dev)}/{len(docs)} dev docs; dev score over those only")
            docs = [d for d in docs if d["id"] in dev]
        sc = rr_scores({d["id"]: d["labels"] for d in docs}, dev, labels)
        json.dump(sc, open(os.path.join(out, "dev_scores.json"), "w"), indent=2)
        json.dump(dev, open(os.path.join(out, "dev_predictions.json"), "w"))
        summary.update(dev_macro_f1=sc["macro_f1"], dev_macro_p=sc["macro_p"], dev_macro_r=sc["macro_r"])
    pj = os.path.join(out, "prediction.json")
    json.dump(test, open(pj, "w"))
    zp = os.path.join(out, f"submission_{args.name}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2)); print("zip:", zp)


if __name__ == "__main__":
    main()
