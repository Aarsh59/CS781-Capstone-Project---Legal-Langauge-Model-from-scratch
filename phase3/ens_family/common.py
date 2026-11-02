"""Shared loading / scoring for the Phase-3 family-ensemble study (plumbing; declare it).

Reads only saved artifacts of the 27 _allep runs (dev_predictions.json, prediction.json, dev_probs.json,
test_probs.json, summary.json, config.json); never writes outside ens_family/. Metric = rr_metrics.rr_scores
(the pipeline's macro-F1: labels present in gold or predictions), label order = Train/label_vocab.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rr_data                                   # noqa: E402
from rr_ensemble import vote                     # noqa: E402,F401  (weighted hard vote, ties -> earliest member)
from rr_metrics import rr_scores                 # noqa: E402

RUNS_DIR = os.path.expanduser("~/kaggle_outputs/allep/phase3/runs")
DATA = os.path.join(os.path.dirname(HERE), "data")
ANCHOR = "B_lora_ctx_wide_allep"


class Fail(SystemExit):
    pass


def load():
    labels = rr_data.load_labels(DATA)
    dev = {D: rr_data.load_domain(DATA, D, "dev") for D in ("CL", "IT")}
    tst = {D: rr_data.load_domain(DATA, D, "test") for D in ("CL", "IT")}
    gold = {d["id"]: d["labels"] for D in dev for d in dev[D]}
    domain = {d["id"]: D for D in dev for d in dev[D]}
    tdomain = {d["id"]: D for D in tst for d in tst[D]}
    nsent = {k: len(v) for k, v in gold.items()}
    tnsent = {d["id"]: len(d["sentences"]) for D in tst for d in tst[D]}
    runs = sorted(d for d in os.listdir(RUNS_DIR) if d.endswith("_allep") and os.path.isdir(os.path.join(RUNS_DIR, d)))
    M = {}
    for n in runs:
        rd = os.path.join(RUNS_DIR, n)
        r = {"dev": json.load(open(os.path.join(rd, "dev_predictions.json"))),
             "test": json.load(open(os.path.join(rd, "prediction.json"))),
             "summary": json.load(open(os.path.join(rd, "summary.json"))),
             "config": json.load(open(os.path.join(rd, "config.json")))}
        if json.load(open(os.path.join(rd, "labels.json"))) != labels:
            raise Fail(f"{n}: labels.json order differs from label_vocab")
        for split, ref in (("dev", nsent), ("test", tnsent)):
            P = r[split]
            if set(P) != set(ref) or any(len(P[k]) != ref[k] for k in ref):
                raise Fail(f"{n}: {split} document ids / sentence counts differ from the data")
            if not {l for v in P.values() for l in v} <= set(labels):
                raise Fail(f"{n}: {split} has a label outside label_vocab")
        for split, fn, ref in (("dev_probs", "dev_probs.json", nsent), ("test_probs", "test_probs.json", tnsent)):
            pr = json.load(open(os.path.join(rd, fn)))
            if pr["labels"] != labels or set(pr["probs"]) != set(ref) or any(len(pr["probs"][k]) != ref[k] for k in ref):
                raise Fail(f"{n}: {fn} misaligned (labels / ids / counts)")
            r[split], r[split + "_decode"] = pr["probs"], pr["decode"]
        c = r["config"]
        r["backbone"] = os.path.basename(os.path.dirname(c["phase1_ckpt"]))
        r["technique"] = c["technique"]["name"]
        r["kd"] = bool(c.get("kd"))
        r["head"] = c["head"]["name"]
        M[n] = r
    return dict(labels=labels, dev=dev, tst=tst, gold=gold, domain=domain, tdomain=tdomain, nsent=nsent,
                tnsent=tnsent, runs=runs, M=M)


def scores(ctx, P):
    """overall / CL / IT macro-F1 + macro P/R + per-label for one complete dev prediction set."""
    g, labels = ctx["gold"], ctx["labels"]
    s = rr_scores(g, P, labels)
    out = {"dev": s["macro_f1"], "P": s["macro_p"], "R": s["macro_r"], "acc": s["micro_f1"], "per": s["per_label"]}
    for D in ("CL", "IT"):
        gD = {k: v for k, v in g.items() if ctx["domain"][k] == D}
        out[D] = rr_scores(gD, {k: P[k] for k in gD}, labels)["macro_f1"]
    return out


def flat(ctx, P, split="dev"):
    order = list(ctx["gold"]) if split == "dev" else [d["id"] for D in ctx["tst"] for d in ctx["tst"][D]]
    return [l for k in order for l in P[k]]
