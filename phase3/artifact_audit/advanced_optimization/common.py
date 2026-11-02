"""Shared loaders for the advanced offline analysis (plumbing; declare it). Reads ../bundle and ../experiment_inventory.csv only."""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
BD = os.path.join(AUD, "bundle")
sys.path.insert(0, os.path.join(BD, "evaluation"))
from rr_ensemble import vote                     # noqa: E402,F401
from rr_metrics import rr_scores                 # noqa: E402

ANCHOR = "B_lora_ctx_wide_allep"
FIXED = {"H": "H_anchor_w2.0+top4net", "G2": "G2_anchor+B_lorma_allep+KD_seed123_allep", "C1": "C1_one_rep_per_family_bestdev"}
labels = json.load(open(os.path.join(BD, "label_vocab.json")))["labels"]
L, li = len(labels), {l: i for i, l in enumerate(labels)}

gold, dom = defaultdict(list), {}
for r in csv.DictReader(open(os.path.join(BD, "dev_gold.csv"))):
    gold[r["doc_id"]].append(r["gold_label"]); dom[r["doc_id"]] = r["domain"]
gold = dict(gold)
docs = list(gold)
HP = defaultdict(lambda: defaultdict(list))
_rd = csv.DictReader(open(os.path.join(BD, "hard_predictions_dev.csv")))
names = [c for c in _rd.fieldnames if c not in ("split", "doc_id", "sentence_index")]
for r in _rd:
    for n in names:
        HP[n][r["doc_id"]].append(r[n])
inv = {r["experiment_name"]: r for r in csv.DictReader(open(os.path.join(AUD, "experiment_inventory.csv")))}
runs = [n for n in names if n in inv and not inv[n]["experiment_id"].startswith("ensemble:")]
PR, DEC = {}, {}
for n in runs:
    f = os.path.join(BD, "probs", f"{n}_dev.csv")
    if os.path.exists(f):
        d = defaultdict(list)
        for r in csv.DictReader(open(f)):
            d[r["doc_id"]].append([float(r[f"p_{l}"]) for l in labels]); DEC[n] = r["decode"]
        PR[n] = {k: np.array(v) for k, v in d.items()}
LOSS = {}
for n in runs:
    cp = inv[n]["config_path"]
    c = json.load(open(cp)) if os.path.exists(cp) else {}
    LOSS[n] = "CRF" if c.get("head", {}).get("name") == "mlp_crf" else "KD" if c.get("kd") else str(c.get("train", {}).get("class_weights"))
GROUP = [n for n in PR if DEC[n] == "argmax" and LOSS[n] == "inv_sqrt"]       # compatible probability group


def score(P, ds, convention="gold_or_pred"):
    """macro-F1 under a label-set convention: gold_or_pred (local scorer), gold_only, all13 (zero F1 for absent labels)."""
    g = {k: gold[k] for k in ds}; p = {k: P[k] for k in ds}
    s = rr_scores(g, p, labels)
    if convention == "gold_or_pred":
        return s
    per = s["per_label"]
    if convention == "gold_only":
        keep = [l for l in labels if any(l in v for v in g.values())]
    else:
        keep = labels
    f1 = [per[l]["f1"] if l in per else 0.0 for l in keep]
    return dict(s, macro_f1=sum(f1) / len(keep))


def mf1(P, ds, convention="gold_or_pred"):
    return score(P, ds, convention)["macro_f1"]


def cl_it(P):
    return {D: mf1(P, [k for k in docs if dom[k] == D]) for D in ("CL", "IT")}
