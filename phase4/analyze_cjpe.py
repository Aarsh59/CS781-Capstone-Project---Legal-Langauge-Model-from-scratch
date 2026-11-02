#!/usr/bin/env python3
"""Descriptive statistics of Phase_4_CJPE (plumbing; declare it). Facts only: no modelling choices.

  python analyze_cjpe.py                 # reads ./data (fetch_data.py), writes ./data_analysis/cjpe_stats.json

Per split: document count, label balance, words and tokens per document (your legal_bpe.json ByteBPE), share that fits
in 896 / 1024 tokens, 1024-token chunks per document, exact duplicates within / across splits, and an estimated court
source per document from a unigram multinomial naive-Bayes classifier fitted on Phase-1 corpus samples (its held-out
accuracy on those samples is reported, so you can judge how far to trust it), plus court-name mentions in the opening text.
"""
import collections
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "data_analysis")
sys.path.insert(0, os.path.join(HERE, "..", "phase1", "ctxext", "p1"))
import a1_tokenizer  # noqa: E402

TOK = os.path.join(HERE, "..", "phase1", "ab_inputs", "legal_bpe.json")
SOURCES = ("supreme_court", "acts", "high_courts", "tribunals")
WORD = re.compile(r"[a-z]+")


def load_split(name):
    rel = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}[name]
    data = json.load(open(os.path.join(DATA, "Phase_4_CJPE", rel), encoding="utf-8"))
    if isinstance(data, dict):                                  # {id: record} layout
        data = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in data.items()]
    return data


def pct(a, qs=(50, 90, 95, 99)):
    return {f"p{q}": int(np.percentile(a, q)) for q in qs}


def words(t):
    return WORD.findall(t.lower())


# ------------------------------------------------------------------ unigram source classifier
def fit_nb(samples, vocab_size=50_000, alpha=1.0):
    cnt = {s: collections.Counter() for s in samples}
    for s, texts in samples.items():
        for t in texts:
            cnt[s].update(words(t))
    total = collections.Counter()
    for c in cnt.values():
        total.update(c)
    vocab = {w: i for i, (w, _) in enumerate(total.most_common(vocab_size))}
    logp = np.zeros((len(samples), len(vocab)))
    for k, s in enumerate(samples):
        v = np.full(len(vocab), alpha)
        for w, n in cnt[s].items():
            if w in vocab:
                v[vocab[w]] += n
        logp[k] = np.log(v / v.sum())
    return list(samples), vocab, logp


def nb_predict(model, text):
    names, vocab, logp = model
    idx = collections.Counter(vocab[w] for w in words(text) if w in vocab)
    if not idx:
        return None, None
    ids, n = np.array(list(idx)), np.array(list(idx.values()), float)
    ll = logp[:, ids] @ n                                       # equal priors: the corpus mix is not CJPE's mix
    ll -= ll.max()
    p = np.exp(ll) / np.exp(ll).sum()
    return names[int(p.argmax())], float(p.max())


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = a1_tokenizer.ByteBPE.from_json(TOK)
    splits = {s: load_split(s) for s in ("train", "dev", "test")}
    rep = {"fields": {s: sorted({k for r in d[:200] for k in r}) for s, d in splits.items()}}

    # classifier on reference samples: 80/20 split by record for the accuracy estimate, then refit on everything
    ref = {}
    for s in SOURCES:
        p = os.path.join(DATA, "ref_samples", f"{s}.jsonl")
        ref[s] = [json.loads(l)["text"] for l in open(p, encoding="utf-8") if len(l) > 200]
    hold = {s: t[::5] for s, t in ref.items()}
    fit = {s: [x for i, x in enumerate(t) if i % 5] for s, t in ref.items()}
    m = fit_nb(fit)
    conf = {s: collections.Counter(nb_predict(m, t)[0] for t in hold[s]) for s in SOURCES}
    rep["source_classifier"] = {
        "reference_records": {s: len(t) for s, t in ref.items()},
        "heldout_accuracy": {s: round(conf[s][s] / max(1, len(hold[s])), 4) for s in SOURCES},
        "heldout_confusion": {s: dict(conf[s]) for s in SOURCES},
        "note": "multinomial naive Bayes on lowercase word unigrams, equal priors; fitted on Phase-1 corpus samples",
    }
    m = fit_nb(ref)

    court_pat = {"supreme_court": r"supreme court", "high_court": r"high court", "tribunal": r"tribunal|appellate authority"}
    hashes = {}
    for name, docs in splits.items():
        texts = [str(r.get("text", "")) for r in docs]
        nw = np.array([len(t.split()) for t in texts])
        ntok = np.array([len(tok.encode(t)) for t in texts])
        labels = collections.Counter(r.get("label") for r in docs if "label" in r)
        src = [nb_predict(m, t) for t in texts]
        src_c = collections.Counter(s for s, _ in src)
        conf_hi = collections.Counter(s for s, p in src if p is not None and p >= 0.99)
        head = [t[:1500].lower() for t in texts]
        mentions = {k: int(sum(bool(re.search(v, h)) for h in head)) for k, v in court_pat.items()}
        hs = [hashlib.md5(" ".join(t.split()).encode()).hexdigest() for t in texts]
        hashes[name] = set(hs)
        rep[name] = {
            "docs": len(docs),
            "labels": {str(k): v for k, v in sorted(labels.items(), key=lambda x: str(x[0]))},
            "label_share": {str(k): round(v / max(1, sum(labels.values())), 4) for k, v in labels.items()},
            "words": {"mean": round(float(nw.mean()), 1), "min": int(nw.min()), **pct(nw), "max": int(nw.max())},
            "tokens": {"mean": round(float(ntok.mean()), 1), "min": int(ntok.min()), **pct(ntok), "max": int(ntok.max()),
                       "total": int(ntok.sum())},
            "tokens_per_word": round(float(ntok.sum() / max(1, nw.sum())), 4),
            "share_fitting": {"<=896": round(float((ntok <= 896).mean()), 4), "<=1024": round(float((ntok <= 1024).mean()), 4)},
            "chunks_1024": {"mean": round(float(np.ceil(ntok / 1024).mean()), 2), **pct(np.ceil(ntok / 1024).astype(int))},
            "estimated_source": {k: v for k, v in src_c.most_common()},
            "estimated_source_share": {k: round(v / len(docs), 4) for k, v in src_c.most_common()},
            "estimated_source_confident_p99": dict(conf_hi.most_common()),
            "court_mentions_first_1500_chars": mentions,
            "exact_duplicates_within": len(hs) - len(set(hs)),
        }
        if labels:
            by_lab = collections.defaultdict(list)
            for r, n in zip(docs, ntok):
                by_lab[str(r.get("label"))].append(n)
            rep[name]["tokens_by_label_median"] = {k: int(np.median(v)) for k, v in by_lab.items()}
        print(f"{name}: {len(docs)} docs done", flush=True)
    rep["exact_overlap"] = {"train&dev": len(hashes["train"] & hashes["dev"]), "train&test": len(hashes["train"] & hashes["test"]),
                            "dev&test": len(hashes["dev"] & hashes["test"])}
    rep["examples"] = {s: [{"id": r.get("id"), "label": r.get("label"), "start": str(r.get("text", ""))[:300],
                            "end": str(r.get("text", ""))[-300:]} for r in splits[s][:2]] for s in ("train", "test")}
    json.dump(rep, open(os.path.join(OUT, "cjpe_stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
