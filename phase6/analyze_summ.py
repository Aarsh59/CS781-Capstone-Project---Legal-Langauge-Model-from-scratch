#!/usr/bin/env python3
"""Phase-6 data statistics (plumbing; declare it): word / ByteBPE-token lengths of documents and summaries,
compression ratio, duplicates, train/test overlap. Facts only.  python analyze_summ.py [--sample N]"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase5"))
from a1_tokenizer import ByteBPE  # noqa: E402

def pct(a):
    a = np.asarray(a)
    return {k: int(np.percentile(a, q)) for k, q in
            [("min", 0), ("p5", 5), ("p25", 25), ("p50", 50), ("mean", None), ("p75", 75), ("p95", 95), ("max", 100)]
            if q is not None} | {"mean": round(float(a.mean()), 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "Phase_6_Summ"))
    ap.add_argument("--tok", default=os.path.join(HERE, "..", "phase1", "ab_inputs", "legal_bpe.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "data_analysis", "summ_stats.json"))
    args = ap.parse_args()
    tok = ByteBPE.from_json(args.tok)
    train = json.load(open(os.path.join(args.data, "Train", "train.json")))
    test = json.load(open(os.path.join(args.data, "Test", "test.json")))
    st = {"n_train": len(train), "n_test": len(test)}
    h = lambda s: hashlib.md5(" ".join(s.split()).encode()).hexdigest()
    tr_doc = [h(x["document"]) for x in train]
    st["train_dup_docs"] = len(tr_doc) - len(set(tr_doc))
    st["train_ids_unique"] = len({x["id"] for x in train}) == len(train)
    st["test_docs_in_train"] = sum(h(x["document"]) in set(tr_doc) for x in test)
    st["test_ids_in_train"] = len({x["id"] for x in test} & {x["id"] for x in train})
    st["empty_train_summaries"] = sum(not x["summary"].strip() for x in train)
    if True:
        enc = lambda texts: [len(ids) for ids in tok.encode_many(texts, workers=4)]
        dw = [len(x["document"].split()) for x in train]; sw = [len(x["summary"].split()) for x in train]
        dt = enc([x["document"] for x in train]); s_t = enc([x["summary"] for x in train])
        tw = [len(x["document"].split()) for x in test]; tt = enc([x["document"] for x in test])
    st["train_doc_words"], st["train_doc_tokens"] = pct(dw), pct(dt)
    st["train_sum_words"], st["train_sum_tokens"] = pct(sw), pct(s_t)
    st["test_doc_words"], st["test_doc_tokens"] = pct(tw), pct(tt)
    st["sum_over_doc_words_ratio"] = pct([100 * s / max(d, 1) for s, d in zip(sw, dw)])  # percent
    st["corr_doc_vs_sum_words"] = round(float(np.corrcoef(dw, sw)[0, 1]), 3)
    st["tok_per_word_doc"] = round(sum(dt) / sum(dw), 3); st["tok_per_word_sum"] = round(sum(s_t) / sum(sw), 3)
    for L in (256, 384, 512, 640):
        st[f"frac_sum_tokens_le_{L}"] = round(float(np.mean(np.array(s_t) <= L)), 3)
    for L in (896, 1024, 2048, 4096):
        st[f"frac_doc_tokens_le_{L}"] = round(float(np.mean(np.array(dt) <= L)), 3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(st, open(args.out, "w"), indent=1)
    print(json.dumps(st, indent=1))

if __name__ == "__main__":
    main()
