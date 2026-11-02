#!/usr/bin/env python3
"""Phase-6 alignment facts (plumbing; declare it): split each document into equal word chunks, assign every summary
sentence to the chunk with the highest unigram+bigram overlap, and report how ordered / spread the assignment is,
plus novel n-gram rates (abstractiveness).  python analyze_align.py [--chunk-words 600] [--n 1500]"""
import argparse, json, os, random, re
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SENT = re.compile(r"(?<=[.?!])\s+(?=[A-Z(\"'])")
WORD = re.compile(r"[a-z0-9]+")

def grams(ws, n): return set(zip(*[ws[i:] for i in range(n)]))

def kendall(xs):
    c = d = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] != xs[j]:
                c += xs[i] < xs[j]; d += xs[i] > xs[j]
    return (c - d) / (c + d) if c + d else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, default=600)
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    data = [x for x in json.load(open(os.path.join(HERE, "data/Phase_6_Summ/Train/train.json"))) if x["summary"].strip()]
    random.Random(0).shuffle(data); data = data[:args.n]
    taus, empty_frac, rel_pos, novel1, novel2, novel4, share_first, cov_ratio = [], [], [], [], [], [], [], []
    for x in data:
        dws = WORD.findall(x["document"].lower())
        K = max(1, round(len(dws) / args.chunk_words))
        size = -(-len(dws) // K)
        chunks = [dws[i:i + size] for i in range(0, len(dws), size)]
        cg = [(set(c), grams(c, 2)) for c in chunks]
        sents = [WORD.findall(s.lower()) for s in SENT.split(x["summary"])]
        sents = [s for s in sents if len(s) >= 4]
        if not sents: continue
        assign, words_per_chunk = [], Counter()
        for s in sents:
            u, b = set(s), grams(s, 2)
            sc = [len(u & c1) + 2 * len(b & c2) for c1, c2 in cg]
            k = int(np.argmax(sc)); assign.append(k); words_per_chunk[k] += len(s)
        if K > 1:
            taus.append(kendall(assign)); empty_frac.append(1 - len(set(assign)) / K)
            rel_pos += [a / (K - 1) for a in assign]
            share_first.append(words_per_chunk[0] / sum(words_per_chunk.values()))
            sw = sum(map(len, sents))
            cov_ratio += [words_per_chunk[k] / sw / (len(chunks[k]) / len(dws)) for k in range(K)]
        sw_all = [w for s in sents for w in s]
        for n, out in ((1, novel1), (2, novel2), (4, novel4)):
            sg, dg = grams(sw_all, n), grams(dws, n)
            out.append(len(sg - dg) / max(1, len(sg)))
    q = lambda a: {k: round(float(np.percentile(a, p)), 3) for k, p in (("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90))} | {"mean": round(float(np.mean(a)), 3)}
    st = {"docs": len(data), "chunk_words": args.chunk_words,
          "kendall_tau_sentence_order_vs_chunk_order": q(taus),
          "frac_chunks_with_no_summary_sentence": q(empty_frac),
          "relative_position_of_aligned_chunk_hist10": np.histogram(rel_pos, bins=10, range=(0, 1))[0].tolist(),
          "summary_word_share_of_first_chunk": q(share_first),
          "per_chunk_summary_share_over_chunk_share": q(cov_ratio),
          "novel_unigram_rate": q(novel1), "novel_bigram_rate": q(novel2), "novel_4gram_rate": q(novel4)}
    out = os.path.join(HERE, "data_analysis", f"align_stats_cw{args.chunk_words}.json")
    json.dump(st, open(out, "w"), indent=1); print(json.dumps(st, indent=1))

if __name__ == "__main__":
    main()
