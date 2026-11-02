"""Phase-6 data plumbing (declare it): load / clean / split, sentence-level ByteBPE encoding (cached), head+tail
source selection, sentence-packed chunks (hierarchical stage 1), target truncation and per-kind batch collation.

Every source is exactly src_len tokens (short ones are LEFT-padded with <pad>), so a batch never mixes source lengths:
D's bidirectional prefix boundary and the generation prefill are identical for every row. Only documents shorter than
src_len are padded (about 2% of train at 896 tokens, fewer at smaller src_len).

Sequence formats (S = src_len, Tt = tgt_len, the target includes the end token when it is appended):
  decoder / prefix (A, B, D): [src S][sep] [tgt ... </s>] [pad ...]      labels = next token on the target part only
                              D: bidirectional over [src + sep] (prefix_len = S + 1), causal on the target
  encdec (C):                 encoder: [src S];  decoder in: [start] + tgt;  labels: tgt (+ </s>)
"""
import hashlib
import json
import os
import pickle
import random
import re

import numpy as np
import torch

SENT = re.compile(r"(?<=[.?!])\s+(?=[A-Z(\"'\[])")


def norm_hash(text):
    return hashlib.md5(" ".join(text.split()).encode()).hexdigest()


def load_json(path):
    d = json.load(open(path))
    if isinstance(d, dict):
        d = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "document": v} for k, v in d.items()]
    return d


def load_splits(data_dir, val_docs, seed):
    """-> train, val, test (lists of {"id", "document", ["summary"]}), report. val = seeded hold-out from train.json."""
    train = load_json(os.path.join(data_dir, "Train", "train.json"))
    test = load_json(os.path.join(data_dir, "Test", "test.json"))
    rep = {"train_raw": len(train), "test": len(test)}
    train = [d for d in train if d.get("summary", "").strip() and d.get("document", "").strip()]
    rep["dropped_empty"] = rep["train_raw"] - len(train)
    seen, uniq = set(), []
    for d in train:
        h = norm_hash(d["document"])
        if h not in seen:
            seen.add(h); uniq.append(d)
    rep["dropped_duplicate_docs"] = len(train) - len(uniq)
    test_h = {norm_hash(d["document"]) for d in test}
    train = [d for d in uniq if norm_hash(d["document"]) not in test_h]
    rep["dropped_overlapping_test"] = len(uniq) - len(train)
    order = sorted(train, key=lambda d: d["id"])
    random.Random(seed).shuffle(order)
    val, train = order[:val_docs], order[val_docs:]
    rep.update(train=len(train), val=len(val))
    return train, val, test, rep


def split_sentences(text):
    text = " ".join(text.split())
    return [s for s in SENT.split(text) if s]


def encode_sentences(tok, docs, field, cache_path=None, workers=None):
    """-> {id: [ids of sentence 0, ids of sentence 1, ...]} for docs[field]; cached (pickle) by ids + text hash."""
    key = hashlib.md5("".join(d["id"] + norm_hash(d[field]) for d in docs).encode()).hexdigest()
    if cache_path and os.path.exists(cache_path):
        c = pickle.load(open(cache_path, "rb"))
        if c.get("key") == key:
            return c["enc"]
    sents = [split_sentences(d[field]) for d in docs]
    flat = [s for ss in sents for s in ss]
    ids = list(tok.encode_many(flat, workers=workers)) if len(flat) > 2000 else [tok.encode(s) for s in flat]
    enc, k = {}, 0
    for d, ss in zip(docs, sents):
        enc[d["id"]] = ids[k:k + len(ss)]
        k += len(ss)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pickle.dump({"key": key, "enc": enc}, open(cache_path + ".tmp", "wb"))
        os.replace(cache_path + ".tmp", cache_path)
    return enc


# ---------------------------------------------------------------------------
# source selection / chunking / targets
# ---------------------------------------------------------------------------

def select_headtail(sents, S, head_frac):
    """Whole sentences from the start up to head_frac * S tokens, then whole sentences from the end up to the rest of
    the budget; leftover budget is filled by continuing the head (partial sentence). Document order is kept.
    A document that fits is returned whole. -> list of <= S ids."""
    flat_len = sum(map(len, sents))
    if flat_len <= S:
        return [t for s in sents for t in s]
    head_budget = int(round(head_frac * S))
    head, used, i = [], 0, 0
    while i < len(sents) and used + len(sents[i]) <= head_budget:
        head.append(sents[i]); used += len(sents[i]); i += 1
    tail, j = [], len(sents) - 1
    while j >= i and used + len(sents[j]) <= S:
        tail.append(sents[j]); used += len(sents[j]); j -= 1
    head_ids = [t for s in head for t in s]
    gap = S - used
    if gap > 0 and i <= j:                       # partial: continue the head into the unused middle
        mid = [t for s in sents[i:j + 1] for t in s]
        head_ids += mid[:gap]
    out = head_ids + [t for s in reversed(tail) for t in s]
    assert len(out) <= S
    return out


def pack_chunks(sents, S):
    """Greedy sentence packing into chunks of <= S tokens (a sentence longer than S is split). -> list of id lists."""
    chunks, cur = [], []
    for s in sents:
        while len(s) > S:
            if cur:
                chunks.append(cur); cur = []
            chunks.append(s[:S]); s = s[S:]
        if len(cur) + len(s) > S:
            chunks.append(cur); cur = []
        cur = cur + s
    if cur:
        chunks.append(cur)
    return chunks


def truncate_target(sents, budget, mode="sentence"):
    """-> (ids, truncated?) with len(ids) <= budget. mode "sentence": whole sentences only (the first sentence is cut
    if it alone is too long); "token": plain token cut."""
    flat = [t for s in sents for t in s]
    if len(flat) <= budget:
        return flat, False
    if mode == "token":
        return flat[:budget], True
    out = []
    for s in sents:
        if len(out) + len(s) > budget:
            break
        out += s
    return (out or flat[:budget]), True


def pad_left(ids, S, pad_id=0):
    return [pad_id] * (S - len(ids)) + list(ids)


# ---------------------------------------------------------------------------
# examples + collation
# ---------------------------------------------------------------------------

def build_examples(docs, src_enc, tgt_enc, fmt, sel, eos_id):
    """docs: list of {"id"}; src_enc / tgt_enc: {id: [sentence ids]} -> list of {"id", "src" (S ids), "tgt"}.
    fmt: src_len, tgt_len, eos_on_truncated, target_trunc; sel: strategy ("headtail"), head_frac."""
    S, Tt = fmt["src_len"], fmt["tgt_len"]
    out, n_trunc, n_short = [], 0, 0
    for d in docs:
        src = select_headtail(src_enc[d["id"]], S, sel["head_frac"])
        n_short += len(src) < S
        ex = {"id": d["id"], "src": pad_left(src, S)}
        if tgt_enc is not None:
            # room for </s> is always reserved; a truncated target gets </s> only if eos_on_truncated
            tgt, trunc = truncate_target(tgt_enc[d["id"]], Tt - 1, fmt.get("target_trunc", "sentence"))
            if (not trunc) or fmt["eos_on_truncated"]:
                tgt = tgt + [eos_id]
            n_trunc += trunc
            ex["tgt"] = tgt
        out.append(ex)
    stats = {"examples": len(out), "padded_sources": n_short}
    if tgt_enc is not None:
        stats.update(truncated_targets=n_trunc,
                     mean_tgt_tokens=round(float(np.mean([len(e["tgt"]) for e in out])), 1))
    return out, stats


def collate(examples, kind, sep_id=None, start_id=None, device="cpu"):
    """-> batch dict for GenModel.target_logits."""
    B = len(examples)
    S = len(examples[0]["src"])
    assert all(len(e["src"]) == S for e in examples)
    if kind == "encdec":
        L = max(len(e["tgt"]) for e in examples)
        dec_in = torch.zeros(B, L, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        for r, e in enumerate(examples):
            first = e["src"][-1] if start_id is None else start_id
            t = e["tgt"]
            dec_in[r, :len(t)] = torch.tensor([first] + t[:-1])
            labels[r, :len(t)] = torch.tensor(t)
        return {"src": torch.tensor([e["src"] for e in examples]).to(device), "dec_in": dec_in.to(device),
                "labels": labels.to(device)}
    L = S + 1 + max(len(e["tgt"]) for e in examples) - 1      # last target token is never an input
    ids = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for r, e in enumerate(examples):
        seq = e["src"] + [sep_id] + e["tgt"]
        ids[r, :len(seq) - 1] = torch.tensor(seq[:-1])
        labels[r, S:S + len(e["tgt"])] = torch.tensor(e["tgt"])   # position S (= sep) predicts tgt[0]
    b = {"ids": ids.to(device), "labels": labels.to(device)}
    if kind == "prefix":
        b["prefix_len"] = S + 1
    return b
