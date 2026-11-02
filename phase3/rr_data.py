"""Phase 3 (Rhetorical Roles) data: documents, labels, sentence encoding, context windows (plumbing; declare it).

A document is an ordered list of sentences (+ one label per sentence for train/dev). The model predicts exactly one
label per sentence. Sentences are encoded separately with your ByteBPE and then packed into windows:

  window = [left-context sentences] [TARGET sentences] [right-context sentences]      (<= window_len tokens)

Only TARGET sentences are classified in a window; every sentence of a document is the target of exactly one window
(checked by check_windows()), so predictions come out complete and in the original order.

context config (cfg["context"], your choices):
  window_len        max tokens per window (<= backbone max_seq_len)
  max_sent_tokens   a sentence longer than this is truncated to its first max_sent_tokens tokens
  max_targets       max target sentences per window (1 = classic sentence-level classification)
  target_tokens     max tokens of the target group (>= 1 sentence is always taken)
  left_tokens       left-context budget in tokens (0 = none)
  right_tokens      right-context budget in tokens (0 = none)
  sep_token         True: put <sep> between sentences (the pooled span excludes it)
"""
import json
import os

import numpy as np

PAD_ID, UNK_ID, SEP_ID = 0, 1, 5          # a1_tokenizer SPECIALS: <pad>=0, <unk>=1, <sep>=5


def data_root(root):
    for cand in (root, os.path.join(root, "Phase_3_RR")):
        if os.path.isdir(os.path.join(cand, "Train")):
            return cand
    raise FileNotFoundError(f"no Phase_3_RR/Train under {root} (run fetch_data.py first)")


def load_labels(root):
    v = json.load(open(os.path.join(data_root(root), "Train", "label_vocab.json"), encoding="utf-8"))
    if isinstance(v, dict):                          # {label: index} or {"labels": [...]}
        if "labels" in v and isinstance(v["labels"], list):
            return list(v["labels"])
        return [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
    return list(v)


def load_docs(path):
    """-> list of {"id", "domain"?, "sentences": [...], "labels": [...] | None} in file order."""
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):                        # {doc_id: {...}} or {doc_id: [sentences]}
        items = []
        for k, v in data.items():
            if isinstance(v, dict):
                v = dict(v, id=v.get("id", k))
            else:
                v = {"id": k, "text": v}
            items.append(v)
        data = items
    docs = []
    for d in data:
        sents = d.get("text", d.get("sentences"))
        labels = d.get("labels", d.get("label"))
        assert isinstance(sents, list), f"document {d.get('id')}: 'text' is not a list of sentences"
        if labels is not None:
            assert len(labels) == len(sents), f"document {d.get('id')}: {len(labels)} labels for {len(sents)} sentences"
        docs.append({"id": str(d.get("id")), "sentences": [str(s) for s in sents], "labels": labels})
    return docs


def load_domain(root, domain, split):
    sub = "Test" if split == "test" else "Train"
    docs = load_docs(os.path.join(data_root(root), sub, f"{domain}_{split}.json"))
    for d in docs:
        d["domain"] = domain
    return docs


# ---------------------------------------------------------------------------
# encoding + windows
# ---------------------------------------------------------------------------

def encode_doc(tok, doc, max_sent_tokens):
    """Token ids per sentence; an empty sentence becomes [<unk>] so it still gets a representation."""
    out = []
    for s in doc["sentences"]:
        ids = tok.encode(s)[:max_sent_tokens]
        out.append(ids if ids else [UNK_ID])
    return out


def build_windows(sent_ids, ctx):
    """-> list of windows: {"ids": [...], "targets": [(sent_index, start, end)], } with end exclusive."""
    n = len(sent_ids)
    sep = [SEP_ID] if ctx["sep_token"] else []
    lens = [len(x) + len(sep) for x in sent_ids]
    windows, i = [], 0
    while i < n:
        # target group: >= 1 sentence, within target_tokens / max_targets / window_len
        j, tlen = i, 0
        while j < n and (j == i or (tlen + lens[j] <= ctx["target_tokens"]
                                    and (ctx["max_targets"] is None or j - i < ctx["max_targets"])
                                    and tlen + lens[j] <= ctx["window_len"])):
            tlen += lens[j]
            j += 1
        budget = ctx["window_len"] - tlen
        # left context (closest sentences first), then right context, within their budgets and the window
        left, ltoks, k = [], 0, i - 1
        while k >= 0 and ltoks + lens[k] <= min(ctx["left_tokens"], budget):
            left.insert(0, k)
            ltoks += lens[k]
            k -= 1
        budget -= ltoks
        right, rtoks, k = [], 0, j
        while k < n and rtoks + lens[k] <= min(ctx["right_tokens"], budget):
            right.append(k)
            rtoks += lens[k]
            k += 1
        ids, targets = [], []
        for si in left + list(range(i, j)) + right:
            start = len(ids)
            ids.extend(sent_ids[si])
            if i <= si < j:
                targets.append((si, start, start + len(sent_ids[si])))
            ids.extend(sep)
        windows.append({"ids": ids, "targets": targets})
        i = j
    return windows


def check_windows(n_sentences, windows, window_len):
    seen = [t[0] for w in windows for t in w["targets"]]
    assert seen == list(range(n_sentences)), "every sentence must be a target exactly once, in order"
    assert all(len(w["ids"]) <= window_len for w in windows), "window longer than window_len"
    assert all(e > s for w in windows for _, s, e in w["targets"]), "empty target span"


def label_counts(docs, labels):
    c = {l: 0 for l in labels}
    for d in docs:
        for l in d["labels"] or []:
            c[l] = c.get(l, 0) + 1
    return c
