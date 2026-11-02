#!/usr/bin/env python3
"""Oracle check of the RR prediction path (plumbing; declare it). No model involved.

For several context settings: build the windows for every train/dev/test document (each sentence must be a target
exactly once, in order, windows within window_len), then feed the GOLD labels through train_rr.predict() (same
batching, collation, per-document assembly) and the prediction.json formatting. Dev macro-F1 must be 1.0 and every
prediction list must have exactly one label per sentence.

  python check_pipeline_rr.py --data data --tokenizer ../phase1/ab_inputs/legal_bpe.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_data                        # noqa: E402
import train_rr                       # noqa: E402
from a1_tokenizer import ByteBPE      # noqa: E402
from rr_metrics import rr_scores      # noqa: E402


class Oracle(nn.Module):
    """Returns one-hot logits of the gold labels, in the exact order train_rr.predict() asks for them."""

    def __init__(self, wins, micro, n_labels):
        super().__init__()
        self.queue = [l for idx in train_rr.batches(wins, micro, False, None) for i in idx for l in wins[i]["labels"]]
        self.n, self.pos, self.head = n_labels, 0, nn.Identity()

    def sentence_logits(self, ids, rows, starts, ends):
        k = len(rows)
        lab = torch.tensor(self.queue[self.pos:self.pos + k])
        self.pos += k
        return torch.nn.functional.one_hot(lab, self.n).float() * 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()
    tok = ByteBPE.from_json(args.tokenizer)
    labels = rr_data.load_labels(args.data)
    lab2id, id2lab = {l: i for i, l in enumerate(labels)}, dict(enumerate(labels))
    print(f"labels ({len(labels)}): {labels}")
    splits = {s: [d for dom in ("CL", "IT") for d in rr_data.load_domain(args.data, dom, s)] for s in ("train", "dev", "test")}
    for s, docs in splits.items():
        n = sum(len(d["sentences"]) for d in docs)
        print(f"{s}: {len(docs)} docs, {n} sentences" + (f", ids unique: {len({d['id'] for d in docs}) == len(docs)}"))
    settings = {
        "sentence-only": dict(window_len=128, max_sent_tokens=128, max_targets=1, target_tokens=128, left_tokens=0, right_tokens=0, sep_token=False),
        "packed, no context": dict(window_len=512, max_sent_tokens=256, max_targets=None, target_tokens=512, left_tokens=0, right_tokens=0, sep_token=True),
        "1 target + context": dict(window_len=512, max_sent_tokens=128, max_targets=1, target_tokens=128, left_tokens=192, right_tokens=192, sep_token=True),
        "packed + context": dict(window_len=768, max_sent_tokens=256, max_targets=None, target_tokens=384, left_tokens=192, right_tokens=192, sep_token=True),
    }
    for name, ctx in settings.items():
        wins = {s: train_rr.prepare(docs, tok, ctx, lab2id) for s, docs in splits.items()}   # includes check_windows
        dev = splits["dev"]
        pred = train_rr.predict(Oracle(wins["dev"], 8, len(labels)), wins["dev"], dev, 8, torch.device("cpu"), False, id2lab)
        pred = json.loads(json.dumps(pred))                                                  # through the JSON format
        sc = rr_scores({d["id"]: d["labels"] for d in dev}, pred, labels)
        ok = all(len(pred[d["id"]]) == len(d["sentences"]) for d in dev)
        trunc = sum(len(tok.encode(x)) > ctx["max_sent_tokens"] for d in splits["train"] for x in d["sentences"])
        print(f"{name:20s} windows train/dev/test {len(wins['train'])}/{len(wins['dev'])}/{len(wins['test'])} | "
              f"oracle dev macro-F1 {sc['macro_f1']} | counts match {ok} | train sentences truncated {trunc}")
        assert sc["macro_f1"] == 1.0 and ok


if __name__ == "__main__":
    main()
