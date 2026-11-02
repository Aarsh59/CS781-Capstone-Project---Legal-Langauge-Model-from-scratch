#!/usr/bin/env python3
"""Oracle check of the whole prediction path (plumbing; declare it). No model involved.

Gold spans -> token tags -> one-hot "logits" per window -> the SAME window merging, decoding, offset conversion
and submission formatting as train_ner.py -> strict macro-F1 against gold. Any bug in windows/merge/offsets shows up
as a score below the alignment ceiling. Run it for the settings you intend to use:

  python check_pipeline.py --data data --tokenizer ../phase1/ab_inputs/legal_bpe.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_align                     # noqa: E402
import ner_data                      # noqa: E402
import train_ner                     # noqa: E402
from a1_tokenizer import ByteBPE     # noqa: E402
from ner_metrics import strict_scores  # noqa: E402


class OracleModel(nn.Module):
    """Returns one-hot logits of the gold tags for each window (looked up by the window's token ids)."""

    def __init__(self, enc, doc_ids, length, stride, n_tags):
        super().__init__()
        self.table = {}
        for d, s, n in train_ner.make_windows(enc, doc_ids, length, stride):
            self.table[enc[d]["ids"][s:s + n].tobytes()] = enc[d]["tags"][s:s + n]
        self.n_tags = n_tags
        self.head = nn.Identity()

    def forward(self, ids):
        rows = [self.table[r.astype(np.int32).tobytes()] for r in ids.cpu().numpy()]
        return torch.nn.functional.one_hot(torch.tensor(np.stack(rows)), self.n_tags).float() * 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    args = ap.parse_args()
    tok = ByteBPE.from_json(args.tokenizer)
    texts, gold, fold_of, labels = ner_data.load_train(args.data)
    docs = sorted(texts)
    for split_punct, scheme, merge, repair in itertools.product((False, True), ("BIO", "BIOES"),
                                                                 ("center", "left_context", "mean"), ("lenient",)):
        tags = ner_align.tag_list(labels, scheme)
        t2i, i2t = {t: i for i, t in enumerate(tags)}, dict(enumerate(tags))
        enc, _ = train_ner.encode_docs(texts, gold, ner_align.OffsetTokenizer(tok, split_punct), t2i, scheme)
        model = OracleModel(enc, docs, args.len, args.stride, len(tags))
        cfg = {"window": {"infer_len": args.len, "infer_stride": args.stride, "merge": merge},
               "train": {"micro_batch": 8}, "decode": {"repair": repair}}
        pred = train_ner.predict_spans(model, enc, docs, cfg, i2t, torch.device("cpu"), False)
        # through the submission format and back, exactly as Codabench reads it
        sub = json.loads(json.dumps(ner_data.to_submission(pred)))
        back = {d: [tuple(x[:3]) for x in v] for d, v in sub.items()}
        sc = strict_scores(gold, back, labels)
        print(f"split_punct={split_punct!s:<5} {scheme:<5} merge={merge:<12} macro-F1 {sc['macro_f1']:.4f}  "
              f"micro-F1 {sc['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
