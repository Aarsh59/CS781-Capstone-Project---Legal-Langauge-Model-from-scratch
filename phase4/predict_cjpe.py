#!/usr/bin/env python3
"""Inference-only re-chunking of a finished CJPE run (plumbing; declare it). No training.

  python predict_cjpe.py <run_dir> <out_dir> [variant ...]
  variant = strategy:max_chunks[:stride]   e.g. even:4  even:8  headtail:4  overlap:4:640  overlap:8:640

Loads <run_dir>/config.json + best_model.pt (the run's selected checkpoint), keeps its chunk length / pooling / head,
and only changes WHICH chunks of every dev / test document are read (attention pooling accepts any number of chunks).
Writes <out_dir>/<variant>/{dev_probs.json, test_probs.json, info.json}; dev / test ids as in the run's own outputs.
Test labels are never read.
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cjpe_data                                         # noqa: E402
import train_cjpe as T                                   # noqa: E402
from a1_tokenizer import ByteBPE                         # noqa: E402


def main():
    run_dir, out_dir, variants = sys.argv[1], sys.argv[2], sys.argv[3:] or ["even:4", "even:8", "headtail:4", "overlap:4:640"]
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    for k in ("data_dir", "tokenizer", "cache_dir"):            # paths of THIS session (env overrides)
        if os.environ.get(f"CJPE_{k.upper()}"):
            cfg[k] = os.environ[f"CJPE_{k.upper()}"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    tok = ByteBPE.from_json(cfg["tokenizer"])
    data, arrays, _ = T.load_data(cfg, tok)
    model, _ = T.load_run(run_dir, device, tok.get_vocab_size())
    L = T.chunk_len(cfg, model.backbone)
    im = cfg["train"].get("infer_micro_docs") or 4
    os.makedirs(out_dir, exist_ok=True)
    for v in variants:
        parts = v.split(":")
        strat, k = parts[0], int(parts[1])
        stride = int(parts[2]) if len(parts) > 2 else None
        t0 = time.time()
        res = {}
        for split in ("dev", "test"):
            items = cjpe_data.build_items(data[split], arrays[split], L, strat, k, stride)
            res[split] = T.probs_of(T.predict_logits(model, items, max(1, im // max(1, k // 4)), device, amp))
            res[split + "_coverage"] = round(cjpe_data.coverage(items), 4)
        d = os.path.join(out_dir, v.replace(":", "_"))
        os.makedirs(d, exist_ok=True)
        json.dump(res["dev"], open(os.path.join(d, "dev_probs.json"), "w"))
        json.dump(res["test"], open(os.path.join(d, "test_probs.json"), "w"))
        dev_items = cjpe_data.build_items(data["dev"], arrays["dev"], L, strat, k, stride)
        sc = T.dev_report(dev_items, res["dev"])
        info = {"run": cfg["name"], "variant": v, "chunk_len": L, "dev_coverage": res["dev_coverage"],
                "test_coverage": res["test_coverage"], "seconds": round(time.time() - t0, 1),
                **{f"{key}_f1_at_0.5": s["macro_f1"] for key, s in sc.items()}}
        json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=1)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
