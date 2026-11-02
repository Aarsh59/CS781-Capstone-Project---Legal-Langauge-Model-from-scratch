#!/usr/bin/env python3
"""Build ~/capstone/capstone_b2_ctxext.ipynb: continued pretraining of B2 with an extended context on Kaggle 2xT4,
resumable across 12 h sessions and accounts (plumbing; declare it).  python build_ctxext_notebook.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "capstone_b2_ctxext.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def writefile(name):
    return code(f"%%writefile {name}\n" + open(os.path.join(HERE, name)).read())


INTRO = """
# B2 context extension (896 → 1024) + continued pretraining on Kaggle 2×T4 (resumable)

**Do not start full training until the data manifest, contamination checks and this configuration have been reviewed.**
Session 1 runs the **probe only** (`RUN_TRAINING = False`): speed, memory and the session count for your token budget.

**Kaggle settings:** Accelerator **GPU T4 ×2**; internet not needed.
**Add Input:**
1. the output of `capstone_b2_ctxdata` (CPU data audit): `ctxdata/new_*.bin`, `ctxdata/heldout_*.bin`, `manifest_ctxext.json`;
2. the output of your Phase-1 **data** notebook: the original `train.bin`, `val.bin` (SC validation);
3. your Phase-1 **B2** output: `run_B2/best.pt` (read-only input: the original checkpoint is never modified);
4. from session 2 on: the previous session's output of THIS notebook (`ctxext/last.pt`). A private notebook's output is not
   visible to another account: on the old account use *Output → New Dataset* (private) and share it with the new account.

**Sampler:** fixed non-overlapping 1024-token windows, each used at most once, in a seeded order saved in `last.pt`
(`nr_sampler.py`); a resumed session continues at the saved cursor.

**Session rules (enforced by the script):** the same number of GPUs (2) and the same CONFIG in every session. A resume that
would change world size, batch tokens, tokens/step, token target, length, RoPE method, LR or schedule stops with an error.

Outputs in `/kaggle/working/ctxext/`:
- `last.pt`: full resumable state;
- `best_newctx.pt`: lowest mean 1024-token loss;
- `best_oldctx.pt`: lowest mean 896-token loss, i.e. retention of B2's original-context ability;
- `log.jsonl`.

Both best files use the Phase-1 `best.pt` format, so `phase3/backbones.py` loads them.
"""

SETUP = """
import os, sys, glob, json, time, subprocess
SESSION_START = time.time()
WORK = "/kaggle/working"
os.chdir(WORK); sys.path.insert(0, WORK)
import torch
N_GPUS = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])
assert N_GPUS == 2, "use Accelerator GPU T4 x2 (the run must use the same GPU count in every session)"

def find(pattern):
    hits = sorted(glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True))
    print(f"{pattern}: {hits if hits else 'NOT FOUND'}")
    return hits
NEW = {s: find(f"ctxdata/new_{s}.bin") for s in ("high_courts", "tribunals")}
UNIQ = {s: find(f"ctxdata/{s}_unique.bin") for s in ("supreme_court", "acts")}
HELD = {s: find(f"ctxdata/heldout_{s}.bin") for s in ("high_courts", "tribunals")}
MANIFEST = find("ctxdata/manifest_ctxext.json")
ORIG_TRAIN = [p for p in find("train.bin") if "/ctxdata/" not in p]
ORIG_VAL = [p for p in find("val.bin") if "/ctxdata/" not in p]
B2 = find("run_B2/best.pt")
PREVIOUS = find("ctxext/last.pt")
assert all(NEW.values()) and all(UNIQ.values()) and all(HELD.values()) and MANIFEST, "attach the capstone_b2_ctxdata output"
assert ORIG_TRAIN and ORIG_VAL and B2, "attach the Phase-1 data output (train.bin, val.bin) and the B2 output (run_B2/best.pt)"
man = json.load(open(MANIFEST[0]))
assert man["replay"].get("exact_match") is True, "the data audit's replay did not match the original train.bin: do not train on it"
print("data audit: replay verified | new training tokens available:", man["available_new_training_tokens"] / 1e9, "B")
print("resuming from a previous session" if PREVIOUS else "no previous ctxext/last.pt attached: session 1")
"""

CONFIG = """
# =====================================================================================
# YOUR CHOICES (course rule 9) - values from your execution plan (2026-10-16). Only the mixture is still open.
# =====================================================================================
TRAIN_MIXTURE = {      # your choice (2026-10-17): new HC + Tribunal data only; Phase-1 SC / Acts / train.bin not repeated
    "new_high_courts": 0.538,
    "new_tribunals": 0.462,
    "original_train": 0.0,
    "sc_unique": 0.0,
    "acts_unique": 0.0,
}
# files: new_high_courts / new_tribunals (genuinely new, deduplicated); sc_unique / acts_unique (Phase-1 SC / Acts docs once);
#        original_train (the Phase-1 train.bin as built). Weight 0 = file not used.
assert TRAIN_MIXTURE is not None, "choose the training mixture (see the data manifest for available tokens per file)"
FILES = {"new_high_courts": NEW["high_courts"][0], "new_tribunals": NEW["tribunals"][0], "sc_unique": UNIQ["supreme_court"][0],
         "acts_unique": UNIQ["acts"][0], "original_train": ORIG_TRAIN[0]}
CFG = {
    "tag": "B2_ctxext1024",
    "init_from": B2[0],
    "train_files": [{"path": FILES[k], "weight": w} for k, w in TRAIN_MIXTURE.items() if w != 0],
    "eval_sets": {"sc_val": ORIG_VAL[0], "hc_heldout": HELD["high_courts"][0], "trib_heldout": HELD["tribunals"][0]},
    "out_dir": os.path.join(WORK, "ctxext"),
    "resume_search": ["/kaggle/input/**/ctxext/last.pt"],
    "max_seq_len": 1024,
    "rope_extension": "extrapolate",
    "train": {
        "max_tokens": 1_650_000_000,     # total additional tokens (all sessions)
        "batch_tokens": 8192,            # global tokens per optimiser step (8 x 1024: 4 sequences per T4)
        "micro_batch": None,             # largest that fits (searched)
        "lr": 5e-5, "min_lr": 5e-6, "warmup_steps": 1000, "schedule": "cosine",
        "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0,
        "seed": 1234,
        # session plumbing (not hyperparameters)
        "compile": True,                 # torch.compile, as in the Phase-1 refine trainer (eager fallback)
        "session_hours": 11.0, "eval_every_min": 60, "ckpt_every_min": 30, "eval_windows": 200,
    },
}
json.dump(CFG, open("ctxext_config.json", "w"), indent=2)
# no-repeat sampler: max_tokens // 1024 fixed windows, each used once (allocation per file is printed by the trainer at setup)
n_win = CFG["train"]["max_tokens"] // CFG["max_seq_len"]
steps = -(-n_win // (CFG["train"]["batch_tokens"] // CFG["max_seq_len"]))
print(f"1024 tokens, extrapolate | {n_win:,} windows = {n_win * CFG['max_seq_len']:,} tokens "
      f"({CFG['train']['max_tokens'] - n_win * CFG['max_seq_len']} of the budget unused) = {steps:,} steps")
"""

PROBE = """
# Probe (session 1): 30 optimiser steps on both GPUs in memory (nothing is saved) -> tokens/s, peak memory per GPU,
# hours and 11-h sessions needed for max_tokens. Then the step-0 evaluation is NOT run; it runs at the start of training.
RUN_PROBE = True
if RUN_PROBE:
    subprocess.run(["torchrun", "--nproc_per_node=2", "train_ctxext.py", "ctxext_config.json", "--probe", "30"], check=True)
"""

TRAIN = """
# Full training: only after the review. Resumes automatically from the newest attached ctxext/last.pt.
RUN_TRAINING = False
if RUN_TRAINING:
    p = subprocess.run(["torchrun", "--nproc_per_node=2", "train_ctxext.py", "ctxext_config.json"])
    print("exit code", p.returncode, "| session hours used", round((time.time() - SESSION_START) / 3600, 2))
else:
    print("RUN_TRAINING = False (probe-only session)")
"""

SUMMARY = """
# progress (all sessions): per eval set, loss at 1024 (positions <896 / >=896) and at 896 (retention); step 0 = B2 itself
import pandas as pd
logf = os.path.join(CFG["out_dir"], "log.jsonl")
if os.path.exists(logf):
    recs = [json.loads(l) for l in open(logf)]
    rows = []
    for r in recs:
        if r["msg"].startswith("eval"):
            for name, v in r["eval"].items():
                rows.append({"step": r.get("step", 0), "set": name, **v})
    if rows:
        display(pd.DataFrame(rows).pivot_table(index="step", columns="set", values=["new_len", "old_len", "pos_ge_old"]))
    print(sorted(os.listdir(CFG["out_dir"])))
"""

cells = [md(INTRO), md("## Setup: GPUs and attached inputs"), code(SETUP), md("## Code files"),
         writefile("arch3.py"), writefile("fastattn.py"), writefile("nr_sampler.py"), writefile("train_ctxext.py"),
         md("## CONFIG (your choices)"), code(CONFIG), md("## Optional probe (speed / memory / ETA)"), code(PROBE),
         md("## Train (resumable)"), code(TRAIN), md("## Summary"), code(SUMMARY)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT, len(cells), "cells")
