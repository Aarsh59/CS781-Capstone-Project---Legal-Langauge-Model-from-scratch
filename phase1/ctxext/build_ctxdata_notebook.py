#!/usr/bin/env python3
"""Build ~/capstone/capstone_b2_ctxdata.ipynb: CPU-only Kaggle data audit + build for the B2 context extension
(plumbing; declare it).  python build_ctxdata_notebook.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "capstone_b2_ctxdata.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def writefile(src, name):
    return code(f"%%writefile {name}\n" + open(os.path.join(HERE, src)).read())


INTRO = """
# B2 context extension: CPU data audit + build of genuinely new pretraining data

**Kaggle settings:** Accelerator **None (CPU)**, Internet **on**, secret **`HF`** (gated course dataset).
**Add Input:** the output of your Phase-1 **data** notebook (`aarsh59/notebook08c3659751`): `train.bin`, `val.bin`, `legal_bpe.json`.

What it does (details: docstring of `data_audit.py`):
1. downloads the Phase-1 corpus (pinned to the dataset revision the Phase-1 build used, when available);
2. **replays the exact Phase-1 selection** (verbatim code of cells [17] + [19] of the Phase-1 notebook) and checks the
   replayed `train.bin` / `val.bin` byte-for-byte against your originals. Only a match counts as a reconstruction;
3. scans every High Court / Tribunal document: used before / exact duplicate / near-duplicate of training text /
   contamination with any Phase 1–7 test, dev or validation file / candidate;
4. encodes the candidates with your exact `legal_bpe.json` into `new_high_courts.bin`, `new_tribunals.bin`, writes every Phase-1
   SC / Acts training document once into `supreme_court_unique.bin` / `acts_unique.bin` (repeats, separately weightable), and per-source
   held-out evaluation files `heldout_*.bin` (never trained on);
5. writes `manifest_ctxext.json` (counts at every step, overlap histograms, contamination sources, available tokens).

Nothing is trained. Expected runtime: about 2–4 h on Kaggle CPU (dominated by the corpus scan and encoding).
"""

SETUP = """
import os, sys, glob, json, time, subprocess
WORK = "/kaggle/working"; os.chdir(WORK); sys.path.insert(0, WORK)
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF")
from huggingface_hub import hf_hub_download, list_repo_files
REPO = "Exploration-Lab/CS-781-Capstone"
PHASE1_REVISION = "89edba6a44d6bb9593fd13b6895b5cd3b06eb82e"   # snapshot recorded in the Phase-1 run configs (run_A test_path)
try:
    files = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN, revision=PHASE1_REVISION); REV = PHASE1_REVISION
except Exception as e:
    print("pinned revision unavailable, using main:", repr(e)[:200]); REV = "main"
    files = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN)
fetch = lambda f: hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset", token=HF_TOKEN, revision=REV)
CORPUS = {s: fetch(f"Phase_1_Pretraining/Train/train_{s}.json") for s in ("supreme_court", "acts", "high_courts", "tribunals")}
for s, p in CORPUS.items():
    print(f"{s:14s} {os.path.getsize(p) / 1e9:6.2f} GB  {p}")
# protected (must never enter training): Phase-1 test + every test / dev / validation file of Phases 2-7 in the repo
latest = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN)        # later phases may only exist on main
is_eval = lambda f: f.endswith(".json") and not f.startswith("Phase_1_Pretraining/Train/") and any(
    k in f.lower() for k in ("test", "dev", "val"))
PROTECTED = []
for f in sorted(set(latest) | set(files)):
    if is_eval(f):
        try:
            PROTECTED.append(hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset", token=HF_TOKEN))
        except Exception as e:
            print("could not fetch", f, repr(e)[:120])
print(len(PROTECTED), "protected files:", [p.split("/snapshots/")[-1] for p in PROTECTED])
find = lambda pat: sorted(glob.glob(os.path.join("/kaggle/input", "**", pat), recursive=True))
ORIG_TRAIN, ORIG_VAL, TOK = find("train.bin"), find("val.bin"), find("legal_bpe.json")
print("original train.bin:", ORIG_TRAIN, "| val.bin:", ORIG_VAL, "| tokenizer:", TOK)
assert ORIG_TRAIN and ORIG_VAL and TOK, "attach the Phase-1 data notebook output (train.bin, val.bin, legal_bpe.json)"
"""

CONFIG = """
# Phase-1 constants: copied verbatim from capstone_phase1_arch12.ipynb cell [4] (needed for an exact replay; not choices)
P1 = dict(SEED=42, TOTAL_TRAIN_TOKENS=1_000_000_000, SC_MAX_EPOCHS=4.0, SC_MAX_SHARE=0.70, ACTS_MAX_EPOCHS=2.0, ACTS_MAX_SHARE=0.08,
          SC_VAL_PERCENT=2, MIN_WORDS=100, MAX_FERTILITY=2.0, N_VAL_PROMPTS=300, PROFILE_CHARS=30_000_000, HT_SCAN_FACTOR=4)
CFG = {
    "out_dir": os.path.join(WORK, "ctxdata"), "scratch_dir": "/tmp/ctxdata_scratch", "p1_code_dir": os.path.join(WORK, "p1"),
    "tokenizer": TOK[0], "corpus_files": CORPUS, "original_train_bin": ORIG_TRAIN[0], "original_val_bin": ORIG_VAL[0],
    "phase1_constants": P1, "workers": os.cpu_count(),
    # data-cleaning settings (defaults; review them in the manifest's overlap histograms):
    "heldout_percent": 1,                       # share of new HC/Tribunal documents kept out of training for evaluation
    "near_dup": {"shingle_words": 13, "sample_mod": 16,
                 "train_threshold": 0.5,        # >= this share of a document's 13-word shingles already in training -> dropped
                 "protected_threshold": 0.2,    # >= this share found in any test/dev/val text -> dropped (contamination)
                 "protected_min_words": 50},
    "protected_files": PROTECTED, "protected_root": "/root/.cache/huggingface/hub",
    "scan_batch_docs": 20000, "continue_if_replay_mismatch": False,
}
json.dump(CFG, open("data_audit_config.json", "w"), indent=2)
print("config written; dataset revision:", REV)
"""

RUN = """
p = subprocess.run([sys.executable, "data_audit.py", "data_audit_config.json"])
print("exit code", p.returncode)
"""

SUMMARY = """
m = json.load(open(os.path.join(CFG["out_dir"], "manifest_ctxext.json")))
print("replay:", json.dumps(m["replay"], indent=1))
if "new_data_scan" in m:
    import pandas as pd
    display(pd.DataFrame(m["original_selection"]).T)
    display(pd.DataFrame(m["new_data_scan"]).T)
    print("available NEW training tokens (HC + Tribunals):", m["available_new_training_tokens"] / 1e9, "B")
    print("repeat-only unique tokens (seen in Phase 1):", {k: v / 1e9 for k, v in m["repeat_only_unique_tokens"].items()}, "B")
    print("protected sources:", m["protected_sources"])
print(sorted(os.listdir(CFG["out_dir"])))
"""

cells = [md(INTRO), md("## Setup: corpus, protected eval files, inputs"), code(SETUP), md("## Code (verbatim Phase-1 cells + audit)"),
         writefile("p1/a1_tokenizer.py", "p1/a1_tokenizer.py"), writefile("p1/cell17_defs.py", "p1/cell17_defs.py"),
         writefile("p1/cell19.py", "p1/cell19.py"), writefile("data_audit.py", "data_audit.py"),
         md("## Config"), code(CONFIG), md("## Run the audit"), code(RUN), md("## Manifest summary"), code(SUMMARY)]
cells.insert(3, code("os.makedirs('p1', exist_ok=True)"))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT, len(cells), "cells")
