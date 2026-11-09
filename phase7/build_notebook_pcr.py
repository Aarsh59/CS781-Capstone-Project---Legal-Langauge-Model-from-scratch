#!/usr/bin/env python3
"""Build the Kaggle notebook ~/capstone/capstone_phase7_pcr.ipynb from the files in this folder (plumbing).
Every .py file becomes a %%writefile cell (your techniques/*.py and experiments.py are editable there too).
    python build_notebook_pcr.py [--out PATH]"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working/phase7"

PLUMBING = ["a1_tokenizer.py"] + [f"phase1_models/{f}" for f in (
    "rope.py", "arch1.py", "arch3.py", "attention3.py", "non_causal_self_attention.py", "cross_attention.py",
    "encoder.py", "decoder4.py", "enc_dec.py", "prefix_attention.py", "prefix_lm.py",
    "fastattn_ab.py", "fastattn_cd.py")] + [
    "fetch_data.py", "pcr_data.py", "backbones.py", "inject.py", "techniques/__init__.py", "techniques/full_ft.py",
    "train_pcr.py", "run_many.py", "pcr_select.py"]
YOURS = ["techniques/partial_ft.py", "techniques/qlora.py", "techniques/lorma.py", "techniques/prefix_tuning.py",
         "techniques/prompt_tuning.py", "techniques/ia3.py", "techniques/kd.py", "experiments.py"]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def writefile(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return code(f"%%writefile {WORK}/{rel}\n" + f.read())


INTRO = """
# CS781 Capstone — Phase 7: Prior Case Retrieval (PCR) on 2×T4

**Kaggle settings:** GPU T4 ×2, Internet **on**, secret **`HF`** or **`HF_TOKEN`** (ticked for this notebook). **Add Input:**
the private Phase-1 model datasets (A, B2, C+D, B3 = `ctxext/best_newctx.pt`; only the ones your matrix uses); from a
second session on also earlier outputs of this notebook, so finished runs are reused (and can be KD teachers).

**Task plumbing:** bi-encoder on your Phase-1 models (Phase-4/5 chunked document encoder: chunks -> chunk pooling ->
attention / mean pooling over chunks -> optional projection), score = cosine or dot / temperature, InfoNCE over each
(query, relevant case) pair's candidates (BM25 hard or random negatives, optional in-batch negatives), gradient caching
so a step's documents share one loss. Selection and evaluation: dev Micro-F1@7 against `dev_candidates.json`.
Output: `prediction.json` = {query id: 7 candidate ids from `test_candidates.json`}. KD: your `kd_loss` over the
student's and the teacher's scores of the same candidate rows.

**Your parts (course rule 9):** `techniques/*.py` (Phase-5 files + IA³ from Phase 3), every value in `experiments.py`,
the fusion weights in Stage 7, and all analysis.

**ACCOUNT (Stage 3):** `1` / `2` = this account's share of `experiments.SPLIT`, `"all"` = everything here.
Outputs: `/kaggle/working/phase7/runs/<name>/`, zips in `/kaggle/working/submissions/`.
"""

SETUP = f"""
import os, sys, json, glob, time, hashlib, subprocess, shutil, inspect
SESSION_START = time.time()
WORK = "{WORK}"
for d in ("phase1_models", "techniques", "configs", "runs", "data", "cache"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
import torch
N_GPUS = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])
"""

DATA = """
# Phase_7_PCR from the gated HF dataset (Kaggle secret "HF"), file schema check, Phase-1 inputs, token cache
from kaggle_secrets import UserSecretsClient
_hf, _errs = None, {}
for _label in ("HF", "HF_TOKEN"):
    try:
        _hf = UserSecretsClient().get_secret(_label)
        if _hf:
            print(f"Hugging Face token: Kaggle secret '{_label}'")
            break
    except Exception as e:
        _errs[_label] = type(e).__name__
if not _hf:
    raise SystemExit(f"Could not read a Kaggle secret 'HF' or 'HF_TOKEN' ({_errs}): Add-ons -> Secrets -> add your Hugging "
                     "Face token under one of these labels and TICK it for this notebook; then rerun this cell.")
subprocess.run([sys.executable, "fetch_data.py", "--out", os.path.join(WORK, "data")], check=True,
               env=dict(os.environ, HF_TOKEN=_hf.strip()))
del _hf
DATA_DIR = os.path.join(WORK, "data", "Phase_7_PCR")

import pcr_data
print(json.dumps(pcr_data.schema_report(DATA_DIR), indent=1))        # check the field names the loader picked up
SPLIT_DATA = {s: pcr_data.load_split(DATA_DIR, s) for s in ("train", "dev", "test")}
for s, (q, p) in SPLIT_DATA.items():
    print(s, pcr_data.split_stats(q, p))

def find_all(pattern):
    return sorted(glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True))

def find_one(pattern):
    hits = find_all(pattern)
    if len(hits) > 1:
        print(f"  note: {len(hits)} matches for {pattern}, using the first: {hits}")
    return hits[0] if hits else None

def sha8(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]

TOKS = find_all("legal_bpe.json")
assert TOKS, "legal_bpe.json not found under /kaggle/input (Add Input: your Phase-1 notebook outputs)"
tok_hashes = {t: sha8(t) for t in TOKS}
assert len(set(tok_hashes.values())) == 1, f"different legal_bpe.json files attached: {tok_hashes}"
TOKENIZER = TOKS[0]
print("tokenizer sha256:", tok_hashes[TOKENIZER], "(Phase-1 legal BPE was 38f51f14)")

PHASE1 = {"A": find_one("run_A/best.pt"), "B2": find_one("run_B2/best.pt"), "B3": find_one("ctxext/best_newctx.pt"),
          "C": find_one("run_C/best.pt"), "D": find_one("run_D/best.pt")}
for k, v in PHASE1.items():
    if v is None:
        print(f"Phase-1 {k}: not attached"); continue
    ck = torch.load(v, map_location="cpu", weights_only=False, mmap=True)
    c = ck["config"]
    lim = c["model"]["max_seq_len"] if "model" in c else c["encoder"]["max_seq_len"]
    print(f"Phase-1 {k}: {v}\\n    tag={c.get('tag')} module={c['module']} step={ck.get('step')} max_len={lim}")
    del ck

from a1_tokenizer import ByteBPE
_tok = ByteBPE.from_json(TOKENIZER)
t0 = time.time()
for s, (q, p) in SPLIT_DATA.items():                 # same cache files train_pcr.load_data uses
    pcr_data.encode_docs(_tok, q, os.path.join(WORK, "cache", f"{s}_queries_ids.npz"))
    pcr_data.encode_docs(_tok, p, os.path.join(WORK, "cache", f"{s}_pool_ids.npz"))
print(f"token cache ready in {time.time() - t0:.0f} s")
del _tok
"""

BM25_REF = """
# BM25 reference on dev (lexical, no model; NOT a valid Phase-7 experiment: every run must start from a Phase-1 model).
# Useful as a yardstick for the dev scores below and for hard-negative quality.
subprocess.run([sys.executable, "pcr_select.py", "--bm25-only", "--data-dir", DATA_DIR,
                "--out", os.path.join(WORK, "ensembles")], check=True)
"""

CONFIG = """
# Experiment matrix = experiments.py (YOUR values). RUN = what this session trains, in queue order.
import importlib, experiments, train_pcr, techniques
importlib.reload(experiments)
EXPERIMENTS = experiments.build(PHASE1, DATA_DIR, TOKENIZER, os.path.join(WORK, "runs"), os.path.join(WORK, "cache"))
ACCOUNT = None                                  # 1 / 2 (two Kaggle accounts in parallel) or "all"
assert ACCOUNT in (1, 2, "all"), "set ACCOUNT = 1, 2 or 'all' in this cell"
RUN = (list(EXPERIMENTS) + [experiments.KD_NAME]) if ACCOUNT == "all" else list(experiments.SPLIT[ACCOUNT])
unknown = [n for n in RUN if n not in EXPERIMENTS and n != experiments.KD_NAME]
assert not unknown, f"SPLIT names not in the matrix: {unknown}"

def find_run(name):
    # a finished run (summary.json + best_model.pt) in WORK/runs or in an attached earlier output of this notebook
    for d in [os.path.join(WORK, "runs", name)] + sorted(glob.glob(f"/kaggle/input/**/runs/{name}", recursive=True)):
        if all(os.path.exists(os.path.join(d, f)) for f in ("summary.json", "best_model.pt", "config.json")):
            if not json.load(open(os.path.join(d, "summary.json"))).get("smoke"):
                return d
    return None

if experiments.KD_NAME in RUN:                  # KD: teacher queued earlier in RUN (run_many waits) or finished/attached
    _t = experiments.KD_TEACHER
    assert _t, "set experiments.KD_TEACHER (a run of the matrix)"
    _td = os.path.join(WORK, "runs", _t) if _t in RUN[:RUN.index(experiments.KD_NAME)] else find_run(_t)
    assert _td, f"KD teacher {_t} is neither queued before {experiments.KD_NAME} nor a finished attached run"
    EXPERIMENTS[experiments.KD_NAME] = experiments.build_kd(_td, PHASE1, DATA_DIR, TOKENIZER, os.path.join(WORK, "runs"),
                                                            os.path.join(WORK, "cache"))

def not_implemented(tech_name):
    src = inspect.getsource(techniques.get(tech_name))
    return "raise NotImplementedError" in src

DONE = {n: find_run(n) for n in RUN if find_run(n)}
if DONE:
    print("already finished (reused, not trained again):", DONE)
CFG_PATHS = []
for name in RUN:
    if name in DONE:
        continue
    cfg = EXPERIMENTS[name]
    if not_implemented(cfg["technique"]["name"]):
        print(f"SKIP {name}: techniques/{cfg['technique']['name']}.py is not implemented"); continue
    train_pcr.check_config(cfg)
    path = os.path.join(WORK, "configs", f"{name}.json")
    json.dump(cfg, open(path, "w"), indent=2)
    CFG_PATHS.append(path)
    ng, ch = cfg["negatives"], cfg["chunking"]
    print(f"{name:12s} | model={cfg['model_key']:3s} | technique={cfg['technique']['name']:13s} | kd={bool(cfg.get('kd'))} | "
          f"lr={cfg['train']['lr']} | {cfg['similarity']}/T={cfg['temperature']} | neg={ng['source']}x{ng['per_positive']} "
          f"in_batch={ng['in_batch']} | chunks={ch['strategy']} x{ch['max_chunks']} | pool={cfg['chunk_pool']}/{cfg['doc_pool']}")
print(f"{len(CFG_PATHS)} runs queued")
"""

SMOKE = """
# Real-model smoke test on both GPUs BEFORE the long runs: every queued config unchanged (checkpoint, technique, chunking,
# micro-batch), on the 8 longest train queries + small pools, 4 optimizer steps, dev/test on 4 queries, zip. Measures peak
# GPU memory and training speed -> rough minutes per full run (written into the configs as est_minutes, used by the
# scheduler when train.max_minutes is unset). Outputs: runs_smoke/.
import numpy as np
RUN_SMOKE = True
def _tokens(split, part):
    z = np.load(os.path.join(WORK, "cache", f"{split}_{part}_ids.npz"))
    return np.diff(z["offsets"])
if RUN_SMOKE and CFG_PATHS:
    smoke_paths = []
    for path in CFG_PATHS:
        c = json.load(open(path))
        c["smoke"] = {"steps": 4, "queries": 8}
        c["out_dir"] = os.path.join(WORK, "runs_smoke")
        if c.get("kd") and os.path.basename(c["kd"]["teacher_run"].rstrip("/")) in RUN:     # the teacher's smoke run
            c["kd"]["teacher_run"] = os.path.join(WORK, "runs_smoke", os.path.basename(c["kd"]["teacher_run"].rstrip("/")))
        sp = os.path.join(WORK, "configs", f"smoke_{c['name']}.json")
        json.dump(c, open(sp, "w"), indent=2)
        smoke_paths.append(sp)
    t0 = time.time()
    gpus = ",".join(str(i) for i in range(max(1, N_GPUS)))
    sp_proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "60", *smoke_paths])
    print(f"\\nsmoke test: {(time.time() - t0) / 60:.1f} min")
    n_pairs = sum(len(q["relevant"]) for q in SPLIT_DATA["train"][0])
    for path in CFG_PATHS:
        c = json.load(open(path))
        sm = os.path.join(WORK, "runs_smoke", c["name"], "summary.json")
        if not os.path.exists(sm):
            print(f"  FAIL {c['name']:12s} -> see {os.path.join(WORK, 'runs_smoke', c['name'] + '.stdout')}"); continue
        s_ = json.load(open(sm))
        tok_s = max(1, s_["train_tok_s"])
        L, k = s_["chunking"]["chunk_len"], c["chunking"]["max_chunks"]
        per_doc = lambda a: float(np.mean(np.minimum(a, L * k)))
        docs_per_pair = 2 + c["negatives"]["per_positive"]
        train_tok = n_pairs * docs_per_pair * per_doc(_tokens("train", "pool"))        # tokens re-encoded with grad
        eval_tok = per_doc(_tokens("dev", "pool")) * len(SPLIT_DATA["dev"][1]) + per_doc(_tokens("dev", "queries")) * len(SPLIT_DATA["dev"][0])
        n_eval = c["train"]["epochs"] * (1 + (n_pairs // c["train"]["batch_pairs"]) // (c["train"]["eval_every_steps"] or 10**9)) + 2
        est = 1.3 * (c["train"]["epochs"] * train_tok / tok_s + n_eval * eval_tok / (2.5 * tok_s)) / 60 + 15
        c["est_minutes"] = round(est, 1)
        json.dump(c, open(path, "w"), indent=2)
        print(f"  OK   {c['name']:12s} peak {s_['peak_mem_GB']} GB of ~15 | trainable {s_['params']['trainable']:,} "
              f"({s_['params']['trainable_pct']}%) | {tok_s} tok/s | ~{c['est_minutes']} min per full run (rough)")
    assert sp_proc.returncode == 0, "smoke test failed: fix the runs marked FAIL before launching the full runs"
    for ck in glob.glob(os.path.join(WORK, "runs_smoke", "*", "best_model.pt")):
        os.remove(ck)
    print("SMOKE TEST PASSED -> run the next cell for the full runs")
"""

LAUNCH = """
# Main runs on BOTH T4s (one run per GPU at a time, shared queue in RUN order; the KD run waits for its teacher).
# A run is only STARTED if its time (train.max_minutes, else est_minutes) fits before SESSION_HOURS; others are SKIPPED.
assert N_GPUS >= 1, "no GPU visible: set Accelerator to 'GPU T4 x2'"
gpus = ",".join(str(i) for i in range(N_GPUS))
SESSION_HOURS = 11.5
STOP_AT = SESSION_START + SESSION_HOURS * 3600
print(f"{(STOP_AT - time.time()) / 3600:.2f} h of session left for starting runs")
if CFG_PATHS:
    proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "600", "--stop-at", str(STOP_AT),
                           "--default-minutes", "180", "--margin", "15", *CFG_PATHS])
    print(f"run_many exit code {proc.returncode} | session time used {(time.time() - SESSION_START) / 3600:.2f} h")
"""

RESULTS = """
# results table (one row per finished run, facts only) + all zips in /kaggle/working/submissions
import pandas as pd
rows = []
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for n in list(experiments.matrix()) + [experiments.KD_NAME]:
    d = find_run(n)
    if not d:
        rows.append(dict(name=n, status="not finished")); continue
    s = json.load(open(os.path.join(d, "summary.json")))
    a7 = s["dev"]["at7"]
    rows.append(dict(name=n, status="ok" if s["full_epoch_completed"] else "INCOMPLETE EPOCH", model=s["model_key"],
                     technique=s["technique"]["name"], kd=s["kd"], sim=s["similarity"], T=s["temperature"],
                     neg=f"{s['negatives']['source']}x{s['negatives']['per_positive']}" + ("+inbatch" if s["negatives"]["in_batch"] else ""),
                     chunks=f"{s['chunking']['strategy']}x{s['chunking']['max_chunks']}@{s['chunking']['chunk_len']}",
                     pool=f"{s['chunk_pool']}/{s['doc_pool']}", params=s["params"]["total"],
                     trainable=s["params"]["trainable"], trainable_pct=s["params"]["trainable_pct"],
                     dev_F1_at7_untrained=s["start_dev_f1_at7"], dev_F1_at7=a7["micro_f1"], dev_P_at7=a7["micro_p"],
                     dev_R_at7=a7["micro_r"], dev_R_at100=s["dev"]["recall@100"], selected_at=s["selected_at"],
                     epochs=s["epochs_completed"], train_min=round(s["train_seconds"] / 60, 1), peak_GB=s["peak_mem_GB"],
                     infer_ms_per_doc=s["infer_ms_per_doc"]))
    z = os.path.join(d, f"submission_{n}.zip")
    if os.path.exists(z):
        shutil.copy(z, "/kaggle/working/submissions/")
df = pd.DataFrame(rows)
display(df)
df.to_csv("/kaggle/working/submissions/summary.csv", index=False)
print(sorted(os.listdir("/kaggle/working/submissions")))
"""

SELECT = """
# Score fusion of finished runs (+ optional BM25), YOUR weights (nothing is tuned): per-query normalised scores
# (norm: zscore | minmax | none), weighted sum -> dev Micro-F1@7 printed, test submission written.
#   name -> {"members": [run names], "weights": [one per member], "bm25_weight": float, "norm": str}
FUSION = {
    "ENS1": {"members": None, "weights": None, "bm25_weight": None, "norm": None},
}
for fname, e in FUSION.items():
    if not e["members"]:
        print(f"{fname}: skipped (fill members / weights / bm25_weight / norm)"); continue
    dirs = [find_run(m) for m in e["members"]]
    assert all(dirs), f"{fname}: unfinished members {[m for m, d in zip(e['members'], dirs) if not d]}"
    subprocess.run([sys.executable, "pcr_select.py", "--name", fname, "--out", os.path.join(WORK, "ensembles"),
                    "--data-dir", DATA_DIR, "--members", *dirs, "--weights", *map(str, e["weights"]),
                    "--bm25-weight", str(e["bm25_weight"]), "--norm", e["norm"]], check=True)
    shutil.copy(os.path.join(WORK, "ensembles", fname, f"submission_{fname}.zip"), "/kaggle/working/submissions/")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "capstone_phase7_pcr.ipynb"))
    args = ap.parse_args()
    cells = [md(INTRO), md("## Stage 0 — session, folders"), code(SETUP),
             md("## Stage 1 — plumbing files (your Phase-1 model code unchanged; declare the rest)")]
    cells += [writefile(f) for f in PLUMBING]
    cells += [md("## Stage 1b — YOUR files: techniques, experiment matrix")]
    cells += [writefile(f) for f in YOURS]
    cells += [md("## Stage 2 — data, schema check, Phase-1 inputs, token cache"), code(DATA),
              md("## Stage 2b — BM25 reference score on dev (not a submission)"), code(BM25_REF),
              md("## Stage 3 — configuration (experiments.py)"), code(CONFIG),
              md("## Stage 3b — real-model smoke test (memory, speed, time per run)"), code(SMOKE),
              md("## Stage 4 — main runs incl. KD (both GPUs)"), code(LAUNCH),
              md("## Stage 5 — results and zips"), code(RESULTS),
              md("## Stage 6 — score fusion (your weights)"), code(SELECT)]
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            compile(src.split("\n", 1)[1] if src.startswith("%%writefile") else src, f"<cell {i}>", "exec")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}: {len(cells)} cells")


if __name__ == "__main__":
    main()
