#!/usr/bin/env python3
"""Build the Kaggle notebook ~/capstone/capstone_phase4_cjpe.ipynb from the files in this folder (plumbing).
Every .py file becomes a %%writefile cell (your heads.py / techniques/*.py / experiments.py are editable there too).
    python build_notebook_cjpe.py [--out PATH]"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working/phase4"

PLUMBING = ["a1_tokenizer.py"] + [f"phase1_models/{f}" for f in (
    "rope.py", "arch1.py", "arch3.py", "attention3.py", "non_causal_self_attention.py", "cross_attention.py",
    "encoder.py", "decoder4.py", "enc_dec.py", "prefix_attention.py", "prefix_lm.py",
    "fastattn_ab.py", "fastattn_cd.py")] + [
    "fetch_data.py", "cjpe_data.py", "backbones.py", "inject.py", "techniques/__init__.py", "techniques/full_ft.py",
    "train_cjpe.py", "run_many.py", "cjpe_select.py", "predict_cjpe.py"]
YOURS = ["heads.py", "techniques/partial_ft.py", "techniques/bitfit.py", "techniques/lorma.py", "techniques/adalora.py",
         "techniques/adapters.py", "techniques/kd.py", "experiments.py"]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def writefile(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return code(f"%%writefile {WORK}/{rel}\n" + f.read())


INTRO = """
# CS781 Capstone — Phase 4: Court Judgment Prediction (CJPE) on 2×T4

**Kaggle settings:** Accelerator **GPU T4 ×2**, Internet **on**, secret **`HF`** or **`HF_TOKEN`** (ticked for this notebook). **Add Input:**
- your Phase-1 outputs with `run_A/best.pt`, `run_B2/best.pt`, `run_C/best.pt`, `run_D/best.pt` and `legal_bpe.json`, as in Phases 2 and 3;
- the B2 context-extension output (`notebook5e8c8b58f8`), which has `ctxext/best_newctx.pt` (= **B3**);
- from session 2 on (or for the final ensemble): this notebook's earlier outputs, from EITHER account, so finished runs
  are reused and not trained again.

**Two accounts:** set `ACCOUNT = 1` or `2` in Stage 3. Inputs on the second account come from PRIVATE datasets shared
with it (never public: capstone §17.1 forbids sharing model weights).

**Your parts (course rule 9):** `heads.py`, `techniques/*.py` and the experiment matrix in `experiments.py`.
Everything else is plumbing (declare it): data, chunking, chunk/document pooling, training loop, macro-F1,
threshold selection, ensembles, prediction.json + zip.

Data rules: train de-duplicated (31,653 distinct texts); dev scored as released (**dev_all**, used for selection) and
without the 1,517 documents that are also in test (**dev_excl_test**); test labels are never read.

Order: 0 → 1 → 1b → 2 (data, inputs, token cache) → 3 (configs) → 3b (smoke: memory, speed, time per run) →
4 (main runs, both GPUs) → 5 (KD with the best finished run as teacher) → 6 (results) → 7 (threshold / ensemble).
Outputs: `/kaggle/working/phase4/runs/<name>/`; all zips in `/kaggle/working/submissions/`.
"""

SETUP = f"""
import os, sys, json, glob, time, hashlib, subprocess, shutil, inspect
SESSION_START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-crf"], check=True)   # heads.py imports torchcrf
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
# CJPE files from the gated HF dataset (Kaggle secret "HF"), Phase-1 inputs, token cache (built once, shared by all runs)
from kaggle_secrets import UserSecretsClient     # read the secret here and hand it to the download script
_hf, _errs = None, {}
for _label in ("HF", "HF_TOKEN"):                # either label works
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
assert _hf and _hf.strip(), "the 'HF' secret is empty"
subprocess.run([sys.executable, "fetch_data.py", "--out", os.path.join(WORK, "data"), "--skip-ref"], check=True,
               env=dict(os.environ, HF_TOKEN=_hf.strip()))
del _hf
DATA_DIR = os.path.join(WORK, "data", "Phase_4_CJPE")

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

PHASE1 = {                                   # first match under /kaggle/input; the original B is never used
    "A": find_one("run_A/best.pt"),
    "B2": find_one("run_B2/best.pt"),
    "B3": find_one("ctxext/best_newctx.pt"),
    "C": find_one("run_C/best.pt"),
    "D": find_one("run_D/best.pt"),
}
for k, v in PHASE1.items():
    if v is None:
        print(f"Phase-1 {k}: NOT FOUND"); continue
    ck = torch.load(v, map_location="cpu", weights_only=False, mmap=True)
    c = ck["config"]
    lim = c["model"]["max_seq_len"] if "model" in c else c["encoder"]["max_seq_len"]
    print(f"Phase-1 {k}: {v}\\n    tag={c.get('tag')} module={c['module']} step={ck.get('step')} max_len={lim}")
    del ck
assert PHASE1["B2"] and "run_B/" not in PHASE1["B2"], "B2 must be run_B2/best.pt (the original B is not used)"

import cjpe_data
from a1_tokenizer import ByteBPE
_tok = ByteBPE.from_json(TOKENIZER)
_tr, DATA_REPORT = cjpe_data.dedup_train(cjpe_data.load_split(DATA_DIR, "train"))
_dv, _te = cjpe_data.load_split(DATA_DIR, "dev"), cjpe_data.load_split(DATA_DIR, "test")
DATA_REPORT["dev_docs_in_test"] = cjpe_data.mark_dev_excl_test(_dv, _te)
t0 = time.time()
for s, docs in (("train", _tr), ("dev", _dv), ("test", _te)):
    cjpe_data.encode_docs(_tok, docs, os.path.join(WORK, "cache", f"{s}_ids.npz"))
print(DATA_REPORT, f"| token cache ready in {time.time() - t0:.0f} s")
N_TRAIN = len(_tr)
del _tr, _dv, _te, _tok
"""

CONFIG = """
# Experiment matrix = experiments.py (YOUR values). RUN = what this session trains, in priority order.
import importlib, experiments, train_cjpe, techniques
importlib.reload(experiments)
BATCH_DOCS = experiments.BASE["train"]["batch_docs"]
STEPS_PER_EPOCH = -(-N_TRAIN // BATCH_DOCS)
EXPERIMENTS = experiments.build(PHASE1, DATA_DIR, TOKENIZER, os.path.join(WORK, "runs"), os.path.join(WORK, "cache"),
                                STEPS_PER_EPOCH)
# Two Kaggle accounts in parallel: set ACCOUNT = 1 on the first account and ACCOUNT = 2 on the second.
# Account 1 also runs KD (Stage 5); its teacher is the best finished run it can see (its own, plus account 2's runs if
# that output is attached). ACCOUNT = "all" runs everything in one account (two sessions).
ACCOUNT = None
SPLIT = {
    1: ["E1_B2_full", "E2_B3_full", "E3_A_partial", "E4_C_adapters", "E5_D_bitfit"],
    2: ["E6_B2_lorma", "E7_B2_adalora", "C1_B2_headtail", "C3_B2_overlap", "W1_B2_full_weighted"],
}
ALL_RUNS = SPLIT[1] + SPLIT[2]
# ACCOUNT = "long": train nothing from the matrix here; only Stage 8 (longer training of finished runs)
# ACCOUNT = "predict": train nothing; only Stage 9 (inference-only re-chunking of finished runs)
assert ACCOUNT in (1, 2, "all", "long", "predict"), "set ACCOUNT = 1 or 2 (or 'all' / 'long' / 'predict') in this cell"
RUN = [] if ACCOUNT in ("long", "predict") else (ALL_RUNS if ACCOUNT == "all" else SPLIT[ACCOUNT])

def find_run(name):
    # a finished run (summary.json + best_model.pt) in WORK/runs or in an attached earlier output of this notebook
    for d in [os.path.join(WORK, "runs", name)] + sorted(glob.glob(f"/kaggle/input/**/runs/{name}", recursive=True)):
        if all(os.path.exists(os.path.join(d, f)) for f in ("summary.json", "best_model.pt", "config.json")):
            if not json.load(open(os.path.join(d, "summary.json"))).get("smoke"):
                return d
    return None

def not_implemented(tech_name):
    src = inspect.getsource(techniques.get(tech_name))
    return "NotImplementedError(" in src and "raise NotImplementedError" in src

DONE = {n: find_run(n) for n in RUN if find_run(n)}
if DONE:
    print("already finished (reused, not trained again):", DONE)
CFG_PATHS = []
for name in RUN:
    if name in DONE:
        continue
    cfg = EXPERIMENTS[name]
    if not_implemented(cfg["technique"]["name"]):
        print(f"SKIP {name}: techniques/{cfg['technique']['name']}.py is not implemented yet"); continue
    train_cjpe.check_config(cfg)
    path = os.path.join(WORK, "configs", f"{name}.json")
    json.dump(cfg, open(path, "w"), indent=2)
    CFG_PATHS.append(path)
    print(f"{name:22s} | model={cfg['model_key']:3s} | technique={cfg['technique']['name']:10s} | lr={cfg['train']['lr']} "
          f"head_lr={cfg['train']['head_lr']} | chunks={cfg['chunking']['strategy']} x{cfg['chunking']['max_chunks']} "
          f"(len {cfg['chunking']['chunk_len'] or 'model limit'}) | pool={cfg['chunk_pool']}/{cfg['doc_pool']} | "
          f"class_weights={cfg['train']['class_weights']}")
print(f"{len(CFG_PATHS)} runs queued | steps/epoch {STEPS_PER_EPOCH} ({N_TRAIN} train docs, {BATCH_DOCS} per step)")
"""

SMOKE = """
# Real-model smoke test on both GPUs BEFORE the long runs: every queued config, unchanged (real checkpoint, technique,
# chunking, micro-batch), on the 80 LONGEST train documents (worst-case memory), 10 optimizer steps, dev/test on 4 docs,
# save -> reload -> zip. Measures peak GPU memory and training speed -> estimated minutes per full run (written into
# the configs, so the scheduler never starts a run that cannot finish in this session). Outputs: runs_smoke/.
RUN_SMOKE = True
if RUN_SMOKE and CFG_PATHS:
    smoke_paths = []
    for path in CFG_PATHS:
        c = json.load(open(path))
        c["smoke"] = {"steps": 10, "docs": 80}          # 80 longest docs / 8 per step = 10 steps (speed estimate)
        c["out_dir"] = os.path.join(WORK, "runs_smoke")
        if c["technique"]["name"] == "adalora":           # smoke trains ~10 steps: short schedule so the pruning code
            c["technique"].update(t_init=0, t_final=0, delta_t=1)   # runs; the real run keeps your values
        sp = os.path.join(WORK, "configs", f"smoke_{c['name']}.json")
        json.dump(c, open(sp, "w"), indent=2)
        smoke_paths.append(sp)
    t0 = time.time()
    gpus = ",".join(str(i) for i in range(max(1, N_GPUS)))
    sp_proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "60", *smoke_paths])
    print(f"\\nsmoke test: {(time.time() - t0) / 60:.1f} min")
    for path in CFG_PATHS:
        c = json.load(open(path))
        sm = os.path.join(WORK, "runs_smoke", c["name"], "summary.json")
        if not os.path.exists(sm):
            print(f"  FAIL {c['name']:22s} -> see {os.path.join(WORK, 'runs_smoke', c['name'] + '.stdout')}"); continue
        s_ = json.load(open(sm))
        tok_s = s_["train_tok_s"]
        c["est_minutes"] = round(1.15 * c["train"]["epochs"] * N_TOKENS_PER_EPOCH[c["name"]] / tok_s / 60 + 15, 1) \\
            if c["name"] in N_TOKENS_PER_EPOCH else None
        json.dump(c, open(path, "w"), indent=2)
        print(f"  OK   {c['name']:22s} peak {s_['peak_mem_GB']} GB of ~15 | trainable {s_['params']['trainable']:,} "
              f"({s_['params']['trainable_pct']}%) | {tok_s} tok/s | est. {c['est_minutes']} min per full run")
    assert sp_proc.returncode == 0, "smoke test failed: fix the runs marked FAIL before launching the full runs"
    for ck in glob.glob(os.path.join(WORK, "runs_smoke", "*", "best_model.pt")):
        os.remove(ck)
    print("SMOKE TEST PASSED -> run the next cell for the full runs")
"""

TOKENS = """
# tokens per training epoch for each queued run (its own chunk length / strategy, full de-duplicated train set)
import numpy as np
_z = np.load(os.path.join(WORK, "cache", "train_ids.npz"))
_lens = np.diff(_z["offsets"])
N_TOKENS_PER_EPOCH = {}
for path in CFG_PATHS:
    c = json.load(open(path))
    _, kind, spec, _ = __import__("backbones").load_phase1(c["phase1_ckpt"], load_weights=False)
    L = c["chunking"]["chunk_len"] or (spec["encoder"]["max_seq_len"] if kind == "encdec" else spec["model"]["max_seq_len"])
    ch = c["chunking"]
    N_TOKENS_PER_EPOCH[c["name"]] = int(sum(e - s for n in _lens for s, e in
                                            cjpe_data.chunk_spans(int(n), L, ch["strategy"], ch["max_chunks"], ch.get("stride"))))
    print(f"{c['name']:22s} chunk_len {L} -> {N_TOKENS_PER_EPOCH[c['name']] / 1e6:.1f}M tokens per epoch")
"""

LONG = """
# Stage 8 — longer training of finished runs (YOUR values, course rule 9). Needs the finished run folders attached
# (account outputs). Each entry copies the finished run's own config (identical technique / chunking / head / KD
# teacher), then applies your changes. mode "continue" = start from its best_model.pt (new folder, original untouched);
# mode "fresh" = start again from the Phase-1 checkpoint. Selection: dev_excl_test, evaluated every eval_every_steps.
# train_chunking (optional): one chunking per epoch for TRAINING, e.g. [{"strategy": "headtail"},
# {"strategy": "overlap", "stride": 640}, {"strategy": "even"}]; dev/test always use the run's own chunking.
# Your request (2026-10-19): fresh runs from the pretrained models, as many epochs as fit in one ~12 h session (both runs
# in parallel, one per T4), original learning rates; three best checkpoints per run (selected on dev_all, dev_in_test,
# dev_excl_test). Epoch counts from the measured speeds (C3 ~118 min, E8 ~65 min per epoch + ~2 min per dev check);
# max_minutes stops training safely before Kaggle's 12 h limit (outputs are still written).
SAVE_BEST_ON = ["dev_all", "dev_in_test", "dev_excl_test"]
LONG_RUNS = {
    "C3_long": {"base": "C3_B2_overlap", "mode": "fresh", "epochs": 5, "lr": 5e-5, "head_lr": 1e-4,
                "warmup_frac": 0.06, "eval_every_steps": 1000, "train_chunking": None, "max_minutes": 630},
    "E8_long": {"base": "E8_KD_B2s12", "mode": "fresh", "epochs": 8, "lr": 5e-5, "head_lr": 1e-4,
                "warmup_frac": 0.06, "eval_every_steps": 1000, "train_chunking": None, "max_minutes": 630},
}
LONG_PATHS = []
for name, o in (LONG_RUNS.items() if ACCOUNT == "long" else []):
    if find_run(name):
        print(f"{name}: already finished ({find_run(name)}), not retrained"); continue
    unset = [k for k, v in o.items() if v is None and k != "train_chunking"]
    if unset:
        print(f"{name}: skipped (fill {unset})"); continue
    src = find_run(o["base"])
    assert src, f"{name}: finished run {o['base']} not found (attach the output that contains runs/{o['base']})"
    c = json.load(open(os.path.join(src, "config.json")))
    model_key = c.get("model_key") or ("B2" if "run_B2" in c["phase1_ckpt"] else None)
    c.update(name=name, phase1_ckpt=PHASE1[model_key], data_dir=DATA_DIR, tokenizer=TOKENIZER,
             out_dir=os.path.join(WORK, "runs"), cache_dir=os.path.join(WORK, "cache"), select_on="dev_excl_test")
    c.pop("smoke", None); c.pop("est_minutes", None)
    if c.get("kd"):                                   # same teacher as the original run (its cached logits are reused)
        t = os.path.basename(c["kd"]["teacher_run"].rstrip("/"))
        c["kd"]["teacher_run"] = find_run(t)
        assert c["kd"]["teacher_run"], f"{name}: KD teacher {t} not found (attach the output with runs/{t})"
    c["init_from_run"] = src if o["mode"] == "continue" else None
    assert o["mode"] in ("continue", "fresh")
    c["train"].update(epochs=o["epochs"], lr=o["lr"], head_lr=o["head_lr"], warmup_frac=o["warmup_frac"],
                      eval_every_steps=o["eval_every_steps"], max_minutes=o.get("max_minutes"))
    c["save_best_on"] = SAVE_BEST_ON
    c["train_chunking"] = o["train_chunking"]
    train_cjpe.check_config(c)
    p = os.path.join(WORK, "configs", f"{name}.json"); json.dump(c, open(p, "w"), indent=2); LONG_PATHS.append(p)
    print(f"{name}: from {src} | {o['mode']} | epochs {o['epochs']} | lr {o['lr']} head_lr {o['head_lr']} | "
          f"warmup {o['warmup_frac']} | eval every {o['eval_every_steps']} steps | train_chunking {o['train_chunking']} | "
          f"max {o.get('max_minutes')} min | best checkpoints on {SAVE_BEST_ON}")
if LONG_PATHS:
    gpus = ",".join(str(i) for i in range(max(1, N_GPUS)))
    STOP_AT = SESSION_START + 11.5 * 3600
    proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "600", "--stop-at", str(STOP_AT),
                           "--default-minutes", "240", "--margin", "15", *LONG_PATHS])
    print("Stage 8 exit code", proc.returncode)
    os.makedirs("/kaggle/working/submissions", exist_ok=True)
    for p in LONG_PATHS:                              # the three submissions per run
        n = json.load(open(p))["name"]
        for z in sorted(glob.glob(os.path.join(WORK, "runs", n, "best_*", "submission_*.zip"))):
            shutil.copy(z, "/kaggle/working/submissions/")
            print("  ", z)
"""

PREDICT = """
# Stage 9 — inference-only re-chunking (ACCOUNT = 'predict'): finished runs predict dev + test again with other
# chunk selections (no training). variant = strategy:max_chunks[:stride]. Outputs: WORK/predict/<run>/<variant>/
# {dev_probs.json, test_probs.json, info.json}; download them for the local threshold / ensemble sweep.
PREDICT_RUNS = ["C3_B2_overlap", "E8_long", "C3_long"]
VARIANTS = ["even:4", "even:8", "headtail:4", "headtail:8", "overlap:4:640", "overlap:8:640"]
if ACCOUNT == "predict":
    env = dict(os.environ, CJPE_DATA_DIR=DATA_DIR, CJPE_TOKENIZER=TOKENIZER, CJPE_CACHE_DIR=os.path.join(WORK, "cache"),
               PYTHONUNBUFFERED="1")
    procs, gpus = [], list(range(max(1, N_GPUS)))
    queue = [(r, find_run(r)) for r in PREDICT_RUNS]
    missing = [r for r, d in queue if not d]
    assert not missing, f"finished runs not found (attach their outputs): {missing}"
    # one process per GPU, runs split round-robin; each process handles its runs one after another
    shards = {g: [q for i, q in enumerate(queue) if i % len(gpus) == g] for g in gpus}
    for g, items in shards.items():
        cmds = " && ".join(f"python predict_cjpe.py {d} {os.path.join(WORK, 'predict', r)} {' '.join(VARIANTS)}" for r, d in items)
        log = open(os.path.join(WORK, f"predict_gpu{g}.log"), "w")
        procs.append(subprocess.Popen(["bash", "-c", cmds], env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=log, stderr=subprocess.STDOUT))
        print(f"GPU {g}: {[r for r, _ in items]}")
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        for g in gpus:
            lines = open(os.path.join(WORK, f"predict_gpu{g}.log")).read().strip().splitlines()
            print(f"GPU {g}: {lines[-1][:220] if lines else '(starting)'}", flush=True)
    print("exit codes:", [p.returncode for p in procs])
    for g in gpus:
        print(open(os.path.join(WORK, f"predict_gpu{g}.log")).read()[-3000:])
    assert all(p.returncode == 0 for p in procs), "a prediction process failed: see the logs above"
"""

LAUNCH = """
# Main runs on BOTH T4s (one run per GPU at a time, shared queue in RUN order). A run is only STARTED if its estimated
# time fits before SESSION_HOURS; runs that do not fit are SKIPPED (train them in the next session: attach this output).
assert N_GPUS >= 1, "no GPU visible: set Accelerator to 'GPU T4 x2'"
gpus = ",".join(str(i) for i in range(N_GPUS))
SESSION_HOURS = 11.5
STOP_AT = SESSION_START + SESSION_HOURS * 3600
print(f"{(STOP_AT - time.time()) / 3600:.2f} h of session left for starting runs")
if CFG_PATHS:
    proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "600", "--stop-at", str(STOP_AT),
                           "--default-minutes", "240", *CFG_PATHS])
    print(f"run_many exit code {proc.returncode} | session time used {(time.time() - SESSION_START) / 3600:.2f} h")
else:
    print("no main runs queued in this session")
"""

KD = """
# E8 knowledge distillation: teacher = the finished run with the highest dev macro-F1 (dev_all; your rule), student =
# B2 cut to 12 of 18 blocks (experiments.KD_STUDENT). The teacher's train-set logits are computed once and cached.
RUN_KD = ACCOUNT in (1, "all")
if ACCOUNT in ("long", "predict"):
    print(f"ACCOUNT = '{ACCOUNT}': Stage 5 skipped")
if RUN_KD:
    cands = {}
    for n in ALL_RUNS:
        d = find_run(n)
        if d:
            cands[n] = (json.load(open(os.path.join(d, "summary.json")))["dev_all"]["macro_f1"], d)
    assert cands, "no finished run to use as teacher"
    t_name = max(cands, key=lambda n: cands[n][0])
    print("teacher candidates (dev_all macro-F1):", {n: v[0] for n, v in cands.items()}, "-> teacher:", t_name)
    kd_cfg = experiments.build_kd(cands[t_name][1], PHASE1, DATA_DIR, TOKENIZER, os.path.join(WORK, "runs"),
                                  os.path.join(WORK, "cache"))
    if find_run(kd_cfg["name"]):
        print(kd_cfg["name"], "already finished:", find_run(kd_cfg["name"]))
    else:
        train_cjpe.check_config(kd_cfg)
        kp = os.path.join(WORK, "configs", f"{kd_cfg['name']}.json")
        json.dump(kd_cfg, open(kp, "w"), indent=2)
        proc = subprocess.run([sys.executable, "run_many.py", "--gpus", "0", "--every", "600", "--stop-at", str(STOP_AT),
                               "--default-minutes", "180", kp])
        print("KD exit code", proc.returncode)
"""

RESULTS = """
# results table (one row per finished run) + all zips in /kaggle/working/submissions
import pandas as pd
rows = []
os.makedirs("/kaggle/working/submissions", exist_ok=True)
names = ALL_RUNS + ["E8_KD_B2s12"]           # runs of both accounts (attach the other account's output to see them)
for n in names:
    d = find_run(n)
    if not d:
        rows.append(dict(name=n, status="not finished")); continue
    s = json.load(open(os.path.join(d, "summary.json")))
    da, dx = s["dev_all"], s["dev_excl_test"]
    rows.append(dict(name=n, status="ok" if s["full_epoch_completed"] else "INCOMPLETE EPOCH", model=s["phase1"].get("tag"),
                     technique=s["technique"]["name"], lr=s["lr"], head_lr=s["head_lr"], kd=s["kd"],
                     trainable=s["params"]["trainable"], trainable_pct=s["params"]["trainable_pct"],
                     chunk_len=s["chunking"]["chunk_len"], strategy=s["chunking"]["strategy"], max_chunks=s["chunking"]["max_chunks"],
                     pool=f"{s['chunk_pool']}/{s['doc_pool']}", epochs=s["epochs_completed"], best_epoch=s["best_epoch"],
                     train_min=round(s["train_seconds"] / 60, 1), test_infer_s=s["test_infer_seconds"], peak_GB=s["peak_mem_GB"],
                     dev_all_F1=da["macro_f1"], dev_excl_test_F1=dx["macro_f1"],
                     P0=da["per_class"]["0"]["precision"], R0=da["per_class"]["0"]["recall"],
                     P1=da["per_class"]["1"]["precision"], R1=da["per_class"]["1"]["recall"],
                     confusion=da["confusion_gold_x_pred"], checkpoint=os.path.join(d, "best_model.pt")))
    z = os.path.join(d, f"submission_{n}.zip")
    if os.path.exists(z):
        shutil.copy(z, "/kaggle/working/submissions/")
df = pd.DataFrame(rows)
display(df)
df.to_csv("/kaggle/working/submissions/summary.csv", index=False)
print(sorted(os.listdir("/kaggle/working/submissions")))
"""

SELECT = """
# Threshold selection (one member) and probability-average ensembles (several), chosen on DEV only
# (select_on dev_all; thresholds 0.05..0.95 step 0.01), then applied unchanged to test. YOUR choice of entries:
#   name -> {"members": [run names], "weights": None or one number per member}
SELECT = {
    "BEST_thr": {"members": None, "weights": None},     # e.g. ["E1_B2_full"]: threshold tuning of the best run
    "ENS_1": {"members": None, "weights": None},        # e.g. 2-3 complementary runs
}
for sel_name, e in SELECT.items():
    if not e["members"]:
        print(f"{sel_name}: skipped (fill members)"); continue
    dirs = [find_run(m) for m in e["members"]]
    assert all(dirs), f"{sel_name}: unfinished members {[m for m, d in zip(e['members'], dirs) if not d]}"
    cmd = [sys.executable, "cjpe_select.py", "--name", sel_name, "--out", os.path.join(WORK, "ensembles"),
           "--data-dir", DATA_DIR, "--members", *dirs]
    if e["weights"]:
        cmd += ["--weights", *map(str, e["weights"])]
    subprocess.run(cmd, check=True)
    shutil.copy(os.path.join(WORK, "ensembles", sel_name, f"submission_{sel_name}.zip"), "/kaggle/working/submissions/")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "capstone_phase4_cjpe.ipynb"))
    args = ap.parse_args()
    cells = [md(INTRO), md("## Stage 0 — session, folders"), code(SETUP),
             md("## Stage 1 — plumbing files (your Phase-1 model code unchanged; declare the rest)")]
    cells += [writefile(f) for f in PLUMBING]
    cells += [md("## Stage 1b — YOUR files: head, techniques, experiment matrix")]
    cells += [writefile(f) for f in YOURS]
    cells += [md("## Stage 2 — data, Phase-1 inputs, token cache"), code(DATA),
              md("## Stage 3 — configuration (experiments.py) and tokens per epoch"), code(CONFIG), code(TOKENS),
              md("## Stage 3b — real-model smoke test (memory, speed, time per run)"), code(SMOKE),
              md("## Stage 4 — main runs (both GPUs)"), code(LAUNCH),
              md("## Stage 5 — knowledge distillation (teacher = best finished run)"), code(KD),
              md("## Stage 6 — results and zips"), code(RESULTS),
              md("## Stage 7 — threshold selection / ensembles (dev only)"), code(SELECT),
              md("## Stage 8 — longer training of finished runs (ACCOUNT = 'long')"), code(LONG),
              md("## Stage 9 — inference-only re-chunking (ACCOUNT = 'predict')"), code(PREDICT)]
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
