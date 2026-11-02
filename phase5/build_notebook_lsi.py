#!/usr/bin/env python3
"""Build the Kaggle notebook ~/capstone/capstone_phase5_lsi.ipynb from the files in this folder (plumbing).
Every .py file becomes a %%writefile cell (your heads.py / techniques/*.py / experiments.py are editable there too).
    python build_notebook_lsi.py [--out PATH]"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working/phase5"

PLUMBING = ["a1_tokenizer.py"] + [f"phase1_models/{f}" for f in (
    "rope.py", "arch1.py", "arch3.py", "attention3.py", "non_causal_self_attention.py", "cross_attention.py",
    "encoder.py", "decoder4.py", "enc_dec.py", "prefix_attention.py", "prefix_lm.py",
    "fastattn_ab.py", "fastattn_cd.py")] + [
    "fetch_data.py", "lsi_data.py", "backbones.py", "inject.py", "techniques/__init__.py", "techniques/full_ft.py",
    "train_lsi.py", "run_many.py", "lsi_select.py", "predict_lsi.py"]
YOURS = ["heads.py", "techniques/partial_ft.py", "techniques/bitfit.py", "techniques/lorma.py", "techniques/adalora.py",
         "techniques/qlora.py", "techniques/prefix_tuning.py", "techniques/prompt_tuning.py",
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
# CS781 Capstone — Phase 5: Legal Statute Identification (LSI, multi-label) on 2×T4

**Kaggle settings:** GPU T4 ×2, Internet **on**, secret **`HF`** or **`HF_TOKEN`** (ticked for this notebook). **Add Input:**
the 4 private model datasets (A, B2, C+D, B3 = `ctxext/best_newctx.pt`); from session 2 on also earlier outputs of this
notebook (either account), so finished runs are reused.

**Task plumbing (adapted from Phase 4):** one sigmoid output per statute (label space = all statute ids in train + dev),
BCE loss (optional `pos_weight`), multi-label macro-F1 (labels present in gold), micro-F1, subset accuracy; predictions =
statutes with P >= 0.5 (tuned global / per-statute thresholds: Stage 7, `lsi_select.py`). KD: your `kd_loss`, applied
per (document, statute) as a 2-class problem.

**Your parts (course rule 9):** `heads.py`, `techniques/*.py`, and every value in `experiments.py` (matrix, SPLIT, KD).

**Modes (Stage 3 `ACCOUNT`):** `1` / `2` = this account's share of the matrix (+ KD on 1) · `"all"` · `"long"` (Stage 8) ·
`"predict"` (Stage 9, inference-only re-chunking). Outputs: `/kaggle/working/phase5/runs/<name>/`, zips in
`/kaggle/working/submissions/`.
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
DATA_DIR = os.path.join(WORK, "data", "Phase_5_LSI")

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

import lsi_data
from a1_tokenizer import ByteBPE
_tok = ByteBPE.from_json(TOKENIZER)
_tr, DATA_REPORT = lsi_data.dedup_train(lsi_data.load_split(DATA_DIR, "train"))
_dv, _te = lsi_data.load_split(DATA_DIR, "dev"), lsi_data.load_split(DATA_DIR, "test")
_held = {lsi_data.norm_hash(d["text"]) for d in _dv + _te}                 # same leakage guard as train_lsi.load_data
_n0 = len(_tr); _tr = [d for d in _tr if lsi_data.norm_hash(d["text"]) not in _held]
DATA_REPORT["train_removed_overlapping_dev_or_test"] = _n0 - len(_tr); DATA_REPORT["train_docs_used"] = len(_tr)
DATA_REPORT["dev_docs_in_test"] = lsi_data.mark_dev_excl_test(_dv, _te)
t0 = time.time()
for s, docs in (("train", _tr), ("dev", _dv), ("test", _te)):
    lsi_data.encode_docs(_tok, docs, os.path.join(WORK, "cache", f"{s}_ids.npz"))
print(DATA_REPORT, f"| token cache ready in {time.time() - t0:.0f} s")
N_TRAIN = len(_tr)
del _tr, _dv, _te, _tok
"""

CONFIG = """
# Experiment matrix = experiments.py (YOUR values). RUN = what this session trains, in priority order.
import importlib, experiments, train_lsi, techniques
importlib.reload(experiments)
BATCH_DOCS = experiments.BASE["train"]["batch_docs"]
STEPS_PER_EPOCH = -(-N_TRAIN // BATCH_DOCS)
EXPERIMENTS = experiments.build(PHASE1, DATA_DIR, TOKENIZER, os.path.join(WORK, "runs"), os.path.join(WORK, "cache"),
                                STEPS_PER_EPOCH)
_kd_name = getattr(experiments, "KD_NAME", None)       # KD in the queue: waits for its teacher (run_many) when both are here
if _kd_name:
    EXPERIMENTS[_kd_name] = experiments.build_kd(os.path.join(WORK, "runs", experiments.KD_TEACHER), PHASE1, DATA_DIR,
                                                 TOKENIZER, os.path.join(WORK, "runs"), os.path.join(WORK, "cache"))
# Two Kaggle accounts in parallel: set ACCOUNT = 1 on the first account and ACCOUNT = 2 on the second.
# Account 1 also runs KD (Stage 5); its teacher is the best finished run it can see (its own, plus account 2's runs if
# that output is attached). ACCOUNT = "all" runs everything in one account (two sessions).
ACCOUNT = None
SPLIT = experiments.SPLIT                       # run names per account: edit experiments.py
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
    if cfg.get("kd"):                                   # teacher trained in this session (queued earlier) or finished earlier
        _t = experiments.KD_TEACHER
        if _t not in RUN[:RUN.index(name)]:
            _td = find_run(_t)
            assert _td, f"{name}: KD teacher {_t} is neither queued before it nor a finished attached run"
            cfg["kd"]["teacher_run"] = _td
    if not_implemented(cfg["technique"]["name"]):
        print(f"SKIP {name}: techniques/{cfg['technique']['name']}.py is not implemented yet"); continue
    train_lsi.check_config(cfg)
    path = os.path.join(WORK, "configs", f"{name}.json")
    json.dump(cfg, open(path, "w"), indent=2)
    CFG_PATHS.append(path)
    print(f"{name:22s} | model={cfg['model_key']:3s} | technique={cfg['technique']['name']:10s} | lr={cfg['train']['lr']} "
          f"head_lr={cfg['train']['head_lr']} | chunks={cfg['chunking']['strategy']} x{cfg['chunking']['max_chunks']} "
          f"(len {cfg['chunking']['chunk_len'] or 'model limit'}) | pool={cfg['chunk_pool']}/{cfg['doc_pool']} | "
          f"pos_weight={cfg['train']['pos_weight']}")
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
        if c.get("kd") and os.path.basename(c["kd"]["teacher_run"].rstrip("/")) in RUN:   # teacher's smoke run
            c["kd"]["teacher_run"] = os.path.join(WORK, "runs_smoke", os.path.basename(c["kd"]["teacher_run"].rstrip("/")))
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
        c["est_minutes"] = round(1.15 * c["train"]["epochs"] * N_TOKENS_PER_EPOCH[c["name"]] / tok_s / 60 + 45, 1) \\
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
                                            lsi_data.chunk_spans(int(n), L, ch["strategy"], ch["max_chunks"], ch.get("stride"))))
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
SAVE_BEST_ON = ["dev_all", "dev_excl_test"]
LONG_RUNS = {      # name -> {"base": finished run, "mode": "continue" | "fresh", "epochs", "lr", "head_lr", "warmup_frac",
                   #          "eval_every_steps", "train_chunking", "max_minutes"}   (your values)
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
    train_lsi.check_config(c)
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
PREDICT_RUNS = []                  # finished run names (your choice)
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
        cmds = " && ".join(f"python predict_lsi.py {d} {os.path.join(WORK, 'predict', r)} {' '.join(VARIANTS)}" for r, d in items)
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
# KD: teacher = the finished run with the highest dev macro-F1 (dev_all), student = experiments.KD_STUDENT.
# The teacher's train-set logits (one per statute) are computed once and cached in the teacher's folder.
RUN_KD = ACCOUNT in (1, "all") and not find_run(getattr(experiments, "KD_NAME", "")) and getattr(experiments, "KD_NAME", None) not in RUN
if ACCOUNT in ("long", "predict"):
    print(f"ACCOUNT = '{ACCOUNT}': Stage 5 skipped")
if RUN_KD:
    t_name = getattr(experiments, "KD_TEACHER", None)          # your fixed teacher (experiments.KD_TEACHER)
    t_dir = find_run(t_name) if t_name else None
    assert t_dir, f"KD teacher {t_name} has not finished (or is not attached): train it first"
    cands = {t_name: (None, t_dir)}
    print("KD teacher:", t_name, t_dir)
    kd_cfg = experiments.build_kd(cands[t_name][1], PHASE1, DATA_DIR, TOKENIZER, os.path.join(WORK, "runs"),
                                  os.path.join(WORK, "cache"))
    if find_run(kd_cfg["name"]):
        print(kd_cfg["name"], "already finished:", find_run(kd_cfg["name"]))
    else:
        train_lsi.check_config(kd_cfg)
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
names = ALL_RUNS + [experiments.KD_NAME]     # runs of both accounts (attach the other account's output to see them)
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
                     macro_P=da["macro_p"], macro_R=da["macro_r"], micro_F1=da["micro_f1"], subset_acc=da["subset_accuracy"],
                     macro_F1_all_labels=da["macro_f1_all"], labels_in_gold=da["labels_in_gold"],
                     mean_pred_labels=da["mean_pred_labels"], mean_gold_labels=da["mean_gold_labels"],
                     checkpoint=os.path.join(d, "best_model.pt")))
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
#   name -> {"members": [run names], "weights": None | list, "mode": "global" | "per_label", "min_support": int,
#            "min_one": bool (empty prediction -> most probable statute), "select_on": "dev_excl_test" | "dev_all"}
SELECT = {
    "BEST_thr": {"members": None, "weights": None, "mode": None, "min_support": None, "min_one": None, "select_on": None},
}
for sel_name, e in SELECT.items():
    if not e["members"]:
        print(f"{sel_name}: skipped (fill members)"); continue
    dirs = [find_run(m) for m in e["members"]]
    assert all(dirs), f"{sel_name}: unfinished members {[m for m, d in zip(e['members'], dirs) if not d]}"
    cmd = [sys.executable, "lsi_select.py", "--name", sel_name, "--out", os.path.join(WORK, "ensembles"),
           "--data-dir", DATA_DIR, "--members", *dirs, "--mode", e["mode"], "--min-support", str(e["min_support"]),
           "--select-on", e["select_on"]] + (["--min-one"] if e["min_one"] else [])
    if e["weights"]:
        cmd += ["--weights", *map(str, e["weights"])]
    subprocess.run(cmd, check=True)
    shutil.copy(os.path.join(WORK, "ensembles", sel_name, f"submission_{sel_name}.zip"), "/kaggle/working/submissions/")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "capstone_phase5_lsi.ipynb"))
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
