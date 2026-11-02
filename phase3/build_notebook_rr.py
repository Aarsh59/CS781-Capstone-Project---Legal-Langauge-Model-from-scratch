#!/usr/bin/env python3
"""Build the Kaggle notebook ~/capstone/capstone_phase3_rr.ipynb from the files in this folder (plumbing).
Every .py file becomes a %%writefile cell (your heads.py / techniques/*.py are editable in the notebook too).
    python build_notebook_rr.py [--out PATH]"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working/phase3"

PLUMBING = ["a1_tokenizer.py"] + [f"phase1_models/{f}" for f in (
    "rope.py", "arch1.py", "arch3.py", "attention3.py", "non_causal_self_attention.py", "cross_attention.py",
    "encoder.py", "decoder4.py", "enc_dec.py", "prefix_attention.py", "prefix_lm.py",
    "fastattn_ab.py", "fastattn_cd.py")] + [
    "fetch_data.py", "rr_data.py", "rr_metrics.py", "backbones.py", "inject.py",
    "techniques/__init__.py", "techniques/full_ft.py", "train_rr.py", "run_many.py", "check_pipeline_rr.py",
    "rr_ensemble.py"]
YOURS = ["heads.py", "techniques/lora.py", "techniques/lorma.py", "techniques/kd.py",
         "techniques/rslora.py", "techniques/ia3.py", "techniques/prompt_tuning.py", "techniques/prefix_tuning.py",
         "techniques/adalora.py", "techniques/qlora.py", "techniques/partial_ft.py", "techniques/bitfit.py"]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def writefile(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return code(f"%%writefile {WORK}/{rel}\n" + f.read())


INTRO = """
# CS781 Capstone — Phase 3: Rhetorical Role prediction (RR) on 2×T4

**Kaggle settings:** Accelerator **GPU T4 ×2**, Internet **on**, secret **`HF`**, **Add Input →** your Phase-1 outputs
(`run_A`, `run_B2`, `run_C`, `run_D` best.pt + `legal_bpe.json`), same as Phase 2.

**Your parts (course rule 9):** `heads.py`, `techniques/*.py` (LoRA, LoRMA, rsLoRA, AdaLoRA, QLoRA, IA³, prompt
tuning, prefix tuning, KD) and every `None` in Stage 3 (configs) and Stage 6 (ensembles).
Everything else is plumbing (declare it): data, sentence windows + context, pooling, training loop, macro-F1,
prediction.json + zip, ensembles.

**Phase-3 technique list (§9.2.1):** LoRA, QLoRA, AdaLoRA, rsLoRA, **LoRMA (compulsory)**, Prefix Tuning, Prompt
Tuning, IA³, **KD (mandatory)** — at least 6 distinct, all four Phase-1 models, both CL and IT.

Order: 0 → 1 → 1b → 2 (data + oracle) → 3 (configs) → 3b (smoke) → 4 (runs, 2 GPUs) → 5 (results) → 6 (ensemble).
Outputs: `/kaggle/working/phase3/runs/<name>/` and all zips in `/kaggle/working/submissions/`.
"""

SETUP = f"""
# session clock + folders + GPUs
import os, sys, json, glob, time, hashlib, subprocess, shutil
SESSION_START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-crf"], check=True)   # heads.py imports torchcrf
WORK = "{WORK}"
for d in ("phase1_models", "techniques", "configs", "runs", "data"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
import torch
N_GPUS = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])
"""
DATA = """
# Phase-3 data from the gated HF dataset (Kaggle secret "HF"), then the Phase-1 inputs
subprocess.run([sys.executable, "fetch_data.py", "--out", os.path.join(WORK, "data")], check=True)
DATA_DIR = os.path.join(WORK, "data", "Phase_3_RR")

def find_all(pattern):
    return sorted(glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True))

def find_one(pattern):
    hits = find_all(pattern)
    if len(hits) > 1:
        print(f"  note: {len(hits)} matches for {pattern}, using the first: {hits}")
    return hits[0] if hits else None

def sha8(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]

# every attached tokenizer must be the same file (the models were trained in different notebooks)
TOKS = find_all("legal_bpe.json")
assert TOKS, "legal_bpe.json not found under /kaggle/input (Add Input: your Phase-1 notebook outputs)"
tok_hashes = {t: sha8(t) for t in TOKS}
for t, h in tok_hashes.items():
    print(f"tokenizer {h}  {t}")
assert len(set(tok_hashes.values())) == 1, ("the attached Phase-1 outputs contain DIFFERENT legal_bpe.json files: "
                                            "models trained with different tokenizers cannot share one pipeline")
TOKENIZER = TOKS[0]
print("tokenizer sha256:", tok_hashes[TOKENIZER], "(Phase-1 legal BPE was 38f51f14)")

PHASE1 = {                                     # first match under /kaggle/input; or set explicit paths here
    "A": find_one("run_A/best.pt"),
    "B": find_one("run_B2/best.pt") or find_one("run_B/best.pt"),
    "C": find_one("run_C/best.pt"),
    "D": find_one("run_D/best.pt"),
}
for k, v in PHASE1.items():
    if v is None:
        print(f"Phase-1 {k}: NOT FOUND")
        continue
    ck = torch.load(v, map_location="cpu", weights_only=False)
    c = ck["config"]
    tok_used = c.get("tokenizer_path") or "?"
    print(f"Phase-1 {k}: {v}\\n    tag={c.get('tag')} module={c['module']} step={ck.get('step')} "
          f"best_val={ck.get('best_val')} | trained with tokenizer at {tok_used}")
    del ck
missing = [k for k, v in PHASE1.items() if v is None]
if missing:
    print("WARNING: no checkpoint found for", missing, "-> runs using them will fail (Add Input: that notebook's output)")
"""
SMOKE = """
# Real-model smoke test (~5-10 min on 2xT4) BEFORE the full runs: every config in RUN, unchanged (real checkpoint,
# technique, head, window, micro-batch -> real GPU memory), but only 3 optimizer steps on 4 train docs, then
# evaluation on 1 dev doc per domain, save -> reload, prediction on 1 test doc per domain and the zip. Outputs go to runs_smoke/ (not
# submittable). If anything crashes or runs out of memory, fix it here instead of an hour into the real run.
RUN_SMOKE = True
if RUN_SMOKE:
    smoke_paths = []
    for path in CFG_PATHS:
        c = json.load(open(path))
        c["smoke"] = {"steps": 3}
        c["out_dir"] = os.path.join(WORK, "runs_smoke")
        c["train"]["max_minutes"] = None
        if c.get("kd") and os.path.basename(c["kd"]["teacher_run"].rstrip("/")) in RUN:   # teacher trained here:
            c["kd"]["teacher_run"] = os.path.join(WORK, "runs_smoke", os.path.basename(c["kd"]["teacher_run"].rstrip("/")))
        # (a finished teacher from an attached earlier output is used as is)
        sp = os.path.join(WORK, "configs", f"smoke_{c['name']}.json")
        json.dump(c, open(sp, "w"), indent=2)
        smoke_paths.append(sp)
    t0 = time.time()
    gpus = ",".join(str(i) for i in range(max(1, N_GPUS)))
    sp_proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "60", *smoke_paths])
    print(f"\\nsmoke test: {(time.time() - t0) / 60:.1f} min")
    for path in CFG_PATHS:
        name = json.load(open(path))["name"]
        sm = os.path.join(WORK, "runs_smoke", name, "summary.json")
        if os.path.exists(sm):
            s_ = json.load(open(sm))
            print(f"  OK   {name:10s} peak GPU memory {s_['peak_mem_GB']} GB of ~15 | trainable {s_['params']['trainable']:,} "
                  f"({s_['params']['trainable_pct']}%) | zip written")
        else:
            print(f"  FAIL {name:10s} -> see {os.path.join(WORK, 'runs_smoke', name + '.stdout')}")
    assert sp_proc.returncode == 0, "smoke test failed: fix the runs marked FAIL before launching the full runs"
    # free disk: smoke checkpoints are full state dicts (~0.6-0.8 GB each); Kaggle's /kaggle/working limit is 20 GB
    freed = 0
    for ck in glob.glob(os.path.join(WORK, "runs_smoke", "*", "best_model.pt")):
        freed += os.path.getsize(ck)
        os.remove(ck)
    print(f"removed smoke checkpoints: {freed / 1e9:.1f} GB freed (logs and summaries kept)")
    print("SMOKE TEST PASSED -> run the next cell for the full runs")
"""
LAUNCH = """
# Runs every config in RUN on BOTH T4s in parallel: one training process per GPU at a time, two fine-tunings at once.
# Shared queue: whichever GPU frees up first takes the next runnable config (in RUN order). A KD run is skipped until its
# teacher has finished (summary.json), so it never blocks a GPU. Status every 5 min; logs: runs/<name>.stdout,
# runs/<name>/log.jsonl. Running `nvidia-smi` in another cell while this runs should show one python process per GPU.
assert N_GPUS >= 1, "no GPU visible: set Accelerator to 'GPU T4 x2'"
if N_GPUS < 2:
    print("WARNING: only", N_GPUS, "GPU visible -> runs will be sequential. Set Accelerator to 'GPU T4 x2'.")
gpus = ",".join(str(i) for i in range(N_GPUS))
print("launching on GPUs", gpus)
SESSION_HOURS = 11.5        # Kaggle kills a session at 12 h: runs that cannot finish before this are SKIPPED, not started
STOP_AT = SESSION_START + SESSION_HOURS * 3600
print(f"runs may start until they fit before {SESSION_HOURS} h of session time "
      f"({(STOP_AT - time.time()) / 3600:.2f} h left now)")
proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "300", "--stop-at", str(STOP_AT), *CFG_PATHS])
print(f"run_many exit code {proc.returncode} (non-zero = at least one run failed: see the summary above)")
print(f"session time used: {(time.time() - SESSION_START) / 3600:.2f} h")
"""


ORACLE = """
# oracle check (no model, ~1-2 min): gold labels through the real windows -> prediction -> JSON path for 4 context
# settings. Expected: dev macro-F1 1.0 and "counts match True" for every setting.
subprocess.run([sys.executable, "check_pipeline_rr.py", "--data", DATA_DIR, "--tokenizer", TOKENIZER], check=True)
"""

CONFIG = """
# =====================================================================================
# YOUR CHOICES (course rule 9) — values transcribed from the user's config (20-run campaign).
# Field meanings: docstrings of train_rr.py / rr_data.py.
# Window limits: A/B/D max_seq_len 896, C encoder 640, C decoder mode 256 (window_len + prompt tokens <= limit).
# =====================================================================================
import copy

BASE = {
    "data_dir": DATA_DIR,
    "tokenizer": TOKENIZER,
    "out_dir": os.path.join(WORK, "runs"),
    "fast_attention": True,

    "backbone": {
        "prefix": None,
        "encdec_mode": None,
        "truncate_layers": None,
        "dropout": 0.0,
    },

    "train_domains": ["CL", "IT"],

    "context": {
        "window_len": 768,
        "max_sent_tokens": 128,
        "max_targets": 4,
        "target_tokens": 128,
        "left_tokens": 64,
        "right_tokens": 64,
        "sep_token": True,
    },

    "pooling": "mean",

    "head": {
        "name": "mlp",
        "hidden": 256,
        "dropout": 0.1,
    },

    "technique": {"name": "lora"},
    "kd": None,

    "train": {
        "epochs": 8,
        "batch_size": 16,
        "micro_batch": 4,
        "lr": 1e-4,
        "head_lr": 3e-4,
        "weight_decay": 0.01,
        "warmup_frac": 0.06,
        "schedule": "cosine",
        "grad_clip": 1.0,
        "class_weights": "inv_sqrt",
        "none_weight": None,
        "patience": None,         # campaign 3: no early stopping -> all epochs; the best epoch is still kept
        "max_minutes": 40,
        "seed": 42,
        "log_every": 20,
        "infer_micro_batch": 8,
    },
}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


LORA_OPTS = {
    "r": 8,
    "alpha": 16,
    "targets": ["attn", "ffn"],
}

TECH_OPTS = {
    "lora": LORA_OPTS,
    "lorma": LORA_OPTS,
    "rslora": LORA_OPTS,

    "qlora": {
        **LORA_OPTS,
        "block_size": 64,
    },

    "adalora": {
        **LORA_OPTS,
        "target_rank": 2,
        "t_init": 100,
        "t_final": 100,
        "delta_t": 10,
        "beta1": 0.85,
        "beta2": 0.85,
        "orth_reg": 0.01,
    },

    "ia3": {},

    "prompt_tuning": {
        "n_tokens": 8,
        "init": "random",
    },

    "prefix_tuning": {
        "n_prefix": 8,
    },
}


def tech(name, **opts):
    return {
        "name": name,
        **copy.deepcopy(TECH_OPTS[name]),
        **opts,
    }


def train(**opts):
    return {"train": opts}


def exp(model, technique, **overrides):
    return {
        "model": model,
        "technique": technique,
        **overrides,
    }


# ============================================================
# CAMPAIGN 3 (2026-10-16): the 27 runs of campaigns 1 + 2, unchanged except BASE patience None (all epochs,
# best epoch kept). Names get the suffix _allep so outputs / zips / Codabench entries are distinguishable.
# KD teachers are the re-trained B_lora_allep of THIS session (no earlier output needs to be attached).
# Ensembles: Stage 6 (left empty; decided after these runs).
# ============================================================
SUFFIX = "_allep"

# ---------------- campaign 1 (20 runs) ----------------
C1 = {
    # --------------------------------------------------------
    # 1. B2 BASELINES: ALL EIGHT REQUIRED PEFT TECHNIQUES
    # --------------------------------------------------------

    "B_lorma": exp(
        "B", tech("lorma", r=8, alpha=16),
        **train(lr=2e-4),
    ),

    "B_lora": exp(
        "B", tech("lora", r=8, alpha=16),
        **train(lr=1e-4),
    ),

    "B_rslora": exp(
        "B", tech("rslora", r=8, alpha=16),
        **train(lr=1e-4),
    ),

    "B_adalora": exp(
        "B", tech(
            "adalora",
            r=8, alpha=16, target_rank=2,
            t_init=100, t_final=100, delta_t=10,
            beta1=0.85, beta2=0.85, orth_reg=0.01,
        ),
        **train(lr=1e-4),
    ),

    "B_qlora": exp(
        "B", tech("qlora", r=8, alpha=16, block_size=64),
        **train(lr=1e-4),
    ),

    "B_ia3": exp(
        "B", tech("ia3"),
        **train(lr=5e-4),
    ),

    "B_prompt_tuning": exp(
        "B", tech("prompt_tuning", n_tokens=8, init="random"),
        **train(lr=1e-3),
    ),

    "B_prefix_tuning": exp(
        "B", tech("prefix_tuning", n_prefix=8),
        **train(lr=5e-4),
    ),

    # --------------------------------------------------------
    # 2. REQUIRED ARCHITECTURE COVERAGE
    # --------------------------------------------------------

    "A_run": exp(
        "A", tech("lora", r=8, alpha=16),
        **train(lr=1e-4),
    ),

    "C_run": exp(
        "C", tech("lora", r=8, alpha=16),
        backbone={"encdec_mode": "encoder"},
        context={"window_len": 512},
        **train(lr=1e-4),
    ),

    "D_run": exp(
        "D", tech("lora", r=8, alpha=16),
        backbone={"prefix": 640},
        **train(lr=1e-4),
    ),

    # KD teacher: B_lora. Must remain after B_lora in RUN.
    "KD": exp(
        "B",
        tech("lora", r=8, alpha=16),
        backbone={"truncate_layers": 12},
        kd={
            "teacher_run": os.path.join(WORK, "runs", "B_lora"),
            "temperature": 2.0,
            "alpha": 0.5,
        },
        **train(lr=1e-4, head_lr=3e-4),
    ),

    # --------------------------------------------------------
    # 3. TARGETED B2 ABLATIONS
    # --------------------------------------------------------

    # Lower adapter learning rate
    "B_lora_lr5e5": exp(
        "B", tech("lora", r=8, alpha=16),
        **train(lr=5e-5),
    ),

    # Higher adapter learning rate
    "B_lora_lr2e4": exp(
        "B", tech("lora", r=8, alpha=16),
        **train(lr=2e-4),
    ),

    # More LoRA capacity
    "B_lora_r16": exp(
        "B", tech("lora", r=16, alpha=32),
        **train(lr=1e-4),
    ),

    # More rsLoRA capacity
    "B_rslora_r16": exp(
        "B", tech("rslora", r=16, alpha=32),
        **train(lr=1e-4),
    ),

    # Retain more AdaLoRA components
    "B_adalora_r4": exp(
        "B", tech(
            "adalora",
            r=8, alpha=16, target_rank=4,
            t_init=100, t_final=100, delta_t=10,
            beta1=0.85, beta2=0.85, orth_reg=0.01,
        ),
        **train(lr=1e-4),
    ),

    # More virtual prompt tokens
    "B_prompt_p16": exp(
        "B", tech("prompt_tuning", n_tokens=16, init="random"),
        **train(lr=1e-3),
    ),

    # Stronger rare-class weighting
    "B_lora_inv_freq": exp(
        "B", tech("lora", r=8, alpha=16),
        **train(class_weights="inv_freq"),
    ),

    # Change actual context budgets, not just window_len.
    # Compare against BASE: left/right = 64/64.
    "B_lora_ctx_local": exp(
        "B", tech("lora", r=8, alpha=16),
        context={
            "window_len": 768,
            "target_tokens": 128,
            "left_tokens": 0,
            "right_tokens": 0,
        },
        **train(lr=1e-4),
    ),
}


# Required runs first, then eight targeted B2 experiments.
C1_RUN = [
    "B_lorma",
    "B_lora",
    "B_rslora",
    "B_adalora",
    "B_qlora",
    "B_ia3",
    "B_prompt_tuning",
    "B_prefix_tuning",

    "A_run",
    "C_run",
    "D_run",
    "KD",

    "B_lora_lr5e5",
    "B_lora_lr2e4",
    "B_lora_r16",
    "B_rslora_r16",
    "B_adalora_r4",
    "B_prompt_p16",
    "B_lora_inv_freq",
    "B_lora_ctx_local",
]

assert len(C1_RUN) == 20 and len(set(C1_RUN)) == 20

# ---------------- campaign 2 (7 runs) ----------------
C2 = {
    # 7-10: New B2 training runs
    "B_rslora_r16_lr5e5": exp(
        "B", tech("rslora", r=16, alpha=32, targets=["attn", "ffn"]),
        **train(lr=5e-5, head_lr=3e-4, seed=42),
    ),
    "B_rslora_r16_lr2e4": exp(
        "B", tech("rslora", r=16, alpha=32, targets=["attn", "ffn"]),
        **train(lr=2e-4, head_lr=3e-4, seed=42),
    ),
    "B_lora_seed123": exp(
        "B", tech("lora", r=8, alpha=16, targets=["attn", "ffn"]),
        **train(lr=1e-4, head_lr=3e-4, seed=123),
    ),
    "KD_seed123": exp(
        "B", tech("lora", r=8, alpha=16, targets=["attn", "ffn"]),
        backbone={"truncate_layers": 12},     # as in the campaign-1 KD run (BASE dropout 0.0); only the seed differs
        kd={
            "teacher_run": "B_lora",          # campaign-1 run (resolved below to the attached output)
            "temperature": 2.0,
            "alpha": 0.5,
        },
        **train(lr=1e-4, head_lr=3e-4, seed=123),
    ),
    # 11: CRF head
    "B_lora_crf": exp(
        "B", tech("lora", r=8, alpha=16, targets=["attn", "ffn"]),
        head={"name": "mlp_crf"},
        **train(lr=1e-4, head_lr=3e-4, class_weights="inv_sqrt", seed=42),
    ),
    # 12: Larger actual context budget
    "B_lora_ctx_wide": exp(
        "B", tech("lora", r=8, alpha=16, targets=["attn", "ffn"]),
        context={
            "window_len": 896,
            "max_sent_tokens": 128,
            "max_targets": 4,
            "target_tokens": 128,
            "left_tokens": 128,
            "right_tokens": 128,
            "sep_token": True,
        },
        **train(lr=1e-4, head_lr=3e-4, seed=42),
    ),
    # 13: No class weighting
    "B_lora_unweighted": exp(
        "B", tech("lora", r=8, alpha=16, targets=["attn", "ffn"]),
        **train(lr=1e-4, head_lr=3e-4, class_weights=None, seed=42),
    ),
}

C2_RUN = [
    "B_rslora_r16_lr5e5",
    "B_rslora_r16_lr2e4",
    "B_lora_seed123",
    "KD_seed123",
    "B_lora_crf",
    "B_lora_ctx_wide",
    "B_lora_unweighted",
]
assert len(C2_RUN) == 7 and len(set(C2_RUN)) == 7

EXPERIMENTS, RUN = {}, []
for _n in C1_RUN + C2_RUN:
    _e = copy.deepcopy(C1[_n] if _n in C1 else C2[_n])
    if _e.get("kd"):
        _e["kd"]["teacher_run"] = os.path.join(WORK, "runs", os.path.basename(str(_e["kd"]["teacher_run"]).rstrip("/")) + SUFFIX)
    EXPERIMENTS[_n + SUFFIX] = _e
    RUN.append(_n + SUFFIX)
assert len(RUN) == 27 and len(set(RUN)) == 27

def find_prev_run(name):
    # a finished campaign-1 run (best_model.pt + summary.json) in WORK/runs or an attached /kaggle/input output
    for d in [os.path.join(WORK, "runs", name)] + sorted(glob.glob(f"/kaggle/input/**/runs/{name}", recursive=True)):
        if all(os.path.exists(os.path.join(d, f)) for f in ("best_model.pt", "summary.json", "config.json", "labels.json")):
            return d
    return None

# KD teachers: a run earlier in RUN (trained in this session) or a finished campaign-1 run
for _n in RUN:
    _kd = EXPERIMENTS[_n].get("kd")
    if _kd:
        _t = os.path.basename(str(_kd["teacher_run"]).rstrip("/"))
        if _t in RUN[:RUN.index(_n)]:
            _kd["teacher_run"] = os.path.join(WORK, "runs", _t)
        else:
            _d = find_prev_run(_t)
            assert _d, (f"{_n}: KD teacher {_t} not found: attach the campaign-1 output (Add Input -> previous version "
                        f"of this notebook) so that /kaggle/input/**/runs/{_t}/best_model.pt exists")
            _kd["teacher_run"] = _d
        print(f"{_n}: KD teacher -> {_kd['teacher_run']}")



import train_rr, backbones
for _n in RUN:                                   # a technique chosen for A/C/D/KD gets its option keys automatically
    _t = EXPERIMENTS[_n]["technique"]
    if _t.get("name") in TECH_OPTS:
        EXPERIMENTS[_n]["technique"] = {**tech(_t["name"]), **_t}
_ph = [n for n in RUN if "<" in str(EXPERIMENTS[n].get("kd", {}).get("teacher_run", ""))]
assert not _ph, f"{_ph}: set kd.teacher_run to a real run name"
CFG_PATHS = []
for name in RUN:
    exp = copy.deepcopy(EXPERIMENTS[name])
    model = exp.pop("model")
    assert model in PHASE1, f"{name}: set 'model' to one of {list(PHASE1)}"
    cfg = deep_update(copy.deepcopy(BASE), exp)
    cfg["name"] = name
    cfg["phase1_ckpt"] = PHASE1[model]
    assert cfg["phase1_ckpt"], f"{name}: no Phase-1 checkpoint found for model {model}"
    train_rr.check_config(cfg)                 # required keys + technique options vs YOUR apply() signature
    _, kind, spec, _ = backbones.load_phase1(cfg["phase1_ckpt"], load_weights=False)
    if kind == "encdec":
        limit = spec["encoder"]["max_seq_len"] if cfg["backbone"]["encdec_mode"] == "encoder" else \\
            min(spec["encoder"]["max_seq_len"], spec["decoder"]["max_seq_len"])
    else:
        limit = spec["model"]["max_seq_len"]
    prompt_tokens = cfg["technique"].get("n_tokens", 0) if cfg["technique"]["name"] == "prompt_tuning" else 0
    assert cfg["context"]["window_len"] + prompt_tokens <= limit, \
        f"{name}: window_len + prompt tokens > {limit} (max length of model {model})"
    if cfg.get("kd"):
        teacher = os.path.basename(cfg["kd"]["teacher_run"].rstrip("/"))
        assert teacher in RUN[:RUN.index(name)] or os.path.exists(os.path.join(cfg["kd"]["teacher_run"], "summary.json")), \
            f"{name}: teacher {teacher} must come earlier in RUN or be a finished run"
    path = os.path.join(WORK, "configs", f"{name}.json")
    json.dump(cfg, open(path, "w"), indent=2)
    CFG_PATHS.append(path)
    print(f"{name:12s} | model={model} ({kind}) | technique={cfg['technique']['name']:13s} | head={cfg['head']['name']} | "
          f"pool={cfg['pooling']} | window={cfg['context']['window_len']}/{limit} | ctx L/R={cfg['context']['left_tokens']}/{cfg['context']['right_tokens']}")
PHASE3_LIST = {"lora", "qlora", "adalora", "rslora", "lorma", "prefix_tuning", "prompt_tuning", "ia3"}   # + KD (§9.2.1)
_cfgs = [json.load(open(p)) for p in CFG_PATHS]
_listed = {c["technique"]["name"] for c in _cfgs if not c.get("kd")} & PHASE3_LIST
_off = sorted({c["name"] for c in _cfgs if not c.get("kd") and c["technique"]["name"] not in PHASE3_LIST})
_kd = any(c.get("kd") for c in _cfgs)
print(f"{len(CFG_PATHS)} configs | Phase-3-listed techniques: {sorted(_listed)} ({len(_listed)}) + KD: {_kd} "
      f"-> {len(_listed) + int(_kd)} distinct (need >= 6, LoRMA compulsory: {'lorma' in _listed})")
if _off:
    print("WARNING: runs using a technique NOT on the Phase-3 list (do not count towards the 6):", _off)
"""

RESULTS = """
# results table + all zips in /kaggle/working/submissions
import pandas as pd
rows = []
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for sp in sorted(glob.glob(os.path.join(WORK, "runs", "*", "summary.json"))):
    s = json.load(open(sp))
    if "dev_macro_f1" not in s or "phase1" not in s:
        continue
    rows.append(dict(name=s["name"], phase1=s["phase1"].get("tag") or s["phase1"]["module"], technique=s["technique"]["name"],
                     head=s["head"]["name"], pooling=s["pooling"], kd=s["kd"], dev_macro_f1=s["dev_macro_f1"],
                     dev_CL=s["dev_CL_macro_f1"], dev_IT=s["dev_IT_macro_f1"], dev_P=s["dev_macro_p"], dev_R=s["dev_macro_r"],
                     trainable=s["params"]["trainable"], trainable_pct=s["params"]["trainable_pct"], total_params=s["params"]["total"],
                     best_epoch=s["best_epoch"], train_min=round(s["train_seconds"] / 60, 1), peak_GB=s["peak_mem_GB"]))
    z = os.path.join(os.path.dirname(sp), f"submission_{s['name']}.zip")
    if os.path.exists(z):
        shutil.copy(z, "/kaggle/working/submissions/")
df = pd.DataFrame(rows)
display(df)
df.to_csv("/kaggle/working/submissions/summary.csv", index=False)
print(sorted(os.listdir("/kaggle/working/submissions")))
"""

PER_LABEL = """
# per-label dev scores (all / CL / IT) and the most frequent confusions (counts only; interpretation is yours)
for vp in sorted(glob.glob(os.path.join(WORK, "runs", "*", "dev_scores.json"))):
    v = json.load(open(vp))
    print(os.path.basename(os.path.dirname(vp)), "| macro-F1 all", v["all"]["macro_f1"], "| CL", v.get("CL", {}).get("macro_f1"),
          "| IT", v.get("IT", {}).get("macro_f1"))
    display(pd.DataFrame(v["all"]["per_label"]).T)
    print("top confusions (gold->pred):", list(v["all"]["confusion"].items())[:8])
"""

ENSEMBLE = """
# Sentence-level weighted majority-vote ensembles of finished runs (members / weights are your choice, decided after
# looking at the dev scores of the 20 runs). One entry per ensemble submission:
#   name -> {"members": [run names], "weights": None or one number per member}
# Works in the SAME session after Stage 4, or in a LATER session: attach this notebook's output as input, run the
# setup + data cells (Stages 0-2, no GPU needed), then this cell. Runs are looked up in WORK/runs first, then in any
# attached /kaggle/input/**/runs/<name>/ folder.
# Campaign 3: fill after the 27 _allep runs finish (members = run names, e.g. "B_lorma_allep"); skipped while None.
ENSEMBLES = {
    "ENS_new": {"members": None, "weights": None},
}

def find_run(name):
    for d in [os.path.join(WORK, "runs", name)] + sorted(glob.glob(f"/kaggle/input/**/runs/{name}", recursive=True)):
        if os.path.exists(os.path.join(d, "prediction.json")) and os.path.exists(os.path.join(d, "dev_predictions.json")):
            return d
    return None

os.makedirs("/kaggle/working/submissions", exist_ok=True)
for ens_name, e in ENSEMBLES.items():
    if not e["members"]:
        print(f"{ens_name}: skipped (fill members)")
        continue
    dirs = {m: find_run(m) for m in e["members"]}
    missing = [m for m, d in dirs.items() if d is None]
    assert not missing, f"{ens_name}: no finished run found for {missing} (in WORK/runs or /kaggle/input/**/runs)"
    print(f"{ens_name}: members", {m: d for m, d in dirs.items()})
    cmd = [sys.executable, "rr_ensemble.py", "--name", ens_name, "--out", os.path.join(WORK, "ensembles"),
           "--members", *dirs.values()]
    if e["weights"]:
        assert len(e["weights"]) == len(e["members"]), f"{ens_name}: one weight per member"
        cmd += ["--weights", *map(str, e["weights"])]
    subprocess.run(cmd, check=True)
    shutil.copy(os.path.join(WORK, "ensembles", ens_name, f"submission_{ens_name}.zip"), "/kaggle/working/submissions/")
    print(f"{ens_name}: dev macro-F1", json.load(open(os.path.join(WORK, "ensembles", ens_name, "summary.json"))).get("dev_macro_f1"),
          "| zip copied to /kaggle/working/submissions/")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "capstone_phase3_rr.ipynb"))
    args = ap.parse_args()
    cells = [md(INTRO), md("## Stage 0 — session, folders"), code(SETUP),
             md("## Stage 1 — plumbing files (your Phase-1 model code unchanged; declare the rest)")]
    cells += [writefile(f) for f in PLUMBING]
    cells += [md("## Stage 1b — YOUR files: head + techniques (contracts and instructions in each docstring)")]
    cells += [writefile(f) for f in YOURS]
    cells += [md("## Stage 2 — data, Phase-1 inputs, oracle check"), code(DATA), code(ORACLE),
              md("## Stage 3 — configuration (yours)"), code(CONFIG),
              md("## Stage 3b — real-model smoke test (few steps per run)"), code(SMOKE),
              md("## Stage 4 — runs (both GPUs)"), code(LAUNCH),
              md("## Stage 5 — results and zips"), code(RESULTS), code(PER_LABEL),
              md("## Stage 6 — optional voting ensemble"), code(ENSEMBLE)]
    for i, c in enumerate(cells):                      # self-check: every code cell must be valid Python
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
