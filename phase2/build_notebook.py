#!/usr/bin/env python3
"""Build the Kaggle notebook ~/capstone/capstone_phase2_lner.ipynb from the files in this folder (plumbing).

Every .py file becomes a %%writefile cell, so you can edit your files (heads.py, techniques/*.py) either here and
rebuild, or directly in the notebook on Kaggle.   python build_notebook.py [--out PATH]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working/phase2"

PLUMBING = ["a1_tokenizer.py"] + [f"phase1_models/{f}" for f in (
    "rope.py", "arch1.py", "arch3.py", "attention3.py", "non_causal_self_attention.py", "cross_attention.py",
    "encoder.py", "decoder4.py", "enc_dec.py", "prefix_attention.py", "prefix_lm.py",
    "fastattn_ab.py", "fastattn_cd.py")] + [
    "fetch_data.py", "ner_data.py", "ner_align.py", "ner_metrics.py", "backbones.py", "inject.py",
    "techniques/__init__.py", "techniques/full_ft.py", "train_ner.py", "run_many.py", "check_pipeline.py",
    "ensemble.py", "error_examples.py", "snap_spans.py", "audit_spans.py"]
YOURS = ["heads.py", "techniques/lora.py", "techniques/lorma.py", "techniques/adapters.py",
         "techniques/bitfit.py", "techniques/partial_ft.py", "techniques/kd.py"]


ENSEMBLE = """
# Ensemble finished runs into one extra submission (your choice of members / mode). Run after Stage 4/5.
#   "prob": average the members' per-token tag probabilities (needs their best_model.pt -> same session; linear/mlp heads)
#   "vote": keep spans predicted by >= MIN_VOTES members (needs only their prediction files)
ENSEMBLE_NAME = "B_ensemble"
ENSEMBLE_MEMBERS = None          # FILL: run names, e.g. ["B_full", "B_lr3e5", ...]
ENSEMBLE_MODE = None             # FILL: "prob" or "vote"
MIN_VOTES = None                 # FILL for "vote" (1..len(members)); ignored for "prob"
if not ENSEMBLE_MEMBERS or ENSEMBLE_MODE not in ("prob", "vote"):
    print("ensemble skipped: fill ENSEMBLE_MEMBERS and ENSEMBLE_MODE (and MIN_VOTES for vote), then run this cell")
else:
    member_dirs = [os.path.join(WORK, "runs", m) for m in ENSEMBLE_MEMBERS]
    for d in member_dirs:
        assert os.path.exists(os.path.join(d, "summary.json")), f"{d} is not a finished run"
    cmd = [sys.executable, "ensemble.py", "--name", ENSEMBLE_NAME, "--runs", *member_dirs, "--mode", ENSEMBLE_MODE,
           "--out", os.path.join(WORK, "runs")]
    if ENSEMBLE_MODE == "vote":
        cmd += ["--min-votes", str(MIN_VOTES)]
    subprocess.run(cmd, check=True)
    z = os.path.join(WORK, "runs", ENSEMBLE_NAME, f"submission_{ENSEMBLE_NAME}.zip")
    shutil.copy(z, "/kaggle/working/submissions/")
    print("ensemble zip copied to /kaggle/working/submissions/")
"""

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def writefile(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        body = f.read()
    return code(f"%%writefile {WORK}/{rel}\n" + body)


INTRO = """
# CS781 Capstone — Phase 2: Legal NER (L-NER) on 2×T4

**Kaggle settings:** Accelerator **GPU T4 ×2**, Internet **on**, secret **`HF`** (your Hugging Face token), and
**Add Input →** the dataset/notebook output that holds your Phase-1 checkpoints (`run_A/best.pt`, `run_B2/best.pt`
or `run_B/best.pt`, `run_C/best.pt`, `run_D/best.pt`) and `legal_bpe.json`.

**Your parts (course rule 9):** `heads.py`, `techniques/{lora,lorma,adapters,bitfit,partial_ft,kd}.py` (Stage 1b) and every
`None` in the configuration cell (Stage 3). Everything else is plumbing (declare it): data, char↔token alignment,
windows, training loop, strict macro-F1, prediction.json + zip.

Order: Stage 0 → 1 → 1b → 2 (data + oracle check) → 3 (configs) → 3b (real-model smoke test) → 4 (runs) → 5 (results / zips).
Outputs: `/kaggle/working/phase2/runs/<name>/` (summary.json, val_scores.json, submission_<name>.zip) and all zips
collected in `/kaggle/working/submissions/`.
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
# Phase-2 data from the gated HF dataset (Kaggle secret "HF"), then the Phase-1 inputs
subprocess.run([sys.executable, "fetch_data.py", "--out", os.path.join(WORK, "data")], check=True)
DATA_DIR = os.path.join(WORK, "data", "Phase_2_LNER")

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

ORACLE = """
# oracle check (no model, ~1-2 min): gold tags through the real windows/merge/decode/offset/JSON path.
# Expected: macro-F1 0.9945 with split_punct=True, 0.9866 with False (the char<->token alignment ceiling).
subprocess.run([sys.executable, "check_pipeline.py", "--data", DATA_DIR, "--tokenizer", TOKENIZER], check=True)
"""

CONFIG = """
# =====================================================================================
# YOUR CHOICES (course rule 9). Values below are yours (transcribed from your config).
# =====================================================================================
import copy

BASE = {
    "data_dir": DATA_DIR,
    "tokenizer": TOKENIZER,
    "out_dir": os.path.join(WORK, "runs"),
    "fast_attention": True,
    "backbone": {"prefix": 640, "encdec_mode": None, "truncate_layers": None, "dropout": 0.1},
    "preprocess": {"split_punct": True, "tag_scheme": "BIO"},
    "split": {"mode": "random", "val_frac": 0.15, "seed": 42},
    # default windows for A, B, D (C overrides them below: encoder limit 640)
    "window": {"train_len": 768, "train_stride": 512, "infer_len": 768, "infer_stride": 512, "merge": "center"},
    "head": {"name": "linear", "dropout": 0.1},
    "technique": {"name": "full_ft"},
    "kd": None,
    "train": {
        "epochs": 8, "batch_size": 16, "micro_batch": 2,
        "lr": 2e-5, "head_lr": 1e-4,                      # full fine-tuning baseline
        "weight_decay": 0.01, "warmup_frac": 0.06, "schedule": "cosine", "grad_clip": 1.0,
        "class_weights": "inv_sqrt", "o_weight": None,
        "patience": 2, "max_minutes": 80, "seed": 42,
        "eval_every_epochs": 1, "log_every": 20, "infer_micro_batch": 2,
    },
    "decode": {"repair": "strict"},
}

LORA_TARGETS = ["q_proj", "v_proj"]       # these names exist in A, B, C and D (checked)
C_WINDOW = {"train_len": 512, "train_stride": 384, "infer_len": 512, "infer_stride": 384, "merge": "center"}

# Round-1 experiment definitions (kept as the source of the round-2 configs; only names listed in RUN are run)
EXPERIMENTS = {
    # 1. B: full fine-tuning baseline
    "B_full":   {"model": "B", "technique": {"name": "full_ft"}},
    # 2. B: LoRMA
    "B_lorma":  {"model": "B", "technique": {"name": "lorma", "r": 8, "alpha": 16, "targets": LORA_TARGETS},
                 "train": {"lr": 2e-4}},
    # 3. C (encoder mode): full fine-tuning
    "C_full":   {"model": "C", "backbone": {"encdec_mode": "encoder"}, "window": C_WINDOW,
                 "technique": {"name": "full_ft"}},
    # 4. C (encoder mode): LoRMA
    "C_lorma":  {"model": "C", "backbone": {"encdec_mode": "encoder"}, "window": C_WINDOW,
                 "technique": {"name": "lorma", "r": 8, "alpha": 16, "targets": LORA_TARGETS},
                 "train": {"lr": 2e-4}},
    # 5. A: LoRA
    "A_lora":   {"model": "A", "technique": {"name": "lora", "r": 8, "alpha": 16, "targets": LORA_TARGETS},
                 "train": {"lr": 1e-4}},
    # 6. B: partial fine-tuning (top 2 blocks + final norm) with the CRF head
    "B_peft":   {"model": "B", "technique": {"name": "partial_ft", "k": 2},
                 "head": {"name": "mlp_crf", "hidden": 256, "dropout": 0.1}},
    # 7. D (prefix-LM mask over the first 640 tokens): BitFit (the other one of partial_ft / bitfit)
    "D_peft":   {"model": "D", "backbone": {"prefix": 640}, "technique": {"name": "bitfit"},
                 "train": {"lr": 5e-5}},
}

# ---------------- KD RE-RUN (this session): teacher B_lr5e5 (retrained) -> student truncated B. Nothing else runs. ----
# Teacher: your round-2 B_lr5e5 config (B full fine-tuning, lr 5e-5). It must be retrained here: KD needs its weights.
EXPERIMENTS["B_lr5e5"] = {"model": "B", "technique": {"name": "full_ft"}, "train": {"lr": 5e-5, "head_lr": 1e-4}}

KD_TEMPERATURE = 2.0             # yours (passed to your kd_loss as "temperature")
KD_ALPHA = 0.3                   # yours (passed to your kd_loss as "alpha")
KD_STUDENT_LAYERS = 9            # yours: student keeps the first 9 of B's 18 blocks (86.6M params)
assert KD_STUDENT_LAYERS is not None and 0 < KD_STUDENT_LAYERS < 18, "fill KD_STUDENT_LAYERS (1..17)"

# Student: B truncated to its first KD_STUDENT_LAYERS blocks, full FT, your lr / epochs, patience None (all epochs),
# CE term of your kd_loss class-weighted with the same inv_sqrt weights as the B runs (cfg["weight"]).
EXPERIMENTS["KD_B_to_Bsmall"] = {
    "model": "B",
    "backbone": {"truncate_layers": KD_STUDENT_LAYERS},
    "technique": {"name": "full_ft"},
    "kd": {"teacher_run": os.path.join(WORK, "runs", "B_lr5e5"),
           "temperature": KD_TEMPERATURE, "alpha": KD_ALPHA, "pass_class_weights": True},
    "train": {"lr": 2e-5, "head_lr": 1e-4, "epochs": 8, "patience": None},
}

# teacher first; the student starts automatically when the teacher's summary.json exists
RUN = ["B_lr5e5", "KD_B_to_Bsmall"]
print(f"KD re-run: {RUN} (submit only KD_B_to_Bsmall; B_lr5e5 is the teacher, already on Codabench)")

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d

import train_ner, backbones
CFG_PATHS = []
for name in RUN:
    exp = copy.deepcopy(EXPERIMENTS[name])
    model = exp.pop("model")
    assert model in PHASE1, f"{name}: set 'model' to one of {list(PHASE1)}"
    assert exp["technique"].get("name") is not None, f"{name}: set technique name"
    assert exp["technique"]["name"] != "partial_ft" or exp["technique"].get("k") is not None, \
        f"{name}: partial_ft needs 'k'"
    cfg = deep_update(copy.deepcopy(BASE), exp)
    cfg["name"] = name
    cfg["phase1_ckpt"] = PHASE1[model]
    assert cfg["phase1_ckpt"], f"{name}: no Phase-1 checkpoint found for model {model}"
    train_ner.check_config(cfg)              # required keys + technique options vs YOUR apply() signature
    # window vs the backbone's real max length (read from the checkpoint config, no weights loaded)
    _, kind, spec, _ = backbones.load_phase1(cfg["phase1_ckpt"], load_weights=False)
    if kind == "encdec":
        limit = spec["encoder"]["max_seq_len"] if cfg["backbone"]["encdec_mode"] == "encoder" else \
            min(spec["encoder"]["max_seq_len"], spec["decoder"]["max_seq_len"])
    else:
        limit = spec["model"]["max_seq_len"]
    w = cfg["window"]
    assert max(w["train_len"], w["infer_len"]) <= limit, f"{name}: window > {limit} (max length of model {model})"
    if cfg.get("kd"):
        teacher = os.path.basename(cfg["kd"]["teacher_run"])
        assert teacher in RUN[:RUN.index(name)], f"{name}: teacher {teacher} must come earlier in RUN"
    path = os.path.join(WORK, "configs", f"{name}.json")
    json.dump(cfg, open(path, "w"), indent=2)
    CFG_PATHS.append(path)
    print(f"{name:10s} | model={model} ({kind}) | technique={cfg['technique']['name']:10s} | "
          f"head={cfg['head']['name']:8s} | window={w['train_len']}/{limit} | lr={cfg['train']['lr']:.1e}")
print(f"{len(CFG_PATHS)} configs validated and written")
"""

SMOKE = """
# Real-model smoke test (~5-10 min on 2xT4) BEFORE the full runs: every config in RUN, unchanged (real checkpoint,
# technique, head, window, micro-batch -> real GPU memory), but only 3 optimizer steps on 4 train docs, then
# evaluation on 2 val docs, save -> reload, prediction on 2 test docs and the zip. Outputs go to runs_smoke/ (not
# submittable). If anything crashes or runs out of memory, fix it here instead of an hour into the real run.
RUN_SMOKE = True
if RUN_SMOKE:
    smoke_paths = []
    for path in CFG_PATHS:
        c = json.load(open(path))
        c["smoke"] = {"steps": 3}
        c["out_dir"] = os.path.join(WORK, "runs_smoke")
        c["train"]["max_minutes"] = None
        if c.get("kd"):                                   # KD student: point at the smoke teacher
            c["kd"]["teacher_run"] = os.path.join(WORK, "runs_smoke", os.path.basename(c["kd"]["teacher_run"].rstrip("/")))
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
proc = subprocess.run([sys.executable, "run_many.py", "--gpus", gpus, "--every", "300", *CFG_PATHS])
print(f"run_many exit code {proc.returncode} (non-zero = at least one run failed: see the summary above)")
print(f"session time used: {(time.time() - SESSION_START) / 3600:.2f} h")
"""

RESULTS = """
# results table + all zips in /kaggle/working/submissions
import pandas as pd
rows = []
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for sp in sorted(glob.glob(os.path.join(WORK, "runs", "*", "summary.json"))):
    s = json.load(open(sp))
    rows.append(dict(name=s["name"], phase1=s["phase1"].get("tag") or s["phase1"]["module"],
                     technique=s["technique"]["name"], head=s["head"]["name"], kd=s["kd"],
                     val_macro_f1=s.get("val_macro_f1"), val_macro_p=s.get("val_macro_p"), val_macro_r=s.get("val_macro_r"),
                     trainable=s["params"]["trainable"], trainable_pct=s["params"]["trainable_pct"],
                     total_params=s["params"]["total"], best_epoch=s["best_epoch"],
                     train_min=round(s["train_seconds"] / 60, 1), peak_GB=s["peak_mem_GB"],
                     infer_tok_s=s["test_infer_tok_s"], test_spans=s["test_spans"]))
    z = os.path.join(os.path.dirname(sp), f"submission_{s['name']}.zip")
    if os.path.exists(z):
        shutil.copy(z, "/kaggle/working/submissions/")
df = pd.DataFrame(rows)
display(df)
df.to_csv("/kaggle/working/submissions/summary.csv", index=False)
print(sorted(os.listdir("/kaggle/working/submissions")))
"""

PER_LABEL = """
# per-label strict scores on the held-out docs (counts only; interpretation is yours)
for vp in sorted(glob.glob(os.path.join(WORK, "runs", "*", "val_scores.json"))):
    v = json.load(open(vp))
    print(os.path.basename(os.path.dirname(vp)), "macro-F1", v["macro_f1"])
    display(pd.DataFrame(v["per_label"]).T)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "capstone_phase2_lner.ipynb"))
    args = ap.parse_args()
    cells = [md(INTRO), md("## Stage 0 — session, folders"), code(SETUP),
             md("## Stage 1 — plumbing files (your Phase-1 model code unchanged; declare the rest)")]
    cells += [writefile(f) for f in PLUMBING]
    cells += [md("## Stage 1b — YOUR files: head + fine-tuning techniques (contracts in each docstring)")]
    cells += [writefile(f) for f in YOURS]
    cells += [md("## Stage 2 — data, Phase-1 inputs, oracle check"), code(DATA), code(ORACLE),
              md("## Stage 3 — configuration (yours)"), code(CONFIG),
              md("## Stage 3b — real-model smoke test (few steps per run, ~5-10 min)"), code(SMOKE),
              md("## Stage 4 — runs (both GPUs)"), code(LAUNCH),
              md("## Stage 5 — results and zips"), code(RESULTS), code(PER_LABEL),
              md("## Stage 6 — optional ensemble of finished runs (members and mode are yours)"), code(ENSEMBLE)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):                      # self-check: every code cell must be valid Python
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            compile(src.split("\n", 1)[1] if src.startswith("%%writefile") else src, f"<cell {i}>", "exec")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}: {len(cells)} cells")


if __name__ == "__main__":
    main()
