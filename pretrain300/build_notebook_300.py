#!/usr/bin/env python3
"""Build capstone_B300_aws.ipynb (plumbing): pretrain a ~300M arch3 model from scratch on an AWS GPU instance.
Every .py file in this folder becomes a %%writefile cell.   python build_notebook_300.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["a1_tokenizer.py", "arch3.py", "fastattn.py", "build_data.py", "train_300.py"]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


INTRO = """
# CS781 — pretrain a ~300M "B" model (your arch3: GQA + RoPE decoder) from scratch on AWS (g6e.2xlarge, 1x L40S)

**Before you start (instance):** Deep Learning AMI with PyTorch (GPU), **EBS volume >= 150 GB** (raw corpus ~20 GB +
token parts ~9 GB + train.bin ~9 GB + checkpoints ~3.6 GB each for last.pt), Jupyter on the instance.
Put **`legal_bpe.json`** (your Phase-1 tokenizer, sha256 38f51f14, from `~/capstone/phase1/ab_inputs/`) next to this
notebook, and export your Hugging Face token in the shell that starts Jupyter: `export HF_TOKEN=...` (never paste it here).

**Your parts (course rule 9):** every `None` in the configuration cell: architecture dims (target 250–300M params, the
Phase-1 limit is 300M), and all training hyperparameters (lr schedule, batch, token budget, init).
Everything else is plumbing (declare it): data build, training loop, probe, monitor, export.

**Course rule 7 reminder:** "Phases 2–7 must start from your Phase 1 (or later-phase) model." Confirm with the TA that
this from-scratch model may be used in Phase 3+ before relying on it.

Order: 0 checks → 1 config → 2 code files → 3 param count → 4 data (once, ~1–2 h CPU) → 5 probe (~10 min) →
6 launch in background (survives closing the browser) → 7 monitor → 8 export.
"""

ENV = """
# environment check
import os, sys, json, time, glob, shutil, hashlib, subprocess, math
import torch
WORK = os.path.abspath(os.getcwd())
if WORK not in sys.path:
    sys.path.insert(0, WORK)                  # the code files are written next to this notebook
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "no GPU visible"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} {p.total_memory / 1e9:.0f} GB | bf16 {torch.cuda.is_bf16_supported()}")
print(f"CPUs {os.cpu_count()} | free disk here {shutil.disk_usage(WORK).free / 1e9:.0f} GB")
assert shutil.disk_usage(WORK).free > 60e9, "less than 60 GB free: enlarge the EBS volume"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "numpy", "matplotlib"], check=True)
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
"""

CONFIG = """
# =====================================================================================
# YOUR CHOICES (course rule 9). Fill every None. Phase-1 limit: <= 300M parameters (target 250-300M).
# For reference, B2 was: d_model 768, n_layers 18, n_heads 12, n_kv_heads 4, d_ff 3456, max_seq_len 896, rope,
# norm_first True, attention_scale 0.125 (148.6M params); lr 3e-4 -> 3e-5 cosine, warmup 1000, init_std 0.02.
# =====================================================================================
TOKENIZER = os.path.join(WORK, "legal_bpe.json")
DATA_DIR = os.path.join(WORK, "data")
OUT_DIR = os.path.join(WORK, "run_B3")
TAG = "B3"

# YOUR B3 configuration (your values). 22 layers = 298,273,792 params (checked with your arch3 code; 21-layer fallback not needed).
MODEL = dict(
    vocab_size=32000,                 # fixed: must match legal_bpe.json (the Phase-2+ pipeline depends on it)
    d_model=1024, n_layers=22, n_heads=16, n_kv_heads=4, d_ff=4608,     # head dim 64, GQA 16 -> 4
    max_seq_len=896, dropout=0.1,
    pos_emb_type="rope", norm_first=True, attention_scale=0.125,       # 0.125 = 1/sqrt(64)
)
INIT_STD = 0.02                       # tied embedding / LM head (arch3 ties them)
TRAIN = dict(
    batch_tokens=16_128,              # your 16,384 is not a multiple of 896; 18 x 896 = 16,128 is what B2's "16384" actually ran
    lr=2e-4, min_lr=2e-5, warmup_steps=1000, schedule="cosine",
    weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, grad_clip=1.0,
    max_tokens="all",                 # "all" = one pass over train.bin (the full cleaned corpus); or an integer
    # plumbing
    eval_every_min=30, eval_windows=360, log_every=50, compile=True, fast_attention=True, seed=42,
    max_micro_batch=64,
)
FERTILITY_FILTER = False             # yours: keep ALL documents (multilingual + noisy OCR); True = Phase-1 rule (> 2.0 tok/word dropped)
MAX_DAYS = 7                          # safety stop only (the schedule is driven by max_tokens)
COST_PER_HOUR = None                  # optional: your instance $/h, for the projection in the probe

def _missing(d, prefix):
    return [prefix + k for k, v in d.items() if v is None and k not in ("attention_scale",)]
todo = _missing(MODEL, "MODEL.") + _missing(TRAIN, "TRAIN.")
assert not todo, "fill these first: " + ", ".join(todo)
assert TRAIN["batch_tokens"] % MODEL["max_seq_len"] == 0, "batch_tokens must be a multiple of max_seq_len"
assert os.path.exists(TOKENIZER), f"put legal_bpe.json at {TOKENIZER}"
sha = hashlib.sha256(open(TOKENIZER, "rb").read()).hexdigest()[:8]
print("tokenizer sha256", sha, "(Phase-1 legal BPE: 38f51f14)")
assert sha == "38f51f14", "this is not your Phase-1 tokenizer: the model would be unusable in the Phase-2+ pipeline"

def resolved_train():
    t = dict(TRAIN)
    if t["max_tokens"] == "all":
        man = os.path.join(DATA_DIR, "manifest.json")
        assert os.path.exists(man), "max_tokens='all' needs the data built first (section 4)"
        t["max_tokens"] = int(json.load(open(man))["train_tokens"])
    return t

def run_config(out_dir, probe_minutes=None):
    cfg = {"tag": TAG, "module": "arch3", "model": MODEL, "init_std": INIT_STD,
           "data_dir": DATA_DIR, "out_dir": out_dir, "tokenizer_path": TOKENIZER,
           "deadline_unix": time.time() + MAX_DAYS * 86400, "session_end_unix": time.time() + MAX_DAYS * 86400,
           "schedule_hours": None, "init_from": None,
           "resume_from": os.path.join(out_dir, "last.pt"),          # auto-resume if it exists
           "train": resolved_train(), "gen": {"max_new_tokens": 1, "decoding": {}}}
    if probe_minutes:
        cfg["probe"] = {"train_minutes": probe_minutes}
        cfg["resume_from"] = None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.json")
    json.dump(cfg, open(path, "w"), indent=2)
    return path
print("configuration OK")
"""

PARAMS = """
# parameter count (CPU, no weights needed) + tokens per parameter
import importlib, arch3
importlib.reload(arch3)
m = arch3.DecoderLM(arch3.LMConfig(**MODEL))
n = sum(p.numel() for p in m.parameters())
emb = MODEL["vocab_size"] * MODEL["d_model"]
print(f"parameters: {n:,} ({n / 1e6:.1f}M) | embedding (tied with the LM head) {emb / 1e6:.1f}M | blocks {(n - emb) / 1e6:.1f}M")
_mt = TRAIN["max_tokens"] if TRAIN["max_tokens"] != "all" else (
    json.load(open(os.path.join(DATA_DIR, "manifest.json")))["train_tokens"] if os.path.exists(os.path.join(DATA_DIR, "manifest.json")) else 4.5e9)
print(f"tokens per parameter at max_tokens ({_mt / 1e9:.2f}B): {_mt / n:.1f}   (B2: ~8.9; Chinchilla-optimal ~20)")
assert n <= 300_000_000, "over the 300M Phase-1 limit"
if n < 250_000_000:
    print("note: below your 250-300M target")
del m
"""

LANG = """
# OPTIONAL, before the full build: language / script mix of the corpus (counts only, writes no token files).
# Per source: documents read per dominant script, and which would be KEPT vs DROPPED by the fertility filter
# (non-Supreme-Court docs with > 2.0 tokens/word are dropped; this removes most non-English text, since your tokenizer
# needs ~4 tokens/word for Hindi and ~20-27 for Tamil/Bengali). Supreme Court docs are exempt from the filter.
# Whether to change the filter (build_data.py --max-fertility) is your decision.
subprocess.run([sys.executable, "build_data.py", "--tokenizer", TOKENIZER, "--out", os.path.join(WORK, "lang_stats"),
                "--dry-stats"], check=True)
rep = json.load(open(os.path.join(WORK, "lang_stats", "script_stats.json")))
for src, st in rep.items():
    scripts = sorted({k.split("_", 2)[2] for k in st if k.startswith("script_read_")})
    print(f"\\n{src}: {st.get('docs_read', 0):,} docs read")
    for sc in scripts:
        kept_t, drop_t = st.get(f"tokens_kept_{sc}", 0), st.get(f"tokens_dropped_fertility_{sc}", 0)
        print(f"  {sc:12s} docs read {st.get(f'script_read_{sc}', 0):>9,} | kept {st.get(f'script_kept_{sc}', 0):>9,} "
              f"({kept_t / 1e6:,.1f}M tok) | dropped by fertility {st.get(f'script_dropped_fertility_{sc}', 0):>8,} ({drop_t / 1e6:,.1f}M tok)")
"""

DATA = """
# build train.bin / val.bin from the WHOLE corpus with your tokenizer (once; resumable; ~1-2 h on 8 vCPUs, mostly encoding)
if os.path.exists(os.path.join(DATA_DIR, "train.bin")) and os.path.exists(os.path.join(DATA_DIR, "manifest.json")):
    print("data already built:"); print(open(os.path.join(DATA_DIR, "manifest.json")).read()[:3000])
else:
    assert os.environ.get("HF_TOKEN"), "export HF_TOKEN before starting Jupyter (gated dataset)"
    subprocess.run([sys.executable, "build_data.py", "--tokenizer", TOKENIZER, "--out", DATA_DIR]
                   + ([] if FERTILITY_FILTER else ["--no-fertility-filter"]), check=True)
man = json.load(open(os.path.join(DATA_DIR, "manifest.json")))
print(f"fertility filter: {man.get('fertility_filter')} | min words {man.get('min_words')} | exact-duplicate removal on")
print(f"train tokens {man['train_tokens']:,} | val tokens {man['val_tokens']:,} | max_tokens / train tokens = "
      f"{resolved_train()['max_tokens'] / man['train_tokens']:.2f} passes")
"""

PROBE = """
# runtime probe: the real training script on the real data for PROBE_MIN minutes (separate folder), then a projection
PROBE_MIN = 8
pdir = os.path.join(WORK, "probe")
shutil.rmtree(pdir, ignore_errors=True)
subprocess.run([sys.executable, "train_300.py", run_config(pdir, probe_minutes=PROBE_MIN)], check=True)
recs = [json.loads(l) for l in open(os.path.join(pdir, "log.jsonl"))]
b = [r for r in recs if r["msg"] == "batch"][0]
tr = [r for r in recs if r["msg"] == "train"]
tok_s = sorted(r["tok_per_s"] for r in tr[len(tr) // 3:])[len(tr[len(tr) // 3:]) // 2]     # median after warm-up
mem = max(r["mem_GB"] for r in tr)
MT = resolved_train()["max_tokens"]
hours = MT / tok_s / 3600
print(f"micro-batch {b['micro_batch']} x accum {b['grad_accum']} | {tok_s:,} tokens/s | peak memory {mem} GB")
print(f"projected training time for {MT / 1e9:.2f}B tokens: {hours:.1f} h"
      + (f" | ~${hours * COST_PER_HOUR:,.0f} at ${COST_PER_HOUR}/h" if COST_PER_HOUR else ""))
compiled = [r for r in recs if "compile" in r["msg"]]
print("torch.compile:", compiled[0]["msg"] if compiled else "not used")
"""

LAUNCH = """
# launch the real run in the BACKGROUND (keeps training if the browser/notebook disconnects; auto-resumes from last.pt)
def is_running(pid_file):
    # True only for a live training process (finished children are reaped, zombies count as stopped)
    if not os.path.exists(pid_file):
        return False
    pid = int(open(pid_file).read().strip())
    try:
        os.waitpid(pid, os.WNOHANG)          # reap it if it is our finished child
    except ChildProcessError:
        pass
    try:
        return open(f"/proc/{pid}/stat").read().split()[2] != "Z"
    except FileNotFoundError:
        return False

cfg_path = run_config(OUT_DIR)
log_path = os.path.join(OUT_DIR, "train.stdout")
pid_path = os.path.join(OUT_DIR, "pid")
if is_running(pid_path):
    print("already running, pid", open(pid_path).read().strip())
else:
    proc = subprocess.Popen(["nohup", sys.executable, "train_300.py", cfg_path], stdout=open(log_path, "a"),
                            stderr=subprocess.STDOUT, cwd=WORK, start_new_session=True)
    open(pid_path, "w").write(str(proc.pid))
    print("started pid", proc.pid, "| log", log_path)
print("stop gracefully (evaluates + saves last.pt):  kill -TERM $(cat run_B3/pid)   |  resume: run this cell again")
"""

MONITOR = """
# monitor (run any time): last log lines + loss curves
import matplotlib.pyplot as plt
lj = os.path.join(OUT_DIR, "log.jsonl")
recs = [json.loads(l) for l in open(lj)] if os.path.exists(lj) else []
for r in recs[-6:]:
    print({k: v for k, v in r.items() if k != "t"})
tr = [r for r in recs if r["msg"] == "train"]; ev = [r for r in recs if r["msg"] in ("eval", "final_eval")]
if tr:
    MT = resolved_train()["max_tokens"]
    done = tr[-1]["tokens_M"] * 1e6 / MT
    print(f"progress {100 * done:.1f}% | {tr[-1]['tok_per_s']:,} tok/s | peak mem {tr[-1]['mem_GB']} GB | "
          f"ETA {(1 - done) * MT / max(1, tr[-1]['tok_per_s']) / 3600:.1f} h")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([r["tokens_M"] for r in tr], [r["loss"] for r in tr]); ax[0].set_title("train loss"); ax[0].set_xlabel("tokens (M)")
    if ev:
        ax[1].plot([r["tokens"] / 1e6 for r in ev], [r["val_loss"] for r in ev], "o-"); ax[1].set_title("val loss")
    plt.show()
pid_path = os.path.join(OUT_DIR, "pid")
print("training process running:", is_running(pid_path))
"""

EXPORT = """
# export: best.pt + tokenizer + manifest into export/ (upload this folder as a PRIVATE Kaggle dataset for Phase 3+)
exp = os.path.join(WORK, "export", "run_B3")
os.makedirs(exp, exist_ok=True)
best = os.path.join(OUT_DIR, "best.pt")
assert os.path.exists(best), "no best.pt yet (first one is written at the first evaluation)"
ck = torch.load(best, map_location="cpu", weights_only=False)
import arch3
m = arch3.DecoderLM(arch3.LMConfig(**ck["config"]["model"])); m.load_state_dict(ck["model"])
n = sum(p.numel() for p in m.parameters())
assert ck["config"]["module"] == "arch3" and ck["config"]["model"]["vocab_size"] == 32000
shutil.copy(best, os.path.join(exp, "best.pt"))
shutil.copy(TOKENIZER, os.path.join(WORK, "export", "legal_bpe.json"))
shutil.copy(os.path.join(DATA_DIR, "manifest.json"), os.path.join(WORK, "export", "pretrain_manifest.json"))
print(f"exported {n / 1e6:.1f}M params | step {ck['step']} | tokens {ck['tokens'] / 1e9:.2f}B | best val {ck['best_val']:.4f}")
print("folder:", os.path.join(WORK, "export"), "-> e.g. `kaggle datasets create -p export` (private), or download it")
print("In the Phase-2+ notebooks it is found like the others (pattern run_B3/best.pt); add it to PHASE1 as a new key.")
"""


def main():
    cells = [md(INTRO), md("## 0 — environment"), code(ENV), md("## 1 — configuration (yours)"), code(CONFIG),
             md("## 2 — code files (plumbing; your arch3 unchanged)")]
    for f in FILES:
        cells.append(code(f"%%writefile {f}\n" + open(os.path.join(HERE, f), encoding="utf-8").read()))
    cells += [md("## 3 — parameter count"), code(PARAMS), md("## 4a — optional: language / script mix (counts only)"), code(LANG),
              md("## 4 — data (whole corpus, once)"), code(DATA),
              md("## 5 — runtime probe (~10 min)"), code(PROBE), md("## 6 — launch training (background)"), code(LAUNCH),
              md("## 7 — monitor"), code(MONITOR), md("## 8 — export for Phase 3+"), code(EXPORT)]
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            compile(src.split("\n", 1)[1] if src.startswith("%%writefile") else src, f"<cell {i}>", "exec")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(HERE, "capstone_B300_aws.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {out}: {len(cells)} cells")


if __name__ == "__main__":
    main()
