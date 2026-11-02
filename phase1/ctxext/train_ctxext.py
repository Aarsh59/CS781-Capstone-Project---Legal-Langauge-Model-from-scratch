#!/usr/bin/env python3
"""Continued pretraining of the Phase-1 B2 model with an extended context (plumbing; declare it).

  torchrun --nproc_per_node=2 train_ctxext.py config.json        (Kaggle 2xT4)
  python train_ctxext.py config.json                              (1 GPU / CPU smoke test)
  python train_ctxext.py config.json --probe 30                   (30 optimiser steps: speed, memory, ETA, then exit)

What it does
  * builds B2's architecture (your arch3.py) with model.max_seq_len = cfg["max_seq_len"] (e.g. 1024 instead of 896) and
    loads B2's weights. The RoPE tables and causal mask are non-persistent buffers, so the saved weights do not depend on
    the sequence length; they are rebuilt here for the new length using cfg["rope_extension"] (your choice):
      "extrapolate": positions 0..L-1 as trained (positions >= 896 were never seen in pretraining)
      "interpolate": positions scaled by old/new length, so 0..L-1 map into the trained range 0..895 (position interpolation)
      "ntk":         RoPE base enlarged by (new/old)^(d_head/(d_head-2)) ("NTK-aware" scaling)
  * trains on fixed, non-overlapping windows of length L+1 from one or more uint16 token files, each window used at most
    once, in an order fixed by the seed (nr_sampler.py: allocation, permutation, per-step / per-rank positions). The order
    and the cursor are stored in last.pt, so a resumed run neither repeats nor skips windows.
  * learning-rate schedule driven by TOKENS (not wall-clock), so it is identical however the run is split into sessions:
    linear warm-up over warmup_steps, then cosine (or linear) decay to min_lr at max_tokens.
  * full checkpoint last.pt (model + optimiser + scaler + step + tokens + RNG) every ckpt_every_min minutes, at the
    session limit (session_hours) and on SIGTERM; best.pt (weights only, same format as Phase-1 best.pt, so the Phase 2/3
    pipelines can load it) at every evaluation that improves the validation loss at the new length.
  * resume: the newest last.pt (most tokens) found in out_dir or under any of cfg["resume_search"] globs
    (on Kaggle: the previous session's output attached as an input, from any account).
  * evaluation on every set in cfg["eval_sets"] ({name: uint16 .bin}): loss over windows of the new length L, split into
    positions < old length and >= old length, and loss over windows of the old length (896). Two selections are kept:
      best_newctx.pt  lowest mean (over eval sets) loss at the new length
      best_oldctx.pt  lowest mean loss at the old length (retention of B2's original-context ability)
    both in the Phase-1 best.pt format (loadable by phase3/backbones.py). The step-0 evaluation is B2 itself (baseline).
  * training mixture: cfg["train_files"] = [{"path": ..., "weight": w | None}]; the max_tokens // L windows of the run are
    split over the files in proportion to w (None for all = proportional to the windows each file has), capped by what a
    file has. The token budget is never exceeded: whole windows only, and a final partial step if the count needs one.
  * session guard: world size, batch_tokens, tokens/step, max_tokens, max_seq_len and rope method are stored in last.pt;
    a resume with any of them different stops with an error (that would silently change the token schedule).
  * RNG: torch CPU and per-rank CUDA generator states (dropout) are saved and restored.
"""
import argparse
import glob
import importlib
import json
import math
import os
import signal
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastattn  # noqa: E402
import nr_sampler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--probe", type=int, default=0, help="run N optimiser steps, report speed/memory/ETA, exit without saving")
args = ap.parse_args()
cfg = json.load(open(args.config))
REQUIRED = {"": ["init_from", "train_files", "eval_sets", "out_dir", "max_seq_len", "rope_extension"],
            "train": ["max_tokens", "batch_tokens", "lr", "min_lr", "warmup_steps", "weight_decay", "betas", "grad_clip",
                      "schedule", "seed", "eval_every_min", "ckpt_every_min", "session_hours", "eval_windows"]}
missing = [f"{s + '.' if s else ''}{k}" for s, ks in REQUIRED.items() for k in ks if (cfg.get(s, {}) if s else cfg).get(k) is None]
assert not missing, "fill these config values first (your choices): " + ", ".join(missing)
assert cfg["rope_extension"] in ("extrapolate", "interpolate", "ntk")
TR, OUT = cfg["train"], cfg["out_dir"]
os.makedirs(OUT, exist_ok=True)

# ---------------- distributed / device
WORLD, RANK, LOCAL = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
DIST, CUDA, MASTER = WORLD > 1, torch.cuda.is_available(), int(os.environ.get("RANK", 0)) == 0
if CUDA:
    torch.cuda.set_device(LOCAL)
device = torch.device(f"cuda:{LOCAL}" if CUDA else "cpu")
if DIST:
    dist.init_process_group("nccl" if CUDA else "gloo", timeout=timedelta(minutes=15))
# bf16 only with native bf16 tensor cores (compute capability >= 8.0). is_bf16_supported() is also True on a T4, where
# bf16 is emulated and several times slower; the T4 uses fp16 + GradScaler, like the Phase-1 B / B2 trainer.
AMP = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if CUDA else torch.float32
USE_SCALER = CUDA and AMP == torch.float16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
SESSION_T0 = time.time()


def log(msg, **kv):
    if not MASTER:
        return
    rec = {"t": round(time.time(), 1), "msg": msg, **kv}
    print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, "log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------- model: B2 architecture at the new length + B2 weights
src = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
scfg = src["config"]
OLD_LEN = scfg["model"]["max_seq_len"]
L = cfg["max_seq_len"]
mdict = dict(scfg["model"], max_seq_len=L)
if cfg.get("dropout") is not None:
    mdict["dropout"] = cfg["dropout"]
arch = importlib.import_module(scfg["module"])
torch.manual_seed(TR["seed"])
raw = arch.DecoderLM(arch.LMConfig(**mdict))


def clean(sd):
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}


def set_rope(model):
    """rebuild every attention module's RoPE tables for length L with the chosen extension method."""
    for blk in model.blocks:
        a = blk.attn
        if not getattr(a, "use_rope", False):
            continue
        d = a.d_head
        base, scale = 10000.0, 1.0
        if cfg["rope_extension"] == "interpolate":
            scale = OLD_LEN / L
        elif cfg["rope_extension"] == "ntk":
            base = 10000.0 * (L / OLD_LEN) ** (d / (d - 2))
        inv = 1.0 / (base ** (torch.arange(0, d // 2, dtype=torch.float32) * 2 / d))
        pos = torch.arange(L, dtype=torch.float32) * scale
        ang = torch.outer(pos, inv)
        ang = torch.cat([ang, ang], -1)                          # split-half convention, as in your rope.py
        a.rope_cos = ang.cos().to(a.rope_cos.device)
        a.rope_sin = ang.sin().to(a.rope_sin.device)


missing_keys, unexpected = raw.load_state_dict(clean(src["model"]), strict=False)
assert not unexpected and not [k for k in missing_keys if "rope" not in k and "mask" not in k], (missing_keys, unexpected)
set_rope(raw)
n_params = sum(p.numel() for p in raw.parameters())
fastattn.patch_model(raw)
raw.to(device)
del src

# ---------------- data
trains = [np.memmap(f["path"], dtype=np.uint16, mode="r") for f in cfg["train_files"]]
flens = [len(m) for m in trains]
assert all(n > L + 1 for n in flens), "a train file is shorter than one window"
wts = [f.get("weight") for f in cfg["train_files"]]
assert all(w is None for w in wts) or all(w is not None and w > 0 for w in wts), "give a weight for every train file, or none"
weights = [(n - 1) // L for n in flens] if wts[0] is None else wts
ORDER = nr_sampler.build_order(flens, weights, TR["max_tokens"], L, TR["seed"])   # identical on every rank / session
N_WIN = len(ORDER["src"])
DIGEST = nr_sampler.order_digest(ORDER)
evals = {k: np.memmap(v, dtype=np.uint16, mode="r") for k, v in cfg["eval_sets"].items()}
assert all(len(m) > L + 1 for m in evals.values()), "an eval set is shorter than one window"
V = mdict["vocab_size"]
assert max(int(m[:2_000_000].max()) for m in trains) < V, "token id >= vocab_size: data / tokenizer mismatch"


def windows(pairs):
    """(file, window id) pairs -> input / target tensors; window w = tokens [w*L, w*L + L + 1)."""
    rows = [trains[f][w * L:w * L + L + 1] for f, w in pairs]
    assert all(len(r) == L + 1 for r in rows)
    t = torch.from_numpy(np.stack(rows).astype(np.int64))
    return t[:, :-1].to(device, non_blocking=True), t[:, 1:].to(device, non_blocking=True)


def dummy_batch(n):
    """first n windows of file 0: memory search, and the zero-weight filler of an empty micro-batch in a final partial step."""
    return windows([(0, i % ORDER["n_avail"][0]) for i in range(n)])


def loss_of(model, x, y, reduce=True):
    with torch.autocast("cuda", dtype=AMP, enabled=CUDA):
        logits = model(x)
    l = F.cross_entropy(logits.float().view(-1, V), y.reshape(-1), reduction="mean" if reduce else "none")
    return l


# ---------------- batch geometry: global batch_tokens split over GPUs, largest micro-batch that fits
seqs = max(WORLD, TR["batch_tokens"] // L)
seqs -= seqs % WORLD
per_rank = seqs // WORLD
mb = TR.get("micro_batch")
if mb is None:
    mb = 1
    for cand in sorted([d for d in range(1, per_rank + 1) if per_rank % d == 0], reverse=True):
        if not CUDA:
            mb = 1; break
        try:
            torch.cuda.empty_cache()
            x, y = dummy_batch(cand)
            loss_of(raw, x, y).backward(); raw.zero_grad(set_to_none=True)
            if torch.cuda.max_memory_allocated() < 0.80 * torch.cuda.get_device_properties(device).total_memory:
                mb = cand; break
        except torch.cuda.OutOfMemoryError:
            raw.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if DIST:
        t = torch.tensor([mb], device=device); dist.all_reduce(t, op=dist.ReduceOp.MIN); mb = int(t.item())
assert per_rank % mb == 0
ACC = per_rank // mb
TOK_PER_STEP = seqs * L                                       # a full step; the final step may be partial
TOTAL_STEPS = math.ceil(N_WIN / seqs)
assert N_WIN * L <= TR["max_tokens"]
ddp = DDP(raw, device_ids=[LOCAL] if CUDA else None, broadcast_buffers=False, gradient_as_bucket_view=True) if DIST else None
base = ddp if DIST else raw
model = base
COMPILED = False
if TR.get("compile", False) and CUDA:                        # as in the Phase-1 refine trainer; eager fallback
    t_c = time.time()
    try:
        model = torch.compile(base)
        x, y = dummy_batch(mb)
        loss_of(model, x, y).backward(); raw.zero_grad(set_to_none=True)
        COMPILED = True
        log("torch.compile ok", seconds=round(time.time() - t_c, 1))
    except Exception as e:
        model = base; raw.zero_grad(set_to_none=True)
        log("torch.compile failed, using eager", error=repr(e)[:300])

decay = [p for p in raw.parameters() if p.dim() >= 2]
no_decay = [p for p in raw.parameters() if p.dim() < 2]
groups = [{"params": decay, "weight_decay": TR["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
try:
    opt = torch.optim.AdamW(groups, lr=TR["lr"], betas=tuple(TR["betas"]), fused=CUDA)
except (RuntimeError, TypeError):
    opt = torch.optim.AdamW(groups, lr=TR["lr"], betas=tuple(TR["betas"]))
scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)

# ---------------- resume
step, tokens, history = 0, 0, []
best = {"newctx": float("inf"), "oldctx": float("inf")}
GUARD = {"world_size": WORLD, "batch_tokens": TR["batch_tokens"], "tokens_per_step": TOK_PER_STEP, "max_tokens": TR["max_tokens"],
         "max_seq_len": L, "rope_extension": cfg["rope_extension"], "warmup_steps": TR["warmup_steps"], "lr": TR["lr"],
         "min_lr": TR["min_lr"], "schedule": TR["schedule"], "sampler": "no_repeat", "sampler_order_sha256": DIGEST}
cands = [os.path.join(OUT, "last.pt")] + [p for g in cfg.get("resume_search", []) for p in glob.glob(g, recursive=True)]
found = []
for p in cands:
    if os.path.exists(p):
        try:
            meta = torch.load(p, map_location="cpu", weights_only=False)
            if meta.get("kind") == "ctxext_last" and meta["max_seq_len"] == L:
                found.append((meta["tokens"], p, meta))
        except Exception as e:
            log("skipping unreadable checkpoint", path=p, error=repr(e)[:200])
if found:
    tokens, path, ck = max(found, key=lambda x: x[0])
    diff = {k: (ck.get("guard", {}).get(k), v) for k, v in GUARD.items() if ck.get("guard", {}).get(k) != v}
    assert not diff, f"resume would change the run's schedule/geometry (checkpoint value, now): {diff}"
    raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    step, best, history = ck["step"], ck["best"], ck.get("history", [])
    sp = ck["sampler"]
    assert sp["seed"] == TR["seed"] and np.array_equal(sp["src"], ORDER["src"]) and np.array_equal(sp["wid"], ORDER["wid"]), \
        "saved window order differs from the rebuilt one (data files, weights, seed or budget changed)"
    assert sp["cursor"] == min(step * seqs, N_WIN) and tokens == sp["cursor"] * L, (sp["cursor"], step, tokens)
    cpu_states = ck["rng"]["cpu"]
    torch.set_rng_state(cpu_states[RANK] if isinstance(cpu_states, list) else cpu_states)
    if CUDA and ck["rng"].get("cuda") and RANK < len(ck["rng"]["cuda"]):
        torch.cuda.set_rng_state(ck["rng"]["cuda"][RANK])
    log("resumed", path=path, step=step, tokens_B=round(tokens / 1e9, 4), windows_consumed=sp["cursor"],
        windows_remaining=N_WIN - sp["cursor"])
    del ck
log("setup", params_M=round(n_params / 1e6, 2), old_len=OLD_LEN, new_len=L, rope_extension=cfg["rope_extension"], gpus=WORLD,
    amp=str(AMP).split(".")[-1], compiled=COMPILED, seqs_per_step=seqs, micro_batch=mb, grad_accum=ACC, tokens_per_step=TOK_PER_STEP,
    total_steps=TOTAL_STEPS, start_step=step, target_tokens_B=round(TR["max_tokens"] / 1e9, 3))
log("sampler (no repeat)", files=[os.path.basename(f["path"]) for f in cfg["train_files"]], weights=list(map(float, weights)),
    windows_available=ORDER["n_avail"].tolist(), windows_allocated=ORDER["alloc"].tolist(),
    share_allocated=[round(a / N_WIN, 6) for a in ORDER["alloc"].tolist()], tokens_allocated=(ORDER["alloc"] * L).tolist(),
    budget_tokens=TR["max_tokens"], run_tokens=N_WIN * L, budget_tokens_unused=TR["max_tokens"] - N_WIN * L,
    final_step_windows=N_WIN - (TOTAL_STEPS - 1) * seqs, repeated_window_ids=0, order_sha256=DIGEST)


def data_progress(cursor):
    c = nr_sampler.per_source_counts(ORDER, cursor)
    return {"windows_consumed": int(cursor), "windows_remaining": int(N_WIN - cursor), "tokens_per_source": (c * L).tolist()}


def lr_at(s):
    if s < TR["warmup_steps"]:
        return TR["lr"] * (s + 1) / TR["warmup_steps"]
    p = min(1.0, (s - TR["warmup_steps"]) / max(1, TOTAL_STEPS - TR["warmup_steps"]))
    f = 0.5 * (1 + math.cos(math.pi * p)) if TR["schedule"] == "cosine" else 1 - p
    return TR["min_lr"] + (TR["lr"] - TR["min_lr"]) * f


@torch.no_grad()
def evaluate():
    """per eval set: new-length loss (all / positions < old / >= old) and old-length loss; plus their means over sets."""
    raw.eval()
    out = {}
    for name, val in evals.items():
        rng = np.random.default_rng(12345)                     # the same windows at every evaluation
        n = TR["eval_windows"]
        starts = rng.integers(0, len(val) - L - 1, n)
        tot_new = tot_lo = tot_hi = tot_old = 0.0
        for i in range(0, n, max(1, mb)):
            rows = np.stack([val[s:s + L + 1] for s in starts[i:i + mb]]).astype(np.int64)
            x, y = torch.from_numpy(rows[:, :-1]).to(device), torch.from_numpy(rows[:, 1:]).to(device)
            ln = loss_of(raw, x, y, reduce=False).view(x.shape[0], L)
            tot_new += ln.mean(1).sum().item(); tot_lo += ln[:, :OLD_LEN].mean(1).sum().item()
            tot_hi += ln[:, OLD_LEN:].mean(1).sum().item() if L > OLD_LEN else 0.0
            lo = loss_of(raw, x[:, :OLD_LEN], y[:, :OLD_LEN], reduce=False).view(x.shape[0], OLD_LEN)
            tot_old += lo.mean(1).sum().item()
        out[name] = {"new_len": tot_new / n, "pos_lt_old": tot_lo / n, "pos_ge_old": tot_hi / n if L > OLD_LEN else None, "old_len": tot_old / n}
    raw.train()
    out["mean"] = {k: float(np.mean([v[k] for v in out.values()])) for k in ("new_len", "old_len", "pos_lt_old")}
    if L > OLD_LEN:
        out["mean"]["pos_ge_old"] = float(np.mean([v["pos_ge_old"] for k, v in out.items() if k != "mean"]))
    return out


RNG_CUDA, RNG_CPU = None, None


def gather_rng():
    """every rank's CPU and CUDA generator states, collected on rank 0 (dropout masks differ per rank)."""
    global RNG_CUDA, RNG_CPU
    mine = (torch.get_rng_state(), torch.cuda.get_rng_state() if CUDA else None)
    if DIST:
        allst = [None] * WORLD
        dist.all_gather_object(allst, mine)
    else:
        allst = [mine]
    RNG_CPU = [a[0] for a in allst]
    RNG_CUDA = [a[1] for a in allst] if CUDA else None


def save_last():
    gather_rng()                                               # collective: every rank calls save_last
    if not MASTER:
        return
    st = {"kind": "ctxext_last", "model": raw.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
          "tokens": tokens, "best": best, "history": history, "max_seq_len": L, "rope_extension": cfg["rope_extension"], "guard": GUARD,
          "rng": {"cpu": RNG_CPU, "cuda": RNG_CUDA},
          "sampler": {"kind": "no_repeat", "seed": TR["seed"], "cursor": min(step * seqs, N_WIN), "order_sha256": DIGEST,
                      "src": ORDER["src"], "wid": ORDER["wid"], "alloc": ORDER["alloc"], "n_avail": ORDER["n_avail"],
                      "file_lengths": ORDER["file_lengths"], "files": [f["path"] for f in cfg["train_files"]]}, "config": {"module": scfg["module"], "model": mdict, "tag": cfg.get("tag", "B2_ctxext")}}
    torch.save(st, os.path.join(OUT, "last.pt.tmp")); os.replace(os.path.join(OUT, "last.pt.tmp"), os.path.join(OUT, "last.pt"))


def save_best(kind, ev):
    """Phase-1 best.pt format (model + config with module/model) -> loadable by phase3/backbones.py."""
    if not MASTER:
        return
    name = f"best_{kind}.pt"
    st = {"model": raw.state_dict(), "config": {"module": scfg["module"], "model": mdict, "tag": cfg.get("tag", "B2_ctxext") + "_" + kind,
          "rope_extension": cfg["rope_extension"], "init_from": cfg["init_from"]}, "step": step, "tokens": tokens,
          "best_val": ev["mean"]["new_len" if kind == "newctx" else "old_len"], "eval": ev}
    torch.save(st, os.path.join(OUT, name + ".tmp")); os.replace(os.path.join(OUT, name + ".tmp"), os.path.join(OUT, name))


STOP = {"flag": False}
signal.signal(signal.SIGTERM, lambda *a: STOP.update(flag=True))

if step == 0 and not args.probe:
    ev = evaluate(); history.append({"step": 0, "tokens": 0, "eval": ev}); log("eval (B2 weights, before training = baseline)", eval=ev)
t_eval = t_ckpt = time.time()
# test-only: record every consumed window (step, order position, file, window id) per rank
REC = open(os.path.join(OUT, f"windows_rank{RANK}.txt"), "a") if cfg.get("debug_record_windows") and not args.probe else None
t_run, tok_run = time.time(), 0
end_step = step + args.probe if args.probe else TOTAL_STEPS
raw.train()
while step < end_step:
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    tot = 0.0
    pos = nr_sampler.step_positions(step, seqs, N_WIN, WORLD, RANK)
    n_step = min(seqs, N_WIN - step * seqs)                   # windows in this step over all ranks (< seqs only at the end)
    assert n_step > 0
    for m in range(ACC):
        chunk = pos[m * mb:(m + 1) * mb]
        if len(chunk):
            x, y = windows(zip(ORDER["src"][chunk].tolist(), ORDER["wid"][chunk].tolist()))
            scale = len(chunk) * WORLD / n_step                # gradient = mean over every window of the step
        else:
            x, y = dummy_batch(1); scale = 0.0                 # keeps the DDP collectives aligned; contributes nothing
        if REC is not None and len(chunk):
            REC.write("".join(f"{step} {k} {ORDER['src'][k]} {ORDER['wid'][k]}\n" for k in chunk.tolist()))
        ctx = ddp.no_sync() if (DIST and m < ACC - 1) else torch.enable_grad()
        with ctx:
            l = loss_of(model, x, y) * scale
            scaler.scale(l).backward()
        tot += l.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(raw.parameters(), TR["grad_clip"])
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    step += 1; tokens += n_step * L; tok_run += n_step * L
    assert tokens <= TR["max_tokens"] and tokens == min(step * seqs, N_WIN) * L
    if not math.isfinite(tot):
        log("non-finite loss; stopping", step=step); break
    now = time.time()
    if MASTER and (step % 20 == 0 or args.probe):
        rate = tok_run / (now - t_run)
        log("train", step=step, tokens_B=round(tokens / 1e9, 4), loss=round(tot, 4), lr=lr, tok_s=round(rate),
            eta_h=round((N_WIN * L - tokens) / rate / 3600, 2), **data_progress(min(step * seqs, N_WIN)))
    if args.probe:
        continue
    # one synchronised decision per step (rank 0's clock, broadcast), so every rank evaluates / saves / stops together
    flags = torch.tensor([int((now - SESSION_T0) / 3600 > TR["session_hours"] or STOP["flag"]),
                          int((now - t_eval) / 60 > TR["eval_every_min"]), int((now - t_ckpt) / 60 > TR["ckpt_every_min"])],
                         device=device)
    if DIST:
        dist.broadcast(flags, src=0)
    session_over, eval_due, ckpt_due = (bool(x) for x in flags.tolist())
    if eval_due or step >= end_step or session_over:
        ev = evaluate(); history.append({"step": step, "tokens": tokens, "eval": ev}); t_eval = time.time()
        saved = []
        for kind, key in (("newctx", "new_len"), ("oldctx", "old_len")):
            if ev["mean"][key] < best[kind]:
                best[kind] = ev["mean"][key]; save_best(kind, ev); saved.append(kind)
        log("eval", step=step, tokens_B=round(tokens / 1e9, 4), saved=saved, eval=ev)
    if ckpt_due or step >= end_step or session_over:
        if REC is not None:
            REC.flush()
        save_last(); t_ckpt = time.time(); log("checkpoint", step=step, tokens_B=round(tokens / 1e9, 4), **data_progress(min(step * seqs, N_WIN)))
    if session_over:
        log("session limit reached: stopping (resume in the next session)", step=step, tokens_B=round(tokens / 1e9, 4)); break

if args.probe and MASTER:
    rate = tok_run / (time.time() - t_run)
    peak = torch.cuda.max_memory_allocated() / 1e9 if CUDA else None
    log("probe", tok_s=round(rate), peak_mem_GB_per_gpu=round(peak, 2) if peak else None,
        hours_for_target=round(TR["max_tokens"] / rate / 3600, 1),
        sessions_needed=round(N_WIN * L / rate / 3600 / TR["session_hours"], 1))
if not args.probe and step >= TOTAL_STEPS:
    log("DONE: all windows consumed", step=step, tokens=tokens, budget_tokens=TR["max_tokens"], best=best,
        **data_progress(min(step * seqs, N_WIN)))
if REC is not None:
    REC.close()
if DIST:
    dist.destroy_process_group()
