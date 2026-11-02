# Plumbing: one decoder-only model trained data-parallel on every visible GPU, then multi-strategy generation
"""Train one hand-written A3 decoder (arch1 / arch3) on all GPUs with DistributedDataParallel, then write
prediction_<strategy>.json for every decoding strategy (written with Claude; declare it).

Usage:  torchrun --standalone --nproc_per_node=2 train_ddp.py run_B.json
        (also runs as a single process: python train_ddp.py run_B.json)

Same model / optimiser / schedule semantics as the single-GPU train_lm.py of the A/B run:
  * the global batch (batch_tokens) is split evenly over the GPUs; each GPU runs micro-batches of its share,
    gradients are averaged across GPUs (DDP), so one optimiser step still sees batch_tokens tokens;
  * the LR schedule is driven by wall-clock progress to `deadline_unix` (or max_tokens), computed on rank 0
    and broadcast every step, so all GPUs always use the same LR and stop at the same step;
  * eval, checkpoints, logging and generation happen on rank 0 only; checkpoints hold the plain (unwrapped)
    state_dict, loadable exactly like the A/B checkpoints (gen_local.py works unchanged).
Validation uses a fixed set of windows (eval_windows, default 360 = the A/B/C/D runs' 40 x 9), so val losses
are comparable with the earlier runs.

Probe mode (cfg["probe"]): same code path on synthetic tokens for a fixed number of minutes, then times
worst-case generation per decoding strategy instead of writing predictions.
"""
import os
import sys
import json
import math
import time
import queue
import random
import threading
import traceback
import gc
import importlib
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import fastattn
from a1_tokenizer import ByteBPE

# ---------------------------------------------------------------------------
# distributed setup
# ---------------------------------------------------------------------------
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DIST = WORLD > 1
CUDA = torch.cuda.is_available()         # CPU path exists only for local smoke tests
if CUDA:
    torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}" if CUDA else "cpu")
if DIST:
    dist.init_process_group("nccl" if CUDA else "gloo", timeout=timedelta(minutes=10))
MASTER = RANK == 0

cfg = json.load(open(sys.argv[1]))
TAG, OUT = cfg["tag"], cfg["out_dir"]
os.makedirs(OUT, exist_ok=True)
TR, GEN = cfg["train"], cfg["gen"]
DEADLINE = cfg["deadline_unix"]
SESSION_END = cfg["session_end_unix"]


def log(msg, force=False, **kv):
    """rank 0 logs everything; other ranks only what is passed with force=True (errors)."""
    if not (MASTER or force):
        return
    rec = {"t": round(time.time(), 1), "msg": msg, **kv}
    if not MASTER:
        rec["rank"] = RANK
    print(json.dumps(rec), flush=True)
    name = "log.jsonl" if MASTER else f"log_rank{RANK}.jsonl"
    with open(os.path.join(OUT, name), "a") as f:
        f.write(json.dumps(rec) + "\n")


seed = TR["seed"]
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)     # identical init on every rank
AMP = torch.float16                      # T4 has no bf16 tensor cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
arch = importlib.import_module(cfg["module"])
mcfg = arch.LMConfig(**cfg["model"])
raw_model = arch.DecoderLM(mcfg)
if cfg.get("init_std"):
    std = cfg["init_std"]
    for m in raw_model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, std)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
n_params = sum(p.numel() for p in raw_model.parameters())
if TR.get("fast_attention", True):
    fastattn.patch_model(raw_model)
raw_model.to(device)
init_from = cfg.get("init_from")
if init_from and not (cfg.get("resume_from") and os.path.exists(cfg["resume_from"])):
    # warm start: weights only (fresh optimiser, fresh warmup + schedule); the architecture must match exactly
    ck = torch.load(init_from, map_location="cpu", weights_only=False)
    src = ck.get("config", {})
    assert src.get("module") == cfg["module"] and src.get("model") == cfg["model"], \
        f"init_from checkpoint architecture {src.get('module')} {src.get('model')} != this run's {cfg['module']} {cfg['model']}"
    raw_model.load_state_dict(ck["model"])
    log("init_from", path=init_from, src_step=ck.get("step"), src_tokens_M=round(ck.get("tokens", 0) / 1e6, 1),
        src_best_val=ck.get("best_val"))
    del ck
T = mcfg.max_seq_len
V = mcfg.vocab_size
assert GEN["max_new_tokens"] < T, "gen.max_new_tokens must be < max_seq_len"
log("model", tag=TAG, module=cfg["module"], params_M=round(n_params / 1e6, 2), seq_len=T, world_size=WORLD)

# ---------------------------------------------------------------------------
# data: random windows from the uint16 token stream, prefetched on a thread (different stream per rank)
# ---------------------------------------------------------------------------
train_mm = np.memmap(os.path.join(cfg["data_dir"], "train.bin"), dtype=np.uint16, mode="r")
val_mm = np.memmap(os.path.join(cfg["data_dir"], "val.bin"), dtype=np.uint16, mode="r")
assert int(train_mm[:10_000_000].max()) < V, "token id >= vocab_size: data/tokenizer mismatch"


def windows(mm, starts):
    arr = np.stack([mm[s:s + T + 1] for s in starts]).astype(np.int64)
    t = torch.from_numpy(arr)
    t = t.pin_memory() if CUDA else t
    return t[:, :-1], t[:, 1:]


class Prefetcher:
    def __init__(self, mb, seed):
        self.mb, self.rng = mb, np.random.default_rng(seed)
        self.q = queue.Queue(maxsize=6)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        hi = len(train_mm) - T - 1
        while True:
            self.q.put(windows(train_mm, self.rng.integers(0, hi, self.mb)))

    def next(self):
        x, y = self.q.get()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def loss_fn(model, x, y):
    with torch.autocast("cuda", dtype=AMP, enabled=CUDA):
        logits = model(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))


def all_min(v):
    if not DIST:
        return v
    t = torch.tensor([v], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


# ---------------------------------------------------------------------------
# batch: global batch split over the GPUs; micro-batch = largest divisor of the per-GPU share that fits
# ---------------------------------------------------------------------------
batch_seqs = max(1, TR["batch_tokens"] // T)
assert batch_seqs % WORLD == 0, (f"batch_tokens // max_seq_len = {batch_seqs} sequences must be divisible by the "
                                 f"{WORLD} GPUs (e.g. batch_tokens = {WORLD * math.ceil(batch_seqs / WORLD) * T})")
rank_seqs = batch_seqs // WORLD
free_mem = torch.cuda.mem_get_info()[0] + torch.cuda.memory_allocated() if CUDA else 0
opt_state_bytes = 2 * 4 * sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
ddp_bytes = 4 * n_params if DIST else 0                      # DDP gradient buckets
budget = free_mem - opt_state_bytes - ddp_bytes - 1.5e9
candidates = sorted([d for d in range(1, min(rank_seqs, TR.get("max_micro_batch", 64)) + 1)
                     if rank_seqs % d == 0], reverse=True)
mb, peak = None, 0
if not CUDA:
    mb = min(candidates, key=lambda d: abs(d - 2))
for cand in candidates if CUDA else []:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x, y = windows(train_mm, np.arange(cand) * 997)
        raw_model.train()
        loss_fn(raw_model, x.to(device), y.to(device)).backward()
        raw_model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated()
        if peak < budget:
            mb = cand
            break
    except torch.cuda.OutOfMemoryError:
        pass
    raw_model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
mb = all_min(mb if mb is not None else 0)                     # same micro-batch on every GPU
assert mb >= 1, "even micro-batch 1 does not fit; reduce max_seq_len or model size"
accum = rank_seqs // mb
log("batch", global_seqs=batch_seqs, gpus=WORLD, seqs_per_gpu=rank_seqs, micro_batch=mb, grad_accum=accum,
    tokens_per_step=batch_seqs * T, probe_peak_GB=round(peak / 1e9, 2))

ddp = DDP(raw_model, device_ids=[LOCAL_RANK] if CUDA else None, broadcast_buffers=False,
          gradient_as_bucket_view=True) if DIST else None
base = ddp if DIST else raw_model                              # what the training step calls
model = base
if TR.get("compile", False):
    t_compile = time.time()
    try:
        model = torch.compile(base)
        x, y = windows(train_mm, np.arange(mb) * 991)
        loss_fn(model, x.to(device), y.to(device)).backward()
        raw_model.zero_grad(set_to_none=True)
        log("torch.compile ok", seconds=round(time.time() - t_compile, 1))
    except Exception as e:                                     # fall back to eager
        model = base
        raw_model.zero_grad(set_to_none=True)
        log("torch.compile failed, using eager", error=repr(e)[:300])

# ---------------------------------------------------------------------------
# optimiser (weight decay on >=2-D weights only); one per rank, identical updates
# ---------------------------------------------------------------------------
decay = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
no_decay = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
groups = [{"params": decay, "weight_decay": TR["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
try:
    opt = torch.optim.AdamW(groups, lr=TR["lr"], betas=tuple(TR["betas"]), eps=TR.get("eps", 1e-8), fused=CUDA)
except (RuntimeError, TypeError):
    opt = torch.optim.AdamW(groups, lr=TR["lr"], betas=tuple(TR["betas"]), eps=TR.get("eps", 1e-8))
scaler = torch.amp.GradScaler("cuda", enabled=CUDA)

step, tokens_seen, train_seconds_before, best_val = 0, 0, 0.0, float("inf")
warmup_end_progress = None
if cfg.get("resume_from") and os.path.exists(cfg["resume_from"]):
    ck = torch.load(cfg["resume_from"], map_location=device, weights_only=False)
    raw_model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    step, tokens_seen = ck["step"], ck["tokens"]
    train_seconds_before, best_val = ck["train_seconds"], ck.get("best_val", float("inf"))
    warmup_end_progress = ck.get("warmup_end_progress")
    log("resumed", path=cfg["resume_from"], step=step, tokens=tokens_seen)

# ---------------------------------------------------------------------------
# schedule: warmup in steps, then decay over wall-clock progress (computed on rank 0, broadcast)
# ---------------------------------------------------------------------------
if cfg.get("probe"):                     # runtime probe: fixed training window from here
    DEADLINE = min(DEADLINE, time.time() + cfg["probe"]["train_minutes"] * 60)
t_start = time.time()
schedule_seconds = (cfg["schedule_hours"] * 3600 if cfg.get("schedule_hours")
                    else train_seconds_before + (DEADLINE - t_start))
max_tokens = TR.get("max_tokens")


def progress_now():
    p = (train_seconds_before + time.time() - t_start) / schedule_seconds
    if max_tokens:
        p = max(p, tokens_seen / max_tokens)
    return p


def lr_at(step, p):
    global warmup_end_progress
    lr, min_lr, warm = TR["lr"], TR["min_lr"], TR["warmup_steps"]
    if step < warm:
        return lr * (step + 1) / warm
    if warmup_end_progress is None:
        warmup_end_progress = min(p, 0.99)
    q = min(1.0, max(0.0, (p - warmup_end_progress) / (1 - warmup_end_progress)))
    kind = TR.get("schedule", "cosine")
    if kind == "cosine":
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * q))
    if kind == "linear":
        return lr - (lr - min_lr) * q
    return lr


def control():
    """rank 0 decides (lr, progress, stop); every rank receives the same values."""
    t = torch.zeros(3, dtype=torch.float64, device=device)
    if MASTER:
        p = progress_now()
        stop = time.time() >= DEADLINE or p >= 1.0
        t[0], t[1], t[2] = (0.0 if stop else lr_at(step, p)), p, float(stop)
    if DIST:
        dist.broadcast(t, src=0)
    return t[0].item(), t[1].item(), bool(t[2].item())


# ---------------------------------------------------------------------------
# eval + checkpoints (rank 0 only, no collectives)
# ---------------------------------------------------------------------------
val_starts = np.random.default_rng(1234).integers(0, len(val_mm) - T - 1, TR.get("eval_windows", 360))


@torch.no_grad()
def evaluate():
    raw_model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val_starts), mb):
        x, y = windows(val_mm, val_starts[i:i + mb])
        tot += loss_fn(raw_model, x.to(device), y.to(device)).item() * len(x)
        n += len(x)
    raw_model.train()
    return tot / n


def save(path, full):
    state = {"model": raw_model.state_dict(), "config": cfg, "step": step, "tokens": tokens_seen,
             "train_seconds": train_seconds_before + time.time() - t_start, "best_val": best_val,
             "warmup_end_progress": warmup_end_progress, "world_size": WORLD}
    if full:
        state.update(opt=opt.state_dict(), scaler=scaler.state_dict())
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def do_eval(tag):
    global best_val
    t0 = time.time()
    vl = evaluate()
    t_eval = time.time() - t0
    improved = vl < best_val
    if improved:
        best_val = vl
        save(os.path.join(OUT, "best.pt"), full=False)
    save(os.path.join(OUT, "last.pt"), full=True)
    log(tag, step=step, tokens=tokens_seen, val_loss=round(vl, 4), val_ppl=round(math.exp(vl), 2),
        best=improved, hours=round((time.time() - t_start) / 3600, 2),
        eval_s=round(t_eval, 1), save_s=round(time.time() - t0 - t_eval, 1))


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
pf = Prefetcher(mb, seed + 1000 * RANK + step)
raw_model.train()
last_eval = time.time()
last_log, tok_at_last_log = time.time(), tokens_seen
eval_every = TR["eval_every_min"] * 60
bad_logs = 0
log("training", deadline_in_h=round((DEADLINE - t_start) / 3600, 2), schedule_hours=round(schedule_seconds / 3600, 2))


def train_step(lr):
    for g in opt.param_groups:
        g["lr"] = lr
    loss_sum = torch.zeros((), device=device)
    for i in range(accum):
        x, y = pf.next()
        sync = (i == accum - 1) or not DIST
        with (nullcontext() if sync else ddp.no_sync()):       # all-reduce once per optimiser step
            loss = loss_fn(model, x, y) / accum
            scaler.scale(loss).backward()
        loss_sum += loss.detach()
    if TR.get("grad_clip"):
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), TR["grad_clip"]).item()
    else:
        gnorm = float("nan")
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return loss_sum, gnorm


while True:
    try:
        lr, p, stop = control()
        if stop:
            break
        loss_sum, gnorm = train_step(lr)
    except Exception:
        # with several GPUs a failure on one rank cannot be recovered in step; stop cleanly instead
        opt.zero_grad(set_to_none=True)
        log("training error; stopping training (rank 0 then evaluates, saves and generates)", force=True,
            step=step, error=traceback.format_exc()[-1500:])
        break
    step += 1
    tokens_seen += batch_seqs * T

    if step % TR.get("log_every", 50) == 0:
        if DIST:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG if CUDA else dist.ReduceOp.SUM)
            if not CUDA:
                loss_sum /= WORLD
        loss_val = loss_sum.item()
        now = time.time()
        log("train", step=step, loss=round(loss_val, 4), lr=lr, grad_norm=round(gnorm, 3),
            tok_per_s=int((tokens_seen - tok_at_last_log) / (now - last_log)),
            tokens_M=round(tokens_seen / 1e6, 1), progress=round(p, 4),
            mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2) if CUDA else 0.0)
        last_log, tok_at_last_log = now, tokens_seen
        bad_logs = 0 if math.isfinite(loss_val) else bad_logs + 1
        if bad_logs >= 3:
            log("loss non-finite for 3 log intervals; stopping training (generation uses the best checkpoint)",
                step=step)
            break                                              # every rank sees the same averaged loss
    if MASTER and time.time() - last_eval >= eval_every:
        do_eval("eval")
        last_eval = time.time()

del pf
if DIST and os.environ.get("DDP_CHECK"):                       # smoke tests: every rank must hold identical weights
    with torch.no_grad():
        s = torch.stack([p.double().sum() for p in raw_model.parameters()]).sum().view(1)
    sums = [torch.zeros_like(s) for _ in range(WORLD)]
    dist.all_gather(sums, s)
    log("ddp_check", param_sums=[round(x.item(), 6) for x in sums], identical=len({x.item() for x in sums}) == 1)
if not MASTER:                                                 # only rank 0 continues
    if DIST:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    sys.exit(0)

try:
    do_eval("final_eval")
except Exception:
    log("final eval failed", error=traceback.format_exc()[-1500:])
    save(os.path.join(OUT, "last.pt"), full=False)
log("training done", steps=step, tokens=tokens_seen, best_val=round(best_val, 4))
if DIST:
    try:
        dist.destroy_process_group()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# generation (rank 0): one prediction file per decoding strategy (+ held-out val prompts for local scoring)
# ---------------------------------------------------------------------------
if not GEN.get("decoding"):                                    # training-only run: generate elsewhere from best.pt
    log("training only (no decoding strategies): best.pt / last.pt are the outputs")
    log("all done")
    sys.exit(0)

del opt
if CUDA:
    torch.cuda.empty_cache()
which = GEN.get("checkpoint", "best")
if not os.path.exists(os.path.join(OUT, f"{which}.pt")):
    log(f"{which}.pt missing (no finite val loss?); falling back to last.pt")
    which = "last"
ck = torch.load(os.path.join(OUT, f"{which}.pt"), map_location=device, weights_only=False)
raw_model.load_state_dict(ck["model"])
raw_model.eval()
if not TR.get("fast_attention", True):
    fastattn.patch_model(raw_model)          # KV cache needs the patched forward


class ProbeTokenizer:
    """Stand-in for the real tokenizer in probe mode: every prompt encodes to `prompt_len` random ids
    and no token ever ends a word, so each batch runs the full max_new_tokens (worst case)."""
    n_special, eow_id = 32, 0

    def __init__(self, vocab_size, prompt_len):
        self.token_eow = [False] * vocab_size
        self.token_bytes = [b"x"] * vocab_size
        self._ids = np.random.default_rng(0).integers(0, vocab_size, prompt_len).tolist()

    def encode(self, text):
        return self._ids

    def decode(self, ids):
        return ""


tokenizer = ProbeTokenizer(V, T - GEN["max_new_tokens"]) if cfg.get("probe") else ByteBPE.from_json(cfg["tokenizer_path"])
STRATEGIES = list(GEN["decoding"].items())         # [(name, {temperature, top_k, top_p, repetition_penalty}), ...]


def run_gen(prompts, dec):
    bs = GEN["batch_size"]
    while True:
        try:
            return fastattn.generate_words(
                raw_model, tokenizer, prompts, max_seq_len=T, target_words=GEN["target_words"],
                max_new_tokens=GEN["max_new_tokens"], batch_size=bs, temperature=dec["temperature"],
                top_k=dec.get("top_k"), top_p=dec.get("top_p"), repetition_penalty=dec.get("repetition_penalty", 1.0),
                device=device, amp_dtype=AMP if CUDA else torch.float32,
                no_repeat_ngram=dec.get("no_repeat_ngram", 0), ngram_scope=dec.get("ngram_scope", "output"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs //= 2
            assert bs >= 1
            log("generation OOM, retrying", batch_size=bs)


def looping_share(texts):
    """share of predictions in which some 4-word sequence occurs 3+ times (a simple repetition indicator)."""
    n = 0
    for t in texts:
        w = t.split()
        grams = {}
        for i in range(len(w) - 3):
            g = tuple(w[i:i + 4])
            grams[g] = grams.get(g, 0) + 1
        n += bool(grams) and max(grams.values()) >= 3
    return round(n / max(1, len(texts)), 3)


if cfg.get("probe"):
    for name, dec in STRATEGIES:
        times = []
        for _ in range(2):                   # the first batch includes CUDA warm-up
            if CUDA:
                torch.cuda.synchronize()
            t0 = time.time()
            run_gen(["probe"] * GEN["batch_size"], dec)
            if CUDA:
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        log("probe_generation", strategy=name, batch_size=GEN["batch_size"], prompt_tokens=T - GEN["max_new_tokens"],
            new_tokens=GEN["max_new_tokens"], first_batch_s=round(times[0], 2), batch_s=round(times[1], 2))
    log("all done")
    sys.exit(0)

test = json.load(open(cfg["test_path"], encoding="utf-8"))
ids = list(test.keys())
test_prompts = [test[k]["text"] for k in ids]
last_minutes = 0.0
for i, (name, dec) in enumerate(STRATEGIES):
    if i and SESSION_END - time.time() < 1.5 * last_minutes * 60 + 300:
        log("skipping decoding strategy: not enough session time left", strategy=name)
        continue
    torch.manual_seed(GEN.get("seed", 0))
    t0 = time.time()
    preds = run_gen(test_prompts, dec)
    last_minutes = (time.time() - t0) / 60
    prediction = {k: {"text": p} for k, p in zip(ids, preds)}
    paths = [os.path.join(OUT, f"prediction_{name}.json")] + ([os.path.join(OUT, "prediction.json")] if i == 0 else [])
    for path in paths:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prediction, f, ensure_ascii=False, indent=2)
    short = sum(len(p.split()) < GEN["target_words"] for p in preds)
    log("prediction.json written", strategy=name, decoding=dec, n=len(preds), minutes=round(last_minutes, 1),
        under_target_words=short, looping_share=looping_share(preds), checkpoint=which)

vp_path = os.path.join(cfg["data_dir"], "val_prompts.json")
n_val = GEN.get("n_val_prompts", 0)
if n_val and os.path.exists(vp_path):
    vp = json.load(open(vp_path, encoding="utf-8"))
    keys = list(vp)[:n_val]
    for name, dec in STRATEGIES:
        if SESSION_END - time.time() < 600 + 0.25 * last_minutes * 60:
            log("skipping val generations: not enough session time left", strategy=name)
            break
        try:
            torch.manual_seed(GEN.get("seed", 0))
            vpreds = run_gen([vp[k]["text"] for k in keys], dec)
            with open(os.path.join(OUT, f"val_generations_{name}.json"), "w", encoding="utf-8") as f:
                json.dump({k: {"prediction": p, "reference": vp[k]["reference"]} for k, p in zip(keys, vpreds)},
                          f, ensure_ascii=False, indent=2)
            log("val_generations written", strategy=name, n=len(keys), looping_share=looping_share(vpreds))
        except Exception:
            log("val generation failed (prediction files are unaffected)", strategy=name,
                error=traceback.format_exc()[-1500:])
log("all done")
