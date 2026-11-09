#!/usr/bin/env python3
"""Fine-tune one Phase-1 model as a PCR (prior case retrieval) bi-encoder and write its Codabench submission
(plumbing; declare it). Document encoder adapted from the Phase-4/5 trainer (same chunking / pooling / techniques).

  python train_pcr.py run_config.json

Model: case text -> your ByteBPE -> up to k chunks (pcr_data.chunk_spans: even | headtail | overlap) -> Phase-1 backbone
+ YOUR technique (techniques/<name>.py) -> one vector per chunk (chunk_pool: mean | last) -> one vector per case
(doc_pool: attn = learned score per chunk, softmax over the case's chunks | mean) -> optional linear projection
(proj_dim: None = keep d_model). The SAME encoder embeds queries and candidates. score(q, c) = sim(q, c) / temperature,
sim = cosine | dot (config "similarity").

Training (one example = one (query, relevant candidate) pair of train.json, every pair once per epoch, shuffled):
  the pair's scores over its candidate set -> cross-entropy with the relevant candidate as the target (InfoNCE).
  Candidate set of a pair = its relevant candidate + negatives.per_positive negatives drawn from the query's own pool:
    source "bm25"   uniformly from the top negatives.bm25_top BM25 candidates that are not relevant (hard negatives)
    source "random" uniformly from the non-relevant pool cases
  + (negatives.in_batch = True) every other candidate document of the step. The query's OTHER relevant candidates and
  the query itself are always masked out of its row (never used as negatives).
  Gradient caching: a step embeds all its documents without gradients, takes d(loss)/d(embeddings), then re-encodes each
  micro-batch (train.micro_docs documents, same RNG state -> same dropout) and back-propagates that gradient, so the
  loss sees the whole step's documents while memory holds one micro-batch.

KD (cfg["kd"] = {"teacher_run": <finished run dir>, ...your kd options}): the teacher's embeddings of every train query
and train-pool case (its own chunking) are computed once and cached (<teacher_run>/train_emb.npz); the teacher's
scores over exactly the student's candidate rows (same masks, the teacher's similarity / temperature) and YOUR
techniques/kd.py kd_loss(student (P, 1, M), teacher (P, 1, M), target (P, 1), **opts) REPLACE the InfoNCE loss.
The student must have fewer parameters than the teacher (checked).

Evaluation: every train.eval_every_steps steps and after every epoch, dev queries are ranked against dev_candidates.json
-> Micro-F1@7 (the competition metric; selection score of best_model.pt), + P/R/F1 at other cut-offs and recall@100.
Test: test queries vs test_candidates.json -> prediction.json {query id: [7 candidate ids]} + zip.

Outputs in <out_dir>/<name>/: config.json, log.jsonl, best_model.pt, dev_scores.json, dev_S.npz / test_S.npz (full score
matrices, float32, + ids; for pcr_select.py fusion), prediction.json, submission_<name>.zip, summary.json.
"""
import inspect
import json
import math
import os
import random
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backbones                       # noqa: E402
import pcr_data                        # noqa: E402
import techniques                      # noqa: E402
from a1_tokenizer import ByteBPE       # noqa: E402
from inject import count_params        # noqa: E402

REQUIRED = {
    "chunking": ["strategy", "max_chunks"],
    "negatives": ["source", "per_positive", "in_batch"],
    "train": ["epochs", "batch_pairs", "micro_docs", "lr", "head_lr", "weight_decay", "warmup_frac", "schedule",
              "grad_clip", "seed"],
}
MASK = -1e4


def check_config(cfg):
    missing = [f"{s}.{k}" for s, ks in REQUIRED.items() for k in ks if (cfg.get(s) or {}).get(k) is None]
    missing += [k for k in ("chunk_pool", "doc_pool", "similarity", "temperature") if cfg.get(k) is None]
    missing += ["technique.name"] if (cfg.get("technique") or {}).get("name") is None else []
    missing += ["negatives.bm25_top"] if (cfg.get("negatives") or {}).get("source") == "bm25" and \
        cfg["negatives"].get("bm25_top") is None else []
    assert not missing, "fill these config values first: " + ", ".join(missing)
    ch, tr, ng = cfg["chunking"], cfg["train"], cfg["negatives"]
    assert ch["strategy"] in ("even", "headtail", "overlap")
    assert ch["strategy"] != "overlap" or ch.get("stride"), "chunking.stride is needed for overlap"
    assert cfg["chunk_pool"] in ("mean", "last") and cfg["doc_pool"] in ("attn", "mean")
    assert cfg["similarity"] in ("cosine", "dot") and cfg["temperature"] > 0
    assert cfg.get("proj_dim") is None or int(cfg["proj_dim"]) > 0
    assert ng["source"] in ("bm25", "random") and ng["per_positive"] >= 0
    assert ng["per_positive"] > 0 or ng["in_batch"], "no negatives at all: per_positive > 0 or in_batch = True"
    assert tr["schedule"] in ("linear", "cosine", "constant")
    tech = cfg["technique"]
    unset = [f"technique.{k}" for k, v in tech.items() if v is None]
    if cfg.get("kd"):
        unset += [f"kd.{k}" for k, v in cfg["kd"].items() if v is None]
    assert not unset, "fill these config values first: " + ", ".join(unset)
    sig = inspect.signature(techniques.get(tech["name"]).apply)
    try:
        sig.bind(None, **{k: v for k, v in tech.items() if k != "name"})
    except TypeError as e:
        raise AssertionError(f"technique '{tech['name']}' options do not fit apply{sig}: {e}")


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg, **kv):
        rec = {"t": round(time.time(), 1), "msg": msg, **kv}
        print(json.dumps(rec), flush=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# model: backbone -> chunk vectors -> case vector (-> projection)
# ---------------------------------------------------------------------------

class DocEncoder(nn.Module):
    def __init__(self, backbone, chunk_pool, doc_pool, proj_dim=None):
        super().__init__()
        self.backbone = backbone
        self.chunk_pool, self.doc_pool = chunk_pool, doc_pool
        self.attn_score = nn.Linear(backbone.d_model, 1) if doc_pool == "attn" else None
        self.proj = nn.Linear(backbone.d_model, int(proj_dim)) if proj_dim else None

    def head_modules(self):
        return [m for m in (self.attn_score, self.proj) if m is not None]

    def chunk_vectors(self, chunks, device):
        """chunks: list of 1-D int arrays -> (n_chunks, d), same order. Equal-length chunks are batched together."""
        out = [None] * len(chunks)
        by_len = {}
        for i, c in enumerate(chunks):
            by_len.setdefault(len(c), []).append(i)
        for n, idx in by_len.items():
            ids = torch.from_numpy(np.stack([chunks[i] for i in idx]).astype(np.int64)).to(device)
            h = self.backbone(ids)                                      # (b, n, d), no padding anywhere
            v = h.mean(1) if self.chunk_pool == "mean" else h[:, -1]
            for j, i in enumerate(idx):
                out[i] = v[j]
        return torch.stack(out)

    def forward(self, items, device):
        """items: list of {"chunks": [...]} -> embeddings (n_docs, d)."""
        chunks = [c for it in items for c in it["chunks"]]
        vec = self.chunk_vectors(chunks, device)
        counts = [len(it["chunks"]) for it in items]
        K, d = max(counts), vec.shape[-1]
        padded = vec.new_zeros(len(items), K, d)
        mask = torch.zeros(len(items), K, dtype=torch.bool, device=vec.device)
        k = 0
        for i, n in enumerate(counts):
            padded[i, :n] = vec[k:k + n]
            mask[i, :n] = True
            k += n
        if self.doc_pool == "attn":
            s = self.attn_score(padded).squeeze(-1).float().masked_fill(~mask, float("-inf"))
            w = torch.softmax(s, dim=-1).to(padded.dtype)
            doc = (w.unsqueeze(-1) * padded).sum(1)
        else:
            doc = (padded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.proj(doc) if self.proj is not None else doc


def similarity(q, c, kind, temperature):
    """q (P, d), c (M, d) float32 -> scores (P, M)."""
    if kind == "cosine":
        q, c = F.normalize(q, dim=-1), F.normalize(c, dim=-1)
    return q @ c.T / temperature


def build_model(cfg, vocab_size, load_weights=True):
    bb, spec, info = backbones.build_backbone(cfg["phase1_ckpt"], cfg.get("backbone", {}), vocab_size,
                                              load_weights=load_weights, fast_attention=cfg.get("fast_attention", True))
    for p in bb.parameters():
        p.requires_grad_(True)
    tech = cfg["technique"]
    tmod = techniques.get(tech["name"])
    tmod.apply(bb, **{k: v for k, v in tech.items() if k != "name"})
    model = DocEncoder(bb, cfg["chunk_pool"], cfg["doc_pool"], cfg.get("proj_dim"))
    for m in model.head_modules():
        for p in m.parameters():
            p.requires_grad_(True)
    return model, tmod, info


def chunk_len(cfg, backbone):
    limit = backbone.max_len - backbone.n_virtual_tokens          # prompt tuning prepends virtual tokens
    L = cfg["chunking"].get("chunk_len") or limit
    assert L <= limit, f"chunk_len {L} > model input limit {limit}"
    return L


def save_model(model, path):
    torch.save(model.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)


def resolve_ckpt(cfg):
    if not os.path.exists(cfg["phase1_ckpt"]):               # earlier session: same checkpoint, new input path
        import glob
        tail = os.path.join(*cfg["phase1_ckpt"].split(os.sep)[-2:])
        hits = sorted(glob.glob(os.path.join("/kaggle/input", "**", tail), recursive=True))
        assert hits, f"Phase-1 checkpoint {cfg['phase1_ckpt']} not found (looked for */{tail})"
        cfg["phase1_ckpt"] = hits[0]
    return cfg


def load_run(run_dir, device, vocab_size):
    cfg = resolve_ckpt(json.load(open(os.path.join(run_dir, "config.json"))))
    model, _, _ = build_model(cfg, vocab_size, load_weights=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu"))
    return model.to(device).eval(), cfg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data(cfg, tok):
    D, cache = cfg["data_dir"], cfg.get("cache_dir")
    data, arrays, rep = {}, {}, {}
    for s in ("train", "dev", "test"):
        q, pool = pcr_data.load_split(D, s)
        data[s] = {"queries": q, "pool": pool}
        rep[s] = pcr_data.split_stats(q, pool)
    qh = {s: {pcr_data.norm_hash(d["text"]) for d in data[s]["queries"]} for s in data}
    rep["train_queries_equal_dev_queries"] = len(qh["train"] & qh["dev"])
    rep["train_queries_equal_test_queries"] = len(qh["train"] & qh["test"])
    for s in data:
        for part in ("queries", "pool"):
            docs = data[s][part]
            arrays[(s, part)] = pcr_data.encode_docs(tok, docs, os.path.join(cache, f"{s}_{part}_ids.npz") if cache else None)
    return data, arrays, rep


def smoke_subset(data, arrays, n_train, seed=0):
    """few real queries + small pools (their relevant cases + random others); not submittable."""
    rnd = random.Random(seed)
    for s, nq, extra in (("train", n_train, 48), ("dev", 4, 24), ("test", 4, 24)):
        qs, pool = data[s]["queries"], data[s]["pool"]
        qi = sorted(range(len(qs)), key=lambda i: -len(arrays[(s, "queries")][i]))[:nq]
        keep = {c for i in qi for c in (qs[i]["relevant"] or [])}
        others = [j for j, d in enumerate(pool) if d["id"] not in keep]
        pj = [j for j, d in enumerate(pool) if d["id"] in keep] + rnd.sample(others, min(extra, len(others)))
        data[s] = {"queries": [qs[i] for i in qi], "pool": [pool[j] for j in pj]}
        arrays[(s, "queries")] = [arrays[(s, "queries")][i] for i in qi]
        arrays[(s, "pool")] = [arrays[(s, "pool")][j] for j in pj]


def make_items(docs, arr, cfg, L):
    ch = cfg["chunking"]
    return pcr_data.build_items(docs, arr, L, ch["strategy"], ch["max_chunks"], ch.get("stride"))


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed(model, items, micro, device, amp):
    """-> float32 tensor (n, d) on device, same order as items (micro-batches grouped by chunk count)."""
    model.eval()
    out = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["chunks"]))
    for a in range(0, len(order), micro):
        idx = order[a:a + micro]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            e = model([items[i] for i in idx], device).float()
        for j, i in enumerate(idx):
            out[i] = e[j]
    return torch.stack(out)


def score_matrix(model, q_items, p_items, cfg, micro, device, amp):
    qe, pe = embed(model, q_items, micro, device, amp), embed(model, p_items, micro, device, amp)
    return similarity(qe, pe, cfg["similarity"], cfg["temperature"]).cpu().numpy()


def grad_cache_backward(model, docs, micro, device, amp, loss_fn, scale=lambda x: x):
    """Gradient caching: loss over the embeddings of ALL docs, memory of one micro-batch.
    pass 1 embeds every micro-batch without gradients (RNG state saved), loss_fn(E (n, d) float32) is back-propagated to
    E only, pass 2 re-encodes each micro-batch with gradients under the same RNG state (identical dropout masks) and
    back-propagates dL/dE into the model. -> (loss, tokens encoded with gradients)."""
    def rng_state():
        return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)

    def set_rng(st):
        torch.set_rng_state(st[0])
        if st[1] is not None:
            torch.cuda.set_rng_state(st[1], device)

    order = sorted(range(len(docs)), key=lambda i: len(docs[i]["chunks"]))
    mbs = [order[i:i + micro] for i in range(0, len(order), micro)]
    states, E = [], [None] * len(docs)
    with torch.no_grad():
        for mb in mbs:
            states.append(rng_state())
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                e = model([docs[i] for i in mb], device).float()
            for j, i in enumerate(mb):
                E[i] = e[j]
    E = torch.stack(E).detach().requires_grad_(True)
    loss = loss_fn(E)
    scale(loss).backward()
    G, ntok = E.grad, 0
    end_state = rng_state()
    for mb, st in zip(mbs, states):
        set_rng(st)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            e = model([docs[i] for i in mb], device).float()
        e.backward(G[mb])
        ntok += sum(len(c) for i in mb for c in docs[i]["chunks"])
    set_rng(end_state)
    return loss.detach(), ntok


def gold_of(queries):
    return {q["id"]: q["relevant"] for q in queries}


def lr_factor(step, total, warmup, schedule):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1.0 - prog if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * prog))


def bm25_top(data, split, top, cache_dir, log=None):
    """{query id: [pool ids by BM25 score, best first, the query itself excluded]} (top `top`), cached."""
    path = os.path.join(cache_dir, f"bm25_{split}_top{top}.json") if cache_dir else None
    qids = [q["id"] for q in data[split]["queries"]]
    if path and os.path.exists(path):
        c = json.load(open(path))
        if sorted(c) == sorted(qids):
            return c
    t0 = time.time()
    bm = pcr_data.BM25(data[split]["pool"])
    S = bm.score_matrix(data[split]["queries"])
    out = pcr_data.rank(S, qids, bm.ids, min(top, len(bm.ids)))
    if path:
        json.dump(out, open(path, "w"))
    if log:
        log("bm25 ranking", split=split, top=top, seconds=round(time.time() - t0, 1))
    return out


def teacher_embeddings(cfg, data, arrays, device, amp, vocab_size, log):
    """teacher embeddings of every train query and train-pool case (teacher's own chunking), cached in the teacher dir."""
    tdir = cfg["kd"]["teacher_run"]
    qids = [d["id"] for d in data["train"]["queries"]]
    pids = [d["id"] for d in data["train"]["pool"]]
    for path in (os.path.join(tdir, "train_emb.npz"), os.path.join(cfg["out_dir"], cfg["name"], "teacher_train_emb.npz")):
        if os.path.exists(path):
            z = np.load(path)
            if list(z["qids"]) == qids and list(z["pids"]) == pids:
                log("teacher embeddings: cached", path=path)
                return {"q": z["q"], "p": z["p"]}
    teacher, tcfg = load_run(tdir, device, vocab_size)
    L = chunk_len(tcfg, teacher.backbone)
    im = tcfg["train"].get("infer_micro_docs") or 4
    t0 = time.time()
    qe = embed(teacher, make_items(data["train"]["queries"], arrays[("train", "queries")], tcfg, L), im, device, amp)
    pe = embed(teacher, make_items(data["train"]["pool"], arrays[("train", "pool")], tcfg, L), im, device, amp)
    emb = {"q": qe.cpu().numpy(), "p": pe.cpu().numpy()}
    path = os.path.join(tdir, "train_emb.npz")
    try:
        np.savez(path + ".tmp.npz", qids=np.array(qids), pids=np.array(pids), **emb)
        os.replace(path + ".tmp.npz", path)
    except OSError:                                   # teacher folder read-only (an attached Kaggle input)
        path = os.path.join(cfg["out_dir"], cfg["name"], "teacher_train_emb.npz")
        np.savez(path, qids=np.array(qids), pids=np.array(pids), **emb)
    log("teacher embeddings: computed", path=path, seconds=round(time.time() - t0, 1), teacher=tcfg["name"])
    del teacher
    torch.cuda.empty_cache()
    return emb


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(cfg_path):
    cfg = json.load(open(cfg_path))
    check_config(cfg)
    name, tr, ng = cfg["name"], cfg["train"], cfg["negatives"]
    out = os.path.join(cfg["out_dir"], name)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.jsonl"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    seed = tr.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    t_start = time.time()
    deadline = t_start + 60 * tr["max_minutes"] if tr.get("max_minutes") else None

    tok = ByteBPE.from_json(cfg["tokenizer"])
    data, arrays, drep = load_data(cfg, tok)
    smoke = cfg.get("smoke")            # {"steps": n, "queries": m}: real model, few queries, small pools
    if smoke:
        smoke_subset(data, arrays, smoke.get("queries", 8), seed)
        tr["epochs"] = 1

    model, tmod, info = build_model(cfg, tok.get_vocab_size())
    model.to(device)
    L = chunk_len(cfg, model.backbone)
    items = {k: make_items(data[k[0]][k[1]], arrays[k], cfg, L) for k in arrays}
    log("data", **{f"{s}_{k}": v for s in ("train", "dev", "test") for k, v in drep[s].items()},
        train_queries_equal_dev_queries=drep["train_queries_equal_dev_queries"],
        train_queries_equal_test_queries=drep["train_queries_equal_test_queries"], chunk_len=L,
        strategy=cfg["chunking"]["strategy"], max_chunks=cfg["chunking"]["max_chunks"],
        **{f"{s}_{p}_coverage": round(pcr_data.coverage(v), 4) for (s, p), v in items.items()})
    params = count_params(model)
    params["head"] = sum(p.numel() for m in model.head_modules() for p in m.parameters())
    log("model", **info, **params, chunk_len=L, max_len=model.backbone.max_len)

    tq, tp = data["train"]["queries"], data["train"]["pool"]
    pidx = {d["id"]: j for j, d in enumerate(tp)}
    pairs = [(i, pidx[c]) for i, q in enumerate(tq) for c in q["relevant"] if c in pidx]
    rel_idx = [{pidx[c] for c in q["relevant"] if c in pidx} for q in tq]
    self_idx = [pidx.get(q["id"]) for q in tq]
    assert pairs, "no (query, relevant candidate) pair found in the train pool"
    hard = None
    if ng["source"] == "bm25" and ng["per_positive"]:
        top = bm25_top(data, "train", int(ng["bm25_top"]) + 64, None if smoke else cfg.get("cache_dir"), log)
        hard = []
        for i, q in enumerate(tq):
            cand = [pidx[c] for c in top[q["id"]] if pidx[c] not in rel_idx[i] and pidx[c] != self_idx[i]]
            hard.append(cand[:int(ng["bm25_top"])])

    t_emb = None
    if cfg.get("kd"):
        kd = {k: v for k, v in cfg["kd"].items() if k != "teacher_run"}
        tcfg = json.load(open(os.path.join(cfg["kd"]["teacher_run"], "config.json")))
        tsum = json.load(open(os.path.join(cfg["kd"]["teacher_run"], "summary.json")))
        assert params["total"] < tsum["params"]["total"], \
            f"KD student ({params['total']:,}) must be smaller than the teacher ({tsum['params']['total']:,})"
        t_emb = teacher_embeddings(cfg, data, arrays, device, amp, tok.get_vocab_size(), log)
        t_emb = {k: torch.from_numpy(v).to(device) for k, v in t_emb.items()}
        kd_loss = techniques.get("kd").kd_loss
        log("teacher", run=tcfg["name"], params=tsum["params"]["total"], student_params=params["total"],
            teacher_dev_f1_at7=tsum.get("dev_f1_at7"))

    if hasattr(tmod, "param_groups"):
        groups = tmod.param_groups(model, tr["lr"])
    else:
        head_params = [p for m in model.head_modules() for p in m.parameters()]
        groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": tr["lr"]},
                  {"params": head_params, "lr": tr["head_lr"]}]
    for g in groups:
        g["params"] = list(g["params"])
        g["base_lr"] = g["lr"]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    extra_loss, on_step = getattr(tmod, "extra_loss", None), getattr(tmod, "on_step", None)

    rng = random.Random(seed)
    B, micro = tr["batch_pairs"], tr["micro_docs"]
    steps_per_epoch = math.ceil(len(pairs) / B)
    total = steps_per_epoch * tr["epochs"]
    warmup = int(tr["warmup_frac"] * total)
    log("schedule", pairs=len(pairs), steps_per_epoch=steps_per_epoch, total_steps=total, warmup_steps=warmup,
        pairs_per_step=B, negatives=ng)

    def sample_negatives(i):
        n = ng["per_positive"]
        if not n:
            return []
        if hard is not None and hard[i]:
            pool_ = hard[i]
        else:
            pool_ = [j for j in range(len(tp)) if j not in rel_idx[i] and j != self_idx[i]]
        return rng.sample(pool_, min(n, len(pool_)))

    def step_rows(batch):
        """-> query rows, candidate columns, target per row, allowed mask (P, M)."""
        cand, col = [], {}
        rows = []
        for qi, pj in batch:
            negs = sample_negatives(qi)
            for j in [pj] + negs:
                if j not in col:
                    col[j] = len(cand); cand.append(j)
            rows.append((qi, pj, negs))
        P, M = len(rows), len(cand)
        allowed = torch.zeros(P, M, dtype=torch.bool)
        target = torch.zeros(P, dtype=torch.long)
        for r, (qi, pj, negs) in enumerate(rows):
            if ng["in_batch"]:
                allowed[r] = True
            else:
                allowed[r, [col[j] for j in [pj] + negs]] = True
            for j in rel_idx[qi] | ({self_idx[qi]} if self_idx[qi] is not None else set()):
                if j != pj and j in col:
                    allowed[r, col[j]] = False
            allowed[r, col[pj]] = True
            target[r] = col[pj]
        return [qi for qi, _, _ in rows], cand, target, allowed

    sel = {"best": -1.0, "at": None, "epoch": -1}
    history = []
    im = tr.get("infer_micro_docs") or 4
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate(tag, epoch, rec):
        t_ev = time.time()
        S = score_matrix(model, items[("dev", "queries")], items[("dev", "pool")], cfg, im, device, amp)
        rep = pcr_data.report(S, [q["id"] for q in data["dev"]["queries"]], [d["id"] for d in data["dev"]["pool"]],
                              gold_of(data["dev"]["queries"]))
        rec.update(at=tag, dev_f1_at7=rep["at7"]["micro_f1"], dev_p_at7=rep["at7"]["micro_p"],
                   dev_r_at7=rep["at7"]["micro_r"], dev_r_at100=rep["recall@100"], eval_s=round(time.time() - t_ev, 1))
        if rep["at7"]["micro_f1"] > sel["best"]:
            sel.update(best=rep["at7"]["micro_f1"], at=tag, epoch=epoch)
            save_model(model, os.path.join(out, "best_model.pt"))
            rec["saved_best"] = True
        history.append(rec)
        model.train()
        return rec

    step, seen_tok, epochs_done, stop = 0, 0, 0, False
    t_train = time.time()
    eval_every = tr.get("eval_every_steps")
    if not smoke and tr.get("eval_at_start", True):
        log("eval", **evaluate("start", 0, dict(epoch=0, step=0)))        # Phase-1 representations, no PCR training
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        order = pairs[:]
        rng.shuffle(order)
        run_loss, n_loss, t_ep, complete = 0.0, 0, time.time(), True
        for a in range(0, len(order), B):
            batch = order[a:a + B]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, warmup, tr["schedule"])
            q_rows, cand, target, allowed = step_rows(batch)
            uq = sorted(set(q_rows))
            docs = [items[("train", "queries")][i] for i in uq] + [items[("train", "pool")][j] for j in cand]
            qpos = {qi: k for k, qi in enumerate(uq)}
            am, tgt = allowed.to(device), target.to(device)

            def step_loss(E):
                S = similarity(E[[qpos[qi] for qi in q_rows]], E[len(uq):], cfg["similarity"], cfg["temperature"])
                S = S.masked_fill(~am, MASK)
                if t_emb is not None:
                    St = similarity(t_emb["q"][q_rows], t_emb["p"][cand], tcfg["similarity"], tcfg["temperature"])
                    return kd_loss(S.unsqueeze(1), St.masked_fill(~am, MASK).unsqueeze(1), tgt.unsqueeze(1), **kd)
                return F.cross_entropy(S, tgt)

            loss, ntok = grad_cache_backward(model, docs, micro, device, amp, step_loss, scaler.scale)
            seen_tok += ntok
            if extra_loss is not None:
                scaler.scale(extra_loss(model)).backward()
            if tr["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tr["grad_clip"])
            scaler.step(opt); scaler.update()
            step += 1
            if on_step is not None:
                on_step(model, step, total)
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item(); n_loss += 1
            if smoke and step >= smoke["steps"]:
                stop = True
                break
            if eval_every and step % eval_every == 0 and step % steps_per_epoch != 0:
                log("eval", **evaluate(f"e{epoch}_step{step}", epoch,
                                       dict(epoch=epoch, step=step, train_loss=round(run_loss / max(1, n_loss), 4))))
            if step % tr.get("log_every", 20) == 0:
                el = time.time() - t_train
                log("train", epoch=epoch, step=step, of=total, loss=round(run_loss / max(1, n_loss), 4),
                    lr=opt.param_groups[0]["lr"], tok_s=round(seen_tok / el), docs_in_step=len(docs),
                    epoch_eta_min=round((steps_per_epoch * epoch - step) * el / step / 60, 1))
            if deadline and time.time() > deadline:
                log("max_minutes reached: stopping inside the epoch (this epoch is INCOMPLETE)", step=step)
                stop, complete = True, False
                break
        if complete and not smoke:
            epochs_done = epoch
        rec = dict(epoch=epoch, complete_epoch=complete and not smoke, step=step,
                   train_loss=round(run_loss / max(1, n_loss), 4), epoch_s=round(time.time() - t_ep, 1))
        log("epoch", **evaluate(f"e{epoch}_end", epoch, rec))
        if stop or (tr.get("patience") and epoch - sel["epoch"] >= tr["patience"]):
            break
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None

    model.load_state_dict(torch.load(os.path.join(out, "best_model.pt"), map_location="cpu"))
    dq, dpool = [q["id"] for q in data["dev"]["queries"]], [d["id"] for d in data["dev"]["pool"]]
    S_dev = score_matrix(model, items[("dev", "queries")], items[("dev", "pool")], cfg, im, device, amp)
    dev_rep = pcr_data.report(S_dev, dq, dpool, gold_of(data["dev"]["queries"]))
    json.dump(dev_rep, open(os.path.join(out, "dev_scores.json"), "w"), indent=2)
    np.savez(os.path.join(out, "dev_S.npz"), S=S_dev.astype(np.float32), qids=np.array(dq), cids=np.array(dpool))
    tq_ids, tpool = [q["id"] for q in data["test"]["queries"]], [d["id"] for d in data["test"]["pool"]]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    S_test = score_matrix(model, items[("test", "queries")], items[("test", "pool")], cfg, im, device, amp)
    inf_s = time.time() - t_inf
    np.savez(os.path.join(out, "test_S.npz"), S=S_test.astype(np.float32), qids=np.array(tq_ids), cids=np.array(tpool))
    pred_path = os.path.join(out, "prediction.json")
    pcr_data.write_prediction(S_test, tq_ids, tpool, pred_path)
    zip_path = os.path.join(out, f"submission_{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    n_test_docs = len(tq_ids) + len(tpool)
    summary = dict(name=name, model_key=cfg.get("model_key"), phase1=info, technique=cfg["technique"],
                   kd=bool(cfg.get("kd")), similarity=cfg["similarity"], temperature=cfg["temperature"],
                   proj_dim=cfg.get("proj_dim"), chunk_pool=cfg["chunk_pool"], doc_pool=cfg["doc_pool"],
                   chunking={**cfg["chunking"], "chunk_len": L}, negatives=ng, lr=tr["lr"], head_lr=tr["head_lr"],
                   params=params, selected_at=sel["at"], best_epoch=sel["epoch"], epochs_completed=epochs_done,
                   full_epoch_completed=epochs_done >= 1, history=history,
                   dev_f1_at7=dev_rep["at7"]["micro_f1"], dev=dev_rep,
                   start_dev_f1_at7=history[0]["dev_f1_at7"] if history and history[0]["at"] == "start" else None,
                   train_seconds=round(train_s, 1), train_tok_s=round(seen_tok / max(train_s, 1e-9)),
                   peak_mem_GB=round(peak, 2) if peak else None, test_infer_seconds=round(inf_s, 1),
                   test_docs_encoded=n_test_docs, infer_ms_per_doc=round(1000 * inf_s / max(1, n_test_docs), 1),
                   total_seconds=round(time.time() - t_start, 1), data=drep, smoke=bool(smoke))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    log("done", name=name, dev_f1_at7=dev_rep["at7"]["micro_f1"], dev_p_at7=dev_rep["at7"]["micro_p"],
        dev_r_at7=dev_rep["at7"]["micro_r"], selected_at=sel["at"], epochs_completed=epochs_done,
        train_min=round(train_s / 60, 1), peak_GB=summary["peak_mem_GB"], trainable=params["trainable"], zip=zip_path)


if __name__ == "__main__":
    main(sys.argv[1])
