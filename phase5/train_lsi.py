#!/usr/bin/env python3
"""Fine-tune one Phase-1 model for LSI (multi-label statute identification) and write its Codabench submission
(plumbing; declare it). Adapted from the Phase-4 CJPE trainer: same chunking / pooling / techniques / checkpoints;
the head has one output per statute (label space = all statute ids in train + dev, saved as labels.json), the loss is
binary cross-entropy per statute (optional pos_weight), probabilities are sigmoids, a document's prediction is every
statute with P >= threshold (0.5 here; tuned thresholds: lsi_select.py), and the metric is multi-label macro-F1.

  python train_lsi.py run_config.json

Pipeline: train/dev/test judgments -> your ByteBPE -> up to k chunks per document (lsi_data.chunk_spans: even |
headtail | overlap) -> Phase-1 backbone (backbones) + YOUR technique (techniques/<name>.py) -> one vector per chunk
(chunk_pool: mean | last over the chunk's tokens) -> one vector per document (doc_pool: attn = learned score per chunk,
softmax over the document's chunks, weighted sum | mean) -> YOUR head (heads.py, n_labels outputs) -> document-level
cross-entropy (optional class weights). One loss and one prediction per judgment; chunks are never separate examples.

Chunks of different lengths are never padded together: each micro-batch is run per chunk length, so no position ever
attends to padding (relevant for C's bidirectional encoder and D's bidirectional prefix).

Evaluation after every epoch on the released dev set (dev_all, 3,505 documents) and on its documents whose text is
not in test.json (dev_excl_test); the checkpoint is selected on cfg["select_on"] (default dev_all). Test labels are
never read.

KD (cfg["kd"] = {"teacher_run": <finished run dir>, ...your kd options}): the teacher's document logits on the
(de-duplicated) train set are computed once with the teacher's own chunking and cached in
<teacher_run>/train_logits.json (ids checked), then YOUR techniques/kd.py kd_loss(student, teacher, labels, **opts)
is the training loss. The student must have fewer parameters than the teacher (checked).

Outputs in <out_dir>/<name>/: config.json, log.jsonl, best_model.pt (saved after every epoch that improves the
selection score), dev_scores.json, dev_probs.json /
test_probs.json ({id: [P(statute) in labels.json order]}), prediction.json ([{"id", "pred": [statute ids]}], P >= 0.5),
labels.json, submission_<name>.zip, summary.json. Threshold tuning (global / per statute) and ensembles: lsi_select.py.
KD (multi-label): every statute is a 2-class problem, logits [0, z] -> softmax = sigmoid(z), so YOUR techniques/kd.py
kd_loss (temperature softmax KL + CE) is applied unchanged per (document, statute).
pos_weight (train.pos_weight): None | "inv_freq" (negatives/positives per statute) | "sqrt_inv_freq", capped at
train.pos_weight_max (default 50); statutes absent from train get 1.
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
import lsi_data                       # noqa: E402
import heads                           # noqa: E402
import techniques                      # noqa: E402
from a1_tokenizer import ByteBPE       # noqa: E402
from inject import count_params        # noqa: E402

REQUIRED = {
    "chunking": ["strategy", "max_chunks"],
    "train": ["epochs", "batch_docs", "micro_docs", "lr", "head_lr", "weight_decay", "warmup_frac", "schedule", "grad_clip", "seed"],
}


def check_config(cfg):
    missing = [f"{s}.{k}" for s, ks in REQUIRED.items() for k in ks if cfg.get(s, {}).get(k) is None]
    missing += [k for k in ("chunk_pool", "doc_pool") if cfg.get(k) is None]
    missing += [f"{s}.name" for s in ("head", "technique") if cfg.get(s, {}).get("name") is None]
    assert not missing, "fill these config values first: " + ", ".join(missing)
    ch, tr = cfg["chunking"], cfg["train"]
    assert ch["strategy"] in ("even", "headtail", "overlap")
    assert ch["strategy"] != "overlap" or ch.get("stride"), "chunking.stride is needed for overlap"
    assert cfg["chunk_pool"] in ("mean", "last") and cfg["doc_pool"] in ("attn", "mean")
    assert tr["batch_docs"] % tr["micro_docs"] == 0, "train.batch_docs must be a multiple of train.micro_docs"
    assert tr["schedule"] in ("linear", "cosine", "constant")
    assert tr.get("pos_weight") in (None, "inv_freq", "sqrt_inv_freq"), "train.pos_weight: None | inv_freq | sqrt_inv_freq"
    assert cfg.get("select_on", "dev_all") in ("dev_all", "dev_excl_test")
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
# model: backbone -> chunk vectors -> document vector -> head
# ---------------------------------------------------------------------------

class DocClassifier(nn.Module):
    def __init__(self, backbone, head, chunk_pool, doc_pool):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.chunk_pool, self.doc_pool = chunk_pool, doc_pool
        self.attn_score = nn.Linear(backbone.d_model, 1) if doc_pool == "attn" else None

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
        """items: list of {"chunks": [...]} -> logits (n_docs, n_labels)."""
        chunks = [c for it in items for c in it["chunks"]]
        vec = self.chunk_vectors(chunks, device)
        counts = [len(it["chunks"]) for it in items]
        K = max(counts)
        d = vec.shape[-1]
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
        return self.head(doc.unsqueeze(0)).squeeze(0)                   # (n_docs, n_labels)


def build_model(cfg, vocab_size, n_labels, load_weights=True):
    bb, spec, info = backbones.build_backbone(cfg["phase1_ckpt"], cfg.get("backbone", {}), vocab_size,
                                              load_weights=load_weights, fast_attention=cfg.get("fast_attention", True))
    for p in bb.parameters():
        p.requires_grad_(True)
    tech = cfg["technique"]
    tmod = techniques.get(tech["name"])
    tmod.apply(bb, **{k: v for k, v in tech.items() if k != "name"})
    head = heads.build_head(bb.d_model, n_labels, **cfg["head"])
    model = DocClassifier(bb, head, cfg["chunk_pool"], cfg["doc_pool"])
    for m in (model.head, model.attn_score):
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(True)
    return model, tmod, info


def chunk_len(cfg, backbone):
    limit = backbone.max_len - backbone.n_virtual_tokens          # prompt tuning prepends virtual tokens
    L = cfg["chunking"].get("chunk_len") or limit
    assert L <= limit, f"chunk_len {L} > model input limit {limit} (max_len {backbone.max_len} - {backbone.n_virtual_tokens} virtual tokens)"
    return L


def save_model(model, path):
    torch.save(model.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_run(run_dir, device, vocab_size):
    import glob
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if not os.path.exists(cfg["phase1_ckpt"]):               # earlier session: same checkpoint, new input path
        tail = os.path.join(*cfg["phase1_ckpt"].split(os.sep)[-2:])
        hits = sorted(glob.glob(os.path.join("/kaggle/input", "**", tail), recursive=True))
        assert hits, f"{run_dir}: Phase-1 checkpoint {cfg['phase1_ckpt']} not found (looked for */{tail})"
        cfg["phase1_ckpt"] = hits[0]
    n_labels = len(json.load(open(os.path.join(run_dir, "labels.json"))))
    model, _, _ = build_model(cfg, vocab_size, n_labels, load_weights=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu"))
    return model.to(device).eval(), cfg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data(cfg, tok, log=None):
    D, cache = cfg["data_dir"], cfg.get("cache_dir")
    train, rep = lsi_data.dedup_train(lsi_data.load_split(D, "train"))
    dev, test = lsi_data.load_split(D, "dev"), lsi_data.load_split(D, "test")
    n_overlap = lsi_data.mark_dev_excl_test(dev, test)
    # leakage guard: train documents whose text also occurs in dev or test are removed from train (reported)
    held_h = {lsi_data.norm_hash(d["text"]) for d in dev} | {lsi_data.norm_hash(d["text"]) for d in test}
    n_before = len(train)
    train = [d for d in train if lsi_data.norm_hash(d["text"]) not in held_h]
    rep["train_removed_overlapping_dev_or_test"] = n_before - len(train)
    rep["train_docs_used"] = len(train)
    rep.update(dev_docs=len(dev), dev_docs_in_test=n_overlap, dev_excl_test_docs=len(dev) - n_overlap, test_docs=len(test))
    tr_h = {lsi_data.norm_hash(d["text"]) for d in train}
    rep["train_dev_overlap"] = sum(lsi_data.norm_hash(d["text"]) in tr_h for d in dev)
    rep["train_test_overlap"] = sum(lsi_data.norm_hash(d["text"]) in tr_h for d in test)
    arrays = {s: lsi_data.encode_docs(tok, docs, os.path.join(cache, f"{s}_ids.npz") if cache else None)
              for s, docs in (("train", train), ("dev", dev), ("test", test))}
    for docs in (train, dev, test):                 # texts are no longer needed (token ids are cached): free the RAM
        for d in docs:
            d.pop("text", None)
    return {"train": train, "dev": dev, "test": test}, arrays, rep


def make_items(docs, arrays, cfg, L):
    ch = cfg["chunking"]
    return lsi_data.build_items(docs, arrays, L, ch["strategy"], ch["max_chunks"], ch.get("stride"))


def micro_batches(items, micro, shuffle, rng):
    """documents grouped by chunk count (similar cost together); shuffled across groups when training."""
    order = sorted(range(len(items)), key=lambda i: (len(items[i]["chunks"]), sum(len(c) for c in items[i]["chunks"])))
    mbs = [order[i:i + micro] for i in range(0, len(order), micro)]
    if shuffle:
        rng.shuffle(mbs)
    return mbs


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def dev_report(dev_items, probs, label_space, threshold=0.5):
    """probs: {id: [P per statute]} -> multi-label scores on dev_all, dev_excl_test and dev_in_test (dev rows whose
    text is also in test.json: their score equals the test score, so selecting on it uses the test labels)."""
    res = {}
    for key, sel in (("dev_all", lambda it: True), ("dev_excl_test", lambda it: not it.get("in_test", False)),
                     ("dev_in_test", lambda it: it.get("in_test", False))):
        its = [it for it in dev_items if sel(it)]
        if not its:
            continue
        gold = [it["labels"] for it in its]
        pred = [[l for l, p in zip(label_space, probs[it["id"]]) if p >= threshold] for it in its]
        res[key] = lsi_data.multilabel_scores(gold, pred, label_space)
        res[key].pop("per_label") if key != "dev_all" else None
    return res


# ---------------------------------------------------------------------------
# inference / training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_logits(model, items, micro, device, amp):
    """-> {id: [logit0, logit1]} (float). Rows repeating an id (dev has 994 exact duplicate rows) are predicted once."""
    model.eval()
    out, seen, uniq = {}, set(), []
    for it in items:
        if it["id"] not in seen:
            seen.add(it["id"]); uniq.append(it)
    items = uniq
    for idx in micro_batches(items, micro, False, None):
        batch = [items[i] for i in idx]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg = model(batch, device).float().cpu()
        for it, row in zip(batch, lg):
            out[it["id"]] = row.tolist()
    assert len(out) == len(items)
    return out


def probs_of(logits):
    return {k: torch.sigmoid(torch.tensor(v)).tolist() for k, v in logits.items()}


def lr_factor(step, total, warmup, schedule):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1.0 - prog if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * prog))


def teacher_train_logits(cfg, data, arrays, device, amp, vocab_size, log):
    """cached document logits of the KD teacher on the de-duplicated train set (teacher's own chunking)."""
    tdir = cfg["kd"]["teacher_run"]
    path = os.path.join(tdir, "train_logits.json")
    ids = [d["id"] for d in data["train"]]
    if os.path.exists(path):
        cached = json.load(open(path))
        if sorted(cached) == sorted(ids):
            log("teacher logits: cached", path=path)
            return cached
    teacher, tcfg = load_run(tdir, device, vocab_size)
    L = chunk_len(tcfg, teacher.backbone)
    t_items = make_items(data["train"], arrays["train"], tcfg, L)
    t0 = time.time()
    logits = predict_logits(teacher, t_items, tcfg["train"].get("infer_micro_docs") or 4, device, amp)
    try:
        json.dump(logits, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    except OSError:                                   # teacher folder read-only (e.g. an attached Kaggle input)
        path = os.path.join(cfg["out_dir"], cfg["name"], "teacher_train_logits.json")
        json.dump(logits, open(path, "w"))
    log("teacher logits: computed", path=path, docs=len(logits), seconds=round(time.time() - t0, 1),
        teacher=tcfg["name"], teacher_params=count_params(teacher)["total"])
    del teacher
    torch.cuda.empty_cache()
    return logits


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(cfg_path):
    cfg = json.load(open(cfg_path))
    check_config(cfg)
    name, tr = cfg["name"], cfg["train"]
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
    smoke = cfg.get("smoke")            # {"steps": n}: real model, real (longest) documents, few steps; not submittable
    if smoke:
        def longest(s, n):
            order = sorted(range(len(data[s])), key=lambda i: -len(arrays[s][i]))[:n]
            data[s] = [data[s][i] for i in order]
            arrays[s] = [arrays[s][i] for i in order]
        longest("train", smoke.get("docs", 16)); longest("dev", 4); longest("test", 4)
        tr["epochs"] = 1
    assert drep["train_dev_overlap"] == 0 and drep["train_test_overlap"] == 0 or smoke, f"train overlaps dev/test: {drep}"

    label_space = sorted({l for d in data["train"] + data["dev"] for l in d["labels"]})   # all statute ids (train + dev)
    lab_idx = {l: i for i, l in enumerate(label_space)}
    json.dump(label_space, open(os.path.join(out, "labels.json"), "w"))
    model, tmod, info = build_model(cfg, tok.get_vocab_size(), len(label_space))
    init_run = cfg.get("init_from_run")             # continue training a finished run (its best_model.pt); new out dir
    if init_run:
        assert os.path.abspath(init_run) != os.path.abspath(out), "init_from_run must not be this run's own folder"
        icfg = json.load(open(os.path.join(init_run, "config.json")))
        for k in ("technique", "head", "chunk_pool", "doc_pool", "backbone"):
            assert icfg.get(k) == cfg.get(k), f"init_from_run: '{k}' differs from {init_run} ({icfg.get(k)} vs {cfg.get(k)})"
        model.load_state_dict(torch.load(os.path.join(init_run, "best_model.pt"), map_location="cpu"))
        info["init_from_run"] = init_run
    model.to(device)
    L = chunk_len(cfg, model.backbone)
    items = {s: make_items(data[s], arrays[s], cfg, L) for s in ("train", "dev", "test")}
    train_chunking = cfg.get("train_chunking")      # optional per-epoch chunking for TRAIN only: [{strategy, stride}, ...]
    if train_chunking:
        for tc in train_chunking:
            assert tc["strategy"] in ("even", "headtail", "overlap") and (tc["strategy"] != "overlap" or tc.get("stride"))
    log("data", **drep, chunk_len=L, strategy=cfg["chunking"]["strategy"], max_chunks=cfg["chunking"]["max_chunks"],
        stride=cfg["chunking"].get("stride"),
        **{f"{s}_chunks_per_doc": round(float(np.mean([len(x["chunks"]) for x in items[s]])), 3) for s in items},
        **{f"{s}_tokens": int(sum(len(c) for x in items[s] for c in x["chunks"])) for s in items},
        **{f"{s}_coverage": round(lsi_data.coverage(items[s]), 4) for s in items},
        n_labels=len(label_space), labels_only_in_dev=len(set(label_space) - {l for d in data["train"] for l in d["labels"]}),
        mean_train_labels_per_doc=round(float(np.mean([len(x["labels"]) for x in items["train"]])), 3))
    params = count_params(model)
    params["head"] = sum(p.numel() for p in model.head.parameters()) + (
        sum(p.numel() for p in model.attn_score.parameters()) if model.attn_score is not None else 0)
    log("model", **info, **params, chunk_len=L, max_len=model.backbone.max_len)

    t_logits = None
    if cfg.get("kd"):
        kd = {k: v for k, v in cfg["kd"].items() if k != "teacher_run"}
        tcfg = json.load(open(os.path.join(cfg["kd"]["teacher_run"], "config.json")))
        tsum = json.load(open(os.path.join(cfg["kd"]["teacher_run"], "summary.json")))
        assert params["total"] < tsum["params"]["total"], \
            f"KD student ({params['total']:,}) must be smaller than the teacher ({tsum['params']['total']:,})"
        t_logits = teacher_train_logits(cfg, data, arrays, device, amp, tok.get_vocab_size(), log)
        missing = [it["id"] for it in items["train"] if it["id"] not in t_logits]
        assert not missing, f"teacher logits missing for {len(missing)} train documents"
        t_labels = json.load(open(os.path.join(cfg["kd"]["teacher_run"], "labels.json")))
        assert t_labels == label_space, "KD teacher and student must share the statute label space (labels.json)"
        kd_loss = techniques.get("kd").kd_loss
        log("teacher", run=tcfg["name"], params=tsum["params"]["total"], student_params=params["total"],
            teacher_dev_macro_f1=tsum.get("dev_macro_f1"))

    # optimizer: backbone (technique) params at train.lr, head + attention pooling at train.head_lr
    if hasattr(tmod, "param_groups"):
        groups = tmod.param_groups(model, tr["lr"])
    else:
        head_params = list(model.head.parameters()) + (list(model.attn_score.parameters()) if model.attn_score is not None else [])
        groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": tr["lr"]},
                  {"params": head_params, "lr": tr["head_lr"]}]
    for g in groups:
        g["params"] = list(g["params"])
        g["base_lr"] = g["lr"]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pos_weight = None
    if tr.get("pos_weight"):
        pos = np.zeros(len(label_space))
        for it in items["train"]:
            pos[[lab_idx[l] for l in it["labels"]]] += 1
        n_tr = len(items["train"])
        pw = np.where(pos > 0, (n_tr - pos) / np.maximum(pos, 1), 1.0)
        pw = np.sqrt(pw) if tr["pos_weight"] == "sqrt_inv_freq" else pw
        pw = np.minimum(pw, (tr.get("pos_weight_max") or 50.0))
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        log("pos_weight", mode=tr["pos_weight"], min=round(float(pw.min()), 3), median=round(float(np.median(pw)), 3),
            max=round(float(pw.max()), 3))

    def multi_hot(batch):
        y = torch.zeros(len(batch), len(label_space), device=device)
        for r, it in enumerate(batch):
            y[r, [lab_idx[l] for l in it["labels"]]] = 1.0
        return y
    extra_loss, on_step = getattr(tmod, "extra_loss", None), getattr(tmod, "on_step", None)

    rng = random.Random(seed)
    micro, accum = tr["micro_docs"], tr["batch_docs"] // tr["micro_docs"]
    steps_per_epoch = math.ceil(len(micro_batches(items["train"], micro, False, None)) / accum)
    total = steps_per_epoch * tr["epochs"]
    warmup = int(tr["warmup_frac"] * total)
    log("schedule", steps_per_epoch=steps_per_epoch, total_steps=total, warmup_steps=warmup, docs_per_step=tr["batch_docs"])

    sel_key = cfg.get("select_on", "dev_all")
    best, best_epoch, step, seen_tok, epochs_done = -1.0, -1, 0, 0, 0
    im = tr.get("infer_micro_docs") or 4
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train, stop, history = time.time(), False, []
    eval_every = tr.get("eval_every_steps")
    best_at = None
    save_keys = cfg.get("save_best_on") or []           # extra best checkpoints: best_model_<key>.pt, e.g. dev_all
    best_extra = {k: {"score": -1.0, "at": None} for k in save_keys}

    def evaluate(tag, epoch, rec):
        nonlocal best, best_epoch, best_at
        t_ev = time.time()
        rep_ = dev_report(items["dev"], probs_of(predict_logits(model, items["dev"], im, device, amp)), label_space)
        rec.update(at=tag, **{k: rep_[k]["macro_f1"] for k in rep_},
                   micro_f1=rep_[sel_key]["micro_f1"], macro_p=rep_[sel_key]["macro_p"], macro_r=rep_[sel_key]["macro_r"],
                   eval_s=round(time.time() - t_ev, 1))
        if rep_[sel_key]["macro_f1"] > best:
            best, best_epoch, best_at = rep_[sel_key]["macro_f1"], epoch, tag
            save_model(model, os.path.join(out, "best_model.pt"))
            rec["saved_best"] = True
        for k in save_keys:
            if k in rep_ and rep_[k]["macro_f1"] > best_extra[k]["score"]:
                best_extra[k] = {"score": rep_[k]["macro_f1"], "at": tag}
                save_model(model, os.path.join(out, f"best_model_{k}.pt"))
                rec.setdefault("saved", []).append(k)
        history.append(rec)
        model.train()
        return rec

    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        if train_chunking:
            tc = train_chunking[(epoch - 1) % len(train_chunking)]
            items["train"] = lsi_data.build_items(data["train"], arrays["train"], L, tc["strategy"],
                                                   cfg["chunking"]["max_chunks"], tc.get("stride"))
            log("train chunking", epoch=epoch, **tc, coverage=round(lsi_data.coverage(items["train"]), 4))
        mbs = micro_batches(items["train"], micro, True, rng)
        run_loss, n_loss, t_ep, complete = 0.0, 0, time.time(), True
        for i in range(0, len(mbs), accum):
            group = mbs[i:i + accum]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, warmup, tr["schedule"])
            n_docs = sum(len(idx) for idx in group)
            for idx in group:
                batch = [items["train"][j] for j in idx]
                y = multi_hot(batch)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(batch, device)
                logits = logits.float()
                if t_logits is not None:
                    tl = torch.tensor([t_logits[it["id"]] for it in batch], dtype=torch.float32, device=device)
                    two = lambda z: torch.stack([torch.zeros_like(z), z], -1).view(1, -1, 2)   # per statute: [0, z]
                    loss = kd_loss(two(logits), two(tl), y.long().view(1, -1), **kd)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
                if extra_loss is not None:
                    loss = loss + extra_loss(model)
                scaler.scale(loss * len(idx) / n_docs).backward()       # mean over the documents of the step
                run_loss += loss.item() * len(idx); n_loss += len(idx)
                seen_tok += sum(len(c) for it in batch for c in it["chunks"])
            if tr["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tr["grad_clip"])
            scaler.step(opt); scaler.update()
            step += 1
            if on_step is not None:          # before zero_grad: your hook can still read p.grad (AdaLoRA importance)
                on_step(model, step, total)
            opt.zero_grad(set_to_none=True)
            if smoke and step >= smoke["steps"]:
                stop = True
                break
            if eval_every and step % eval_every == 0 and step % steps_per_epoch != 0:
                log("eval", **evaluate(f"e{epoch}_step{step}", epoch, dict(epoch=epoch, step=step,
                                                                           train_loss=round(run_loss / max(1, n_loss), 4))))
            if step % tr.get("log_every", 50) == 0:
                el = time.time() - t_train
                log("train", epoch=epoch, step=step, of=total, loss=round(run_loss / max(1, n_loss), 4),
                    lr=opt.param_groups[0]["lr"], tok_s=round(seen_tok / el),
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
        if stop or (tr.get("patience") and epoch - best_epoch >= tr["patience"]):
            break
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None

    model.load_state_dict(torch.load(os.path.join(out, "best_model.pt"), map_location="cpu"))
    dev_logits = predict_logits(model, items["dev"], im, device, amp)
    dev_probs = probs_of(dev_logits)
    dev_sc = dev_report(items["dev"], dev_probs, label_space)
    json.dump(dev_sc, open(os.path.join(out, "dev_scores.json"), "w"), indent=2)
    t_inf = time.time()
    test_logits = predict_logits(model, items["test"], im, device, amp)
    inf_s = time.time() - t_inf
    test_probs = probs_of(test_logits)
    json.dump(dev_probs, open(os.path.join(out, "dev_probs.json"), "w"))
    json.dump(test_probs, open(os.path.join(out, "test_probs.json"), "w"))
    pred = [{"id": it["id"], "pred": [int(l) for l, p in zip(label_space, test_probs[it["id"]]) if p >= 0.5]} for it in items["test"]]
    assert len(pred) == len(items["test"]) and len({p["id"] for p in pred}) == len(pred)
    assert all(isinstance(p["pred"], list) and len(set(p["pred"])) == len(p["pred"]) for p in pred)
    pred_path = os.path.join(out, "prediction.json")
    json.dump(pred, open(pred_path, "w"))
    zip_path = os.path.join(out, f"submission_{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    summary = dict(name=name, phase1=info, technique=cfg["technique"], head=cfg["head"], chunk_pool=cfg["chunk_pool"],
                   doc_pool=cfg["doc_pool"], chunking={**cfg["chunking"], "chunk_len": L}, kd=bool(cfg.get("kd")),
                   lr=tr["lr"], head_lr=tr["head_lr"], pos_weight=tr.get("pos_weight"), params=params,
                   select_on=sel_key, best_epoch=best_epoch, epochs_completed=epochs_done,
                   full_epoch_completed=epochs_done >= 1, history=history,
                   dev_macro_f1=dev_sc[sel_key]["macro_f1"], dev_all=dev_sc["dev_all"], dev_excl_test=dev_sc["dev_excl_test"],
                   train_seconds=round(train_s, 1), train_tok_s=round(seen_tok / max(train_s, 1e-9)),
                   train_tokens_per_epoch=int(sum(len(c) for x in items['train'] for c in x['chunks'])), peak_mem_GB=round(peak, 2) if peak else None,
                   test_infer_seconds=round(inf_s, 1), test_docs=len(items["test"]),
                   total_seconds=round(time.time() - t_start, 1), data=drep, smoke=bool(smoke))
    extra_out = {}
    cache = {best_at: (dev_probs, test_probs)}      # predictions per checkpoint (same checkpoint -> no re-prediction)
    for k in save_keys:                                # one folder + submission per extra best checkpoint
        ck = os.path.join(out, f"best_model_{k}.pt")
        if not os.path.exists(ck):
            continue
        kd_ = os.path.join(out, f"best_{k}"); os.makedirs(kd_, exist_ok=True)
        if best_extra[k]["at"] in cache:
            dp_, tp_ = cache[best_extra[k]["at"]]
        else:
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            dp_ = probs_of(predict_logits(model, items["dev"], im, device, amp))
            tp_ = probs_of(predict_logits(model, items["test"], im, device, amp))
            cache[best_extra[k]["at"]] = (dp_, tp_)
        sc_ = dev_report(items["dev"], dp_, label_space)
        json.dump(dp_, open(os.path.join(kd_, "dev_probs.json"), "w")); json.dump(tp_, open(os.path.join(kd_, "test_probs.json"), "w"))
        json.dump(sc_, open(os.path.join(kd_, "dev_scores.json"), "w"), indent=2)
        pr_ = [{"id": it["id"], "pred": [int(l) for l, p in zip(label_space, tp_[it["id"]]) if p >= 0.5]} for it in items["test"]]
        assert len(pr_) == len(items["test"]) and len({x["id"] for x in pr_}) == len(pr_)
        json.dump(pr_, open(os.path.join(kd_, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(kd_, f"submission_{name}_best_{k}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(kd_, "prediction.json"), arcname="prediction.json")
        extra_out[k] = {"selected_at": best_extra[k]["at"], "selection_score": best_extra[k]["score"],
                        **{kk: v["macro_f1"] for kk, v in sc_.items()}}
        log("best checkpoint", key=k, **extra_out[k])
    summary["best_per_key"] = extra_out
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    log("done", name=name, dev_all=dev_sc["dev_all"]["macro_f1"], dev_excl_test=dev_sc["dev_excl_test"]["macro_f1"],
        best_epoch=best_epoch, epochs_completed=epochs_done, train_min=round(train_s / 60, 1),
        peak_GB=summary["peak_mem_GB"], trainable=params["trainable"], zip=zip_path)


if __name__ == "__main__":
    main(sys.argv[1])
