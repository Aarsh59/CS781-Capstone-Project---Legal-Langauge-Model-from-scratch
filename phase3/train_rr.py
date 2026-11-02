#!/usr/bin/env python3
"""Fine-tune one Phase-1 model for Rhetorical Role prediction and write its Codabench submission (plumbing; declare it).

  python train_rr.py run_config.json

Pipeline: CL/IT documents -> sentences encoded with your ByteBPE -> context windows (rr_data.build_windows)
-> Phase-1 backbone (backbones) + YOUR technique (techniques/<name>.py) -> one pooled vector per target sentence
-> YOUR head (heads.py) -> one label per sentence. AdamW (fp16 autocast on GPU), macro-F1 on the official CL/IT dev
sets after every epoch, best checkpoint -> dev report -> test predictions for CL_test + IT_test -> prediction.json + zip.

Outputs in <out_dir>/<name>/: config.json, labels.json, log.jsonl, best_model.pt, dev_scores.json (all / CL / IT),
dev_predictions.json, prediction.json, submission_<name>.zip, summary.json (scores, params, memory, timings),
dev_probs.json / test_probs.json: {"labels": [label order], "decode": "argmax" | "head.decode",
  "probs": {doc_id: [[p_label0, ..., p_label12] per sentence, same order as the predictions]}} = softmax of the head's
  per-sentence logits of the best model (for soft voting / thresholds; with a CRF head the predictions come from
  head.decode (Viterbi), so they need not equal the argmax of these probabilities).

Head contract (your heads.py, unchanged from Phase 2): build_head(d_model, n_labels, name, **cfg); head(h) maps
(1, N, d_model) sentence vectors -> (1, N, n_labels) logits. Optional head.loss(logits, labels) / head.decode(logits)
are used per window (training) and per document (prediction), so a CRF head runs over the sentence sequence.

Technique contract (techniques/<name>.py): apply(backbone, **cfg) as in Phase 2, plus OPTIONAL hooks:
  extra_loss(model) -> scalar added to the training loss every micro-batch      (e.g. AdaLoRA's orthogonality term)
  on_step(model, step, total_steps) -> None, called after every optimizer step, BEFORE gradients are zeroed
                                                                                 (e.g. AdaLoRA's rank budget update)
  param_groups(model, base_lr) -> optimizer param groups                          (as in Phase 2)
"""
import glob
import inspect
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backbones                       # noqa: E402
import heads                           # noqa: E402
import rr_data                         # noqa: E402
import techniques                      # noqa: E402
from a1_tokenizer import ByteBPE       # noqa: E402
from inject import count_params        # noqa: E402
from rr_metrics import rr_scores       # noqa: E402

DOMAINS = ("CL", "IT")
REQUIRED = {
    "context": ["window_len", "max_sent_tokens", "target_tokens", "left_tokens", "right_tokens", "sep_token"],
    "train": ["epochs", "batch_size", "micro_batch", "lr", "weight_decay", "warmup_frac", "schedule", "grad_clip"],
}


def check_config(cfg):
    missing = [f"{sec}.{k}" for sec, keys in REQUIRED.items() for k in keys if cfg.get(sec, {}).get(k) is None]
    for key in ("pooling",):
        if cfg.get(key) is None:
            missing.append(key)
    if cfg.get("head", {}).get("name") is None:
        missing.append("head.name")
    if cfg.get("technique", {}).get("name") is None:
        missing.append("technique.name")
    assert not missing, "fill these config values first: " + ", ".join(missing)
    c, tr = cfg["context"], cfg["train"]
    assert cfg["pooling"] in ("mean", "last", "first", "max"), "pooling: mean | last | first | max"
    assert c["max_sent_tokens"] <= c["window_len"], "max_sent_tokens must be <= window_len"
    assert tr["batch_size"] % tr["micro_batch"] == 0, "train.batch_size must be a multiple of train.micro_batch"
    assert tr["schedule"] in ("linear", "cosine", "constant")
    assert set(cfg.get("train_domains", DOMAINS)) <= set(DOMAINS)
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
# model: backbone -> pooled target-sentence vectors -> head
# ---------------------------------------------------------------------------

def pool(h, rows, starts, ends, how):
    """h: (B, T, d); one vector per (row, start, end) span -> (N, d)."""
    out = []
    for r, s, e in zip(rows, starts, ends):
        seg = h[r, s:e]
        if how == "mean":
            out.append(seg.mean(0))
        elif how == "last":
            out.append(seg[-1])
        elif how == "first":
            out.append(seg[0])
        else:
            out.append(seg.max(0).values)
    return torch.stack(out)


class RRModel(nn.Module):
    def __init__(self, backbone, head, pooling):
        super().__init__()
        self.backbone, self.head, self.pooling = backbone, head, pooling

    def sentence_logits(self, ids, rows, starts, ends):
        h = self.backbone(ids)
        vec = pool(h, rows, starts, ends, self.pooling)
        return self.head(vec.unsqueeze(0)).squeeze(0)          # (N, n_labels)


def build_model(cfg, n_labels, vocab_size, load_weights=True):
    bb, spec, info = backbones.build_backbone(cfg["phase1_ckpt"], cfg.get("backbone", {}), vocab_size,
                                              load_weights=load_weights, fast_attention=cfg.get("fast_attention", True))
    for p in bb.parameters():
        p.requires_grad_(True)
    tech = cfg["technique"]
    tmod = techniques.get(tech["name"])
    tmod.apply(bb, **{k: v for k, v in tech.items() if k != "name"})
    if not cfg.get("fast_attention", True):           # the prefix_kv hook lives only in the fused attention
        assert not any(hasattr(m, "prefix_kv") for m in bb.modules()), "prefix tuning needs fast_attention: true"
    head = heads.build_head(bb.d_model, n_labels, **cfg["head"])
    for p in head.parameters():
        p.requires_grad_(True)
    return RRModel(bb, head, cfg["pooling"]), tmod, info


def save_model(model, path):
    torch.save(model.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_run(run_dir, device, vocab_size):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    labels = json.load(open(os.path.join(run_dir, "labels.json")))
    if not os.path.exists(cfg["phase1_ckpt"]):       # run from an earlier session: same checkpoint, new input path
        tail = os.path.join(*cfg["phase1_ckpt"].split(os.sep)[-2:])          # e.g. run_B2/best.pt
        hits = sorted(glob.glob(os.path.join("/kaggle/input", "**", tail), recursive=True))
        assert hits, f"{run_dir}: Phase-1 checkpoint {cfg['phase1_ckpt']} not found (attach it; looked for */{tail})"
        cfg["phase1_ckpt"] = hits[0]
    model, _, _ = build_model(cfg, len(labels), vocab_size, load_weights=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu"))
    return model.to(device).eval(), cfg, labels


# ---------------------------------------------------------------------------
# data -> windows -> micro-batches
# ---------------------------------------------------------------------------

def prepare(docs, tok, ctx, lab2id):
    """-> list of windows {"doc", "ids", "targets": [(sent_idx, s, e)], "labels": [...]}, checked complete."""
    out = []
    for d in docs:
        sent_ids = rr_data.encode_doc(tok, d, ctx["max_sent_tokens"])
        wins = rr_data.build_windows(sent_ids, ctx)
        rr_data.check_windows(len(sent_ids), wins, ctx["window_len"])
        for w in wins:
            w["doc"] = d["id"]
            w["labels"] = [lab2id[d["labels"][si]] for si, _, _ in w["targets"]] if d["labels"] is not None else None
            out.append(w)
    return out


def batches(wins, micro, shuffle, rng):
    """Length-bucketed micro-batches (similar lengths together -> little padding)."""
    order = sorted(range(len(wins)), key=lambda i: len(wins[i]["ids"]))
    chunks = [order[i:i + micro] for i in range(0, len(order), micro)]
    if shuffle:
        rng.shuffle(chunks)
    return chunks


def collate(wins, idx, device):
    L = max(len(wins[i]["ids"]) for i in idx)
    ids = torch.full((len(idx), L), rr_data.PAD_ID, dtype=torch.long)
    rows, starts, ends, labels, per_win = [], [], [], [], []
    for r, i in enumerate(idx):
        w = wins[i]
        ids[r, :len(w["ids"])] = torch.tensor(w["ids"])
        per_win.append(len(w["targets"]))
        for k, (_, s, e) in enumerate(w["targets"]):
            rows.append(r); starts.append(s); ends.append(e)
        if w["labels"] is not None:
            labels.extend(w["labels"])
    lab = torch.tensor(labels, dtype=torch.long, device=device) if labels else None
    return ids.to(device), rows, starts, ends, lab, per_win


def class_weights(wins, n, mode, none_weight, none_id):
    if mode is None and none_weight is None:
        return None
    w = torch.ones(n)
    if mode is not None:
        c = np.bincount([l for x in wins for l in x["labels"]], minlength=n).astype(np.float64)
        c = np.maximum(c, 1.0)
        w = torch.tensor((c.sum() / c) ** {"inv_freq": 1.0, "inv_sqrt": 0.5}[mode], dtype=torch.float32)
        w = w / w.mean()
    if none_weight is not None and none_id is not None:
        w[none_id] = float(none_weight)
    return w


@torch.no_grad()
def predict(model, wins, docs, micro, device, amp, id2lab, probs_out=None):
    """-> {doc_id: [label, ...]} with exactly one label per sentence, in order.
    probs_out (optional dict) is filled with {doc_id: [[softmax probabilities per class] per sentence]}."""
    model.eval()
    per_doc = {d["id"]: [None] * len(d["sentences"]) for d in docs}
    for idx in batches(wins, micro, False, None):
        ids, rows, starts, ends, _, _ = collate(wins, idx, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg = model.sentence_logits(ids, rows, starts, ends).float().cpu()
        k = 0
        for i in idx:
            for si, _, _ in wins[i]["targets"]:
                per_doc[wins[i]["doc"]][si] = lg[k]
                k += 1
    dec = getattr(model.head, "decode", None)
    out = {}
    for d in docs:
        L = torch.stack(per_doc[d["id"]])
        if probs_out is not None:
            probs_out[d["id"]] = torch.softmax(L, dim=-1).tolist()
        tags = dec(L.unsqueeze(0).to(device)).squeeze(0).cpu().tolist() if dec is not None else L.argmax(-1).tolist()
        out[d["id"]] = [id2lab[int(t)] for t in tags]
        assert len(out[d["id"]]) == len(d["sentences"])
    return out


def lr_factor(step, total, warmup, schedule):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1.0 - prog if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * prog))


def score_domains(gold_docs, pred, labels):
    res = {"all": rr_scores({d["id"]: d["labels"] for d in gold_docs}, pred, labels)}
    for dom in DOMAINS:
        g = {d["id"]: d["labels"] for d in gold_docs if d["domain"] == dom}
        if g:
            res[dom] = rr_scores(g, {k: pred[k] for k in g}, labels)
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(cfg_path):
    cfg = json.load(open(cfg_path))
    check_config(cfg)
    name, tr, ctx = cfg["name"], cfg["train"], cfg["context"]
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

    # ---- data
    tok = ByteBPE.from_json(cfg["tokenizer"])
    labels = rr_data.load_labels(cfg["data_dir"])
    lab2id, id2lab = {l: i for i, l in enumerate(labels)}, dict(enumerate(labels))
    json.dump(labels, open(os.path.join(out, "labels.json"), "w"))
    train_docs = [d for dom in cfg.get("train_domains", DOMAINS) for d in rr_data.load_domain(cfg["data_dir"], dom, "train")]
    dev_docs = [d for dom in DOMAINS for d in rr_data.load_domain(cfg["data_dir"], dom, "dev")]
    test_docs = [d for dom in DOMAINS for d in rr_data.load_domain(cfg["data_dir"], dom, "test")]
    assert len({d["id"] for d in test_docs}) == len(test_docs), "test document ids collide between CL and IT"
    smoke = cfg.get("smoke")                 # {"steps": n}: real model + real windows, tiny data (not submittable)
    if smoke:
        train_docs = train_docs[:4]
        dev_docs = [d for dom in DOMAINS for d in [x for x in dev_docs if x["domain"] == dom][:1]]
        test_docs = [d for dom in DOMAINS for d in [x for x in test_docs if x["domain"] == dom][:1]]
        tr["epochs"] = 1
    unknown = sorted({l for d in train_docs + dev_docs for l in d["labels"]} - set(labels))
    assert not unknown, f"labels not in label_vocab.json: {unknown}"
    tr_w, dev_w, te_w = prepare(train_docs, tok, ctx, lab2id), prepare(dev_docs, tok, ctx, lab2id), prepare(test_docs, tok, ctx, lab2id)
    log("data", train_docs=len(train_docs), dev_docs=len(dev_docs), test_docs=len(test_docs),
        train_sentences=sum(len(d["sentences"]) for d in train_docs), train_windows=len(tr_w),
        mean_targets_per_window=round(np.mean([len(w["targets"]) for w in tr_w]), 2),
        mean_window_tokens=round(np.mean([len(w["ids"]) for w in tr_w]), 1), labels=len(labels))

    # ---- model
    model, tmod, info = build_model(cfg, len(labels), tok.get_vocab_size())
    model.to(device)
    P = model.backbone.n_virtual_tokens
    assert ctx["window_len"] + P <= model.backbone.max_len, \
        f"window_len {ctx['window_len']} + {P} virtual tokens > backbone max_seq_len {model.backbone.max_len}"
    params = count_params(model)
    params["head"] = sum(p.numel() for p in model.head.parameters())
    log("model", **info, **params, virtual_tokens=P)

    teacher = None
    if cfg.get("kd"):
        kd = dict(cfg["kd"])
        teacher, tcfg, tlabels = load_run(kd.pop("teacher_run"), device, tok.get_vocab_size())
        assert tlabels == labels, "teacher and student must share label_vocab"
        assert ctx["window_len"] + teacher.backbone.n_virtual_tokens <= teacher.backbone.max_len
        t_par = count_params(teacher)["total"]
        assert params["total"] < t_par, f"KD student ({params['total']:,}) must be smaller than the teacher ({t_par:,})"
        for p in teacher.parameters():
            p.requires_grad_(False)
        kd_loss = techniques.get("kd").kd_loss
        pass_w = kd.pop("pass_class_weights", False)
        log("teacher", run=tcfg["name"], params=t_par, student_params=params["total"])

    # ---- optimizer
    if hasattr(tmod, "param_groups"):
        groups = tmod.param_groups(model, tr["lr"])
    else:
        groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": tr["lr"]},
                  {"params": list(model.head.parameters()), "lr": tr.get("head_lr") or tr["lr"]}]
    for g in groups:
        g["params"] = list(g["params"])
        g["base_lr"] = g["lr"]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cw = class_weights(tr_w, len(labels), tr.get("class_weights"), tr.get("none_weight"), lab2id.get("None"))
    if cw is not None:
        cw = cw.to(device)
        log("class_weights", weights={l: round(float(x), 3) for l, x in zip(labels, cw)})
    if teacher is not None and pass_w:
        assert cw is not None, "kd.pass_class_weights needs train.class_weights"
        kd["weight"] = cw
    extra_loss, on_step = getattr(tmod, "extra_loss", None), getattr(tmod, "on_step", None)

    rng = random.Random(seed)
    micro, accum = tr["micro_batch"], tr["batch_size"] // tr["micro_batch"]
    steps_per_epoch = math.ceil(len(batches(tr_w, micro, False, None)) / accum)
    total = steps_per_epoch * tr["epochs"]
    warmup = int(tr["warmup_frac"] * total)
    log("schedule", steps_per_epoch=steps_per_epoch, total_steps=total, warmup_steps=warmup)

    best, best_epoch, bad, step, seen = -1.0, -1, 0, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train, stop = time.time(), False
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        mbs = batches(tr_w, micro, True, rng)
        run_loss, n_loss, t_ep = 0.0, 0, time.time()
        for i in range(0, len(mbs), accum):
            group = mbs[i:i + accum]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, warmup, tr["schedule"])
            for idx in group:
                ids, rows, starts, ends, lab, per_win = collate(tr_w, idx, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model.sentence_logits(ids, rows, starts, ends)
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher.sentence_logits(ids, rows, starts, ends)
                logits = logits.float()
                if teacher is not None:
                    loss = kd_loss(logits.unsqueeze(0), t_logits.float().unsqueeze(0), lab.unsqueeze(0), **kd)
                elif hasattr(model.head, "loss"):                 # e.g. CRF: one sequence per window
                    parts, k = [], 0
                    for n in per_win:
                        parts.append(model.head.loss(logits[k:k + n].unsqueeze(0), lab[k:k + n].unsqueeze(0)))
                        k += n
                    loss = torch.stack(parts).mean()
                else:
                    loss = F.cross_entropy(logits, lab, weight=cw)
                if extra_loss is not None:
                    loss = loss + extra_loss(model)
                scaler.scale(loss / len(group)).backward()
                run_loss += loss.item(); n_loss += 1; seen += len(rows)
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
            if step % tr.get("log_every", 20) == 0:
                log("train", epoch=epoch, step=step, loss=round(run_loss / max(1, n_loss), 4),
                    lr=opt.param_groups[0]["lr"], sent_s=round(seen / (time.time() - t_train)))
            if deadline and time.time() > deadline:
                log("deadline reached: stopping training", step=step)
                stop = True
                break
        rec = dict(epoch=epoch, step=step, train_loss=round(run_loss / max(1, n_loss), 4), epoch_s=round(time.time() - t_ep, 1))
        t_ev = time.time()
        sc = score_domains(dev_docs, predict(model, dev_w, dev_docs, tr.get("infer_micro_batch") or micro * 2, device, amp, id2lab), labels)
        sel = sc["all"]["macro_f1"]
        rec.update(dev_macro_f1=sel, dev_CL=sc.get("CL", {}).get("macro_f1"), dev_IT=sc.get("IT", {}).get("macro_f1"),
                   eval_s=round(time.time() - t_ev, 1))
        if sel > best:
            best, best_epoch, bad = sel, epoch, 0
            save_model(model, os.path.join(out, "best_model.pt"))
            rec["saved"] = True
        else:
            bad += 1
        log("epoch", **rec)
        if stop or (tr.get("patience") and bad >= tr["patience"]):
            break
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None

    # ---- best model -> dev report + test predictions
    model.load_state_dict(torch.load(os.path.join(out, "best_model.pt"), map_location="cpu"))
    im = tr.get("infer_micro_batch") or micro * 2
    dev_probs, test_probs = {}, {}
    dev_pred = predict(model, dev_w, dev_docs, im, device, amp, id2lab, probs_out=dev_probs)
    dev_sc = score_domains(dev_docs, dev_pred, labels)
    json.dump(dev_sc, open(os.path.join(out, "dev_scores.json"), "w"), indent=2)
    json.dump(dev_pred, open(os.path.join(out, "dev_predictions.json"), "w"))
    t_inf = time.time()
    test_pred = predict(model, te_w, test_docs, im, device, amp, id2lab, probs_out=test_probs)
    inf_s = time.time() - t_inf
    decode = "head.decode" if getattr(model.head, "decode", None) is not None else "argmax"
    for fn, pr in (("dev_probs.json", dev_probs), ("test_probs.json", test_probs)):
        json.dump({"labels": labels, "decode": decode, "probs": pr}, open(os.path.join(out, fn), "w"))
    for d in test_docs:
        assert len(test_pred[d["id"]]) == len(d["sentences"]) and set(test_pred[d["id"]]) <= set(labels)
    pred_path = os.path.join(out, "prediction.json")
    json.dump(test_pred, open(pred_path, "w"))
    zip_path = os.path.join(out, f"submission_{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    summary = dict(name=name, phase1=info, technique=cfg["technique"], head=cfg["head"], pooling=cfg["pooling"],
                   context=ctx, kd=bool(cfg.get("kd")), params=params, best_epoch=best_epoch,
                   dev_macro_f1=dev_sc["all"]["macro_f1"], dev_macro_p=dev_sc["all"]["macro_p"],
                   dev_macro_r=dev_sc["all"]["macro_r"], dev_CL_macro_f1=dev_sc.get("CL", {}).get("macro_f1"),
                   dev_IT_macro_f1=dev_sc.get("IT", {}).get("macro_f1"), train_seconds=round(train_s, 1),
                   peak_mem_GB=round(peak, 2) if peak else None, test_infer_seconds=round(inf_s, 1),
                   test_sentences=sum(len(d["sentences"]) for d in test_docs), total_seconds=round(time.time() - t_start, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    log("done", **{k: v for k, v in summary.items() if k not in ("phase1", "context")}, zip=zip_path)


if __name__ == "__main__":
    main(sys.argv[1])
