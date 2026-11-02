#!/usr/bin/env python3
"""Fine-tune one Phase-1 model for L-NER and write its Codabench submission (plumbing; declare it).

  python train_ner.py run_config.json

Pipeline: load docs + gold -> tokenize with char offsets (ner_align) -> gold spans to token tags -> sliding windows
-> Phase-1 backbone (backbones) + YOUR technique (techniques/<name>.py) + YOUR head (heads.py)
-> AdamW training (fp16 autocast on GPU) with strict macro-F1 on the held-out docs after every epoch
-> best checkpoint -> val report -> test predictions -> prediction.json + submission_<name>.zip.

Outputs in <out_dir>/<name>/:
  config.json, log.jsonl, best_model.pt (full state, fp32), val_scores.json,
  val_predictions.json, prediction.json, submission_<name>.zip, summary.json (scores, params, memory, timings).

Config keys whose value is None in the template are your choices (course rule 9) and must be filled; see
configs/template.json. Everything marked "plumbing" has a fixed meaning and does not change results.
"""
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

import backbones                      # noqa: E402
import heads                          # noqa: E402
import ner_align                      # noqa: E402
import ner_data                       # noqa: E402
import techniques                     # noqa: E402
from a1_tokenizer import ByteBPE      # noqa: E402
from inject import count_params       # noqa: E402
from ner_metrics import strict_scores  # noqa: E402

IGNORE = -100

# keys that must be set (not None); KD / backbone-specific keys are checked separately
REQUIRED = {
    "preprocess": ["split_punct", "tag_scheme"],
    "window": ["train_len", "train_stride", "infer_len", "infer_stride", "merge"],
    "train": ["epochs", "batch_size", "micro_batch", "lr", "weight_decay", "warmup_frac", "schedule", "grad_clip"],
    "decode": ["repair"],
}


def check_config(cfg):
    missing = [f"{sec}.{k}" for sec, keys in REQUIRED.items() for k in keys if cfg.get(sec, {}).get(k) is None]
    if cfg.get("head", {}).get("name") is None:
        missing.append("head.name")
    if cfg.get("technique", {}).get("name") is None:
        missing.append("technique.name")
    if cfg["split"]["mode"] == "random" and cfg["split"].get("val_frac") is None:
        missing.append("split.val_frac")
    assert not missing, "fill these config values first: " + ", ".join(missing)
    check_technique_options(cfg)
    tr = cfg["train"]
    assert tr["batch_size"] % tr["micro_batch"] == 0, "train.batch_size must be a multiple of train.micro_batch"
    assert cfg["preprocess"]["tag_scheme"] in ("BIO", "BIOES")
    assert cfg["train"]["schedule"] in ("linear", "cosine", "constant")
    assert cfg["decode"]["repair"] in ("lenient", "strict")


def check_technique_options(cfg):
    """Fail before any GPU time is spent if the technique options do not fit YOUR apply() signature
    (unknown key such as "rank" instead of "r", or a missing required option)."""
    import inspect
    tech = cfg["technique"]
    opts = {k: v for k, v in tech.items() if k != "name"}
    fn = techniques.get(tech["name"]).apply
    sig = inspect.signature(fn)
    try:
        sig.bind(None, **opts)                       # None stands in for the backbone argument
    except TypeError as e:
        raise AssertionError(f"technique '{tech['name']}' options {opts} do not fit apply{sig}: {e}")


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg, **kv):
        rec = {"t": round(time.time(), 1), "msg": msg, **kv}
        print(json.dumps(rec), flush=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class NERModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, ids):
        return self.head(self.backbone(ids))


def build_model(cfg, n_tags, vocab_size, load_weights=True, apply_technique=True):
    bb, spec, info = backbones.build_backbone(cfg["phase1_ckpt"], cfg.get("backbone", {}), vocab_size,
                                              load_weights=load_weights,
                                              fast_attention=cfg.get("fast_attention", True))
    for p in bb.parameters():
        p.requires_grad_(True)
    tech = cfg["technique"]
    tmod = techniques.get(tech["name"])
    if apply_technique:
        tmod.apply(bb, **{k: v for k, v in tech.items() if k != "name"})
    head = heads.build_head(bb.d_model, n_tags, **cfg["head"])
    for p in head.parameters():
        p.requires_grad_(True)
    return NERModel(bb, head), tmod, info


def save_model(model, path):
    # full precision for every tensor: reloading must reproduce the evaluated model exactly
    torch.save(model.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_run(run_dir, device, vocab_size):
    """Rebuild a finished run's model (e.g. a KD teacher) from its config.json + best_model.pt."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    tags = json.load(open(os.path.join(run_dir, "tags.json")))
    model, _, _ = build_model(cfg, len(tags), vocab_size, load_weights=False)
    sd = torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return model.to(device).eval(), cfg, tags


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def encode_docs(texts, gold, otok, tag2id, scheme):
    out, skipped = {}, 0
    for d, text in texts.items():
        ids, st, en = otok(text)
        if gold is not None:
            tags, sk = ner_align.spans_to_tags(gold.get(d, []), st, en, tag2id, scheme)
            skipped += sk
        else:
            tags = None
        out[d] = dict(ids=ids, starts=st, ends=en, tags=tags, text=text)
    return out, skipped


def make_windows(enc, doc_ids, length, stride):
    """-> list of (doc_id, start, window_length)."""
    wins = []
    for d in doc_ids:
        n = len(enc[d]["ids"])
        if n == 0:                                   # empty document (fold3 has one): no windows, predicts []
            continue
        for s in ner_align.window_starts(n, length, stride):
            wins.append((d, s, min(length, n)))
    return wins


def batches(wins, micro, shuffle, rng):
    """Group windows of equal length into micro-batches (no padding anywhere)."""
    by_len = {}
    for w in wins:
        by_len.setdefault(w[2], []).append(w)
    out = []
    for group in by_len.values():
        if shuffle:
            rng.shuffle(group)
        out += [group[i:i + micro] for i in range(0, len(group), micro)]
    if shuffle:
        rng.shuffle(out)
    return out


def to_tensors(enc, mb, device, with_tags=True):
    ids = torch.tensor(np.stack([enc[d]["ids"][s:s + n] for d, s, n in mb]), dtype=torch.long, device=device)
    if not with_tags:
        return ids, None
    tags = torch.tensor(np.stack([enc[d]["tags"][s:s + n] for d, s, n in mb]), dtype=torch.long, device=device)
    return ids, tags


def class_weights(enc, doc_ids, n_tags, mode, o_weight):
    if mode is None and o_weight is None:
        return None
    w = torch.ones(n_tags)
    if mode is not None:
        counts = np.bincount(np.concatenate([enc[d]["tags"] for d in doc_ids]), minlength=n_tags).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        power = {"inv_freq": 1.0, "inv_sqrt": 0.5}[mode]
        w = torch.tensor((counts.sum() / counts) ** power, dtype=torch.float32)
        w = w / w.mean()
    if o_weight is not None:
        w[0] = float(o_weight)
    return w


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def doc_logits(model, enc, doc_ids, length, stride, merge, micro, device, amp):
    """Merged per-token logits for every doc: {doc_id: (N, n_tags) float32 tensor on CPU}."""
    model.eval()
    wins = make_windows(enc, doc_ids, length, stride)
    best = {d: np.full(len(enc[d]["ids"]), -1.0) for d in doc_ids}
    sums = {}
    counts = {d: np.zeros(len(enc[d]["ids"])) for d in doc_ids}
    for mb in batches(wins, micro, False, None):
        ids, _ = to_tensors(enc, mb, device, with_tags=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(ids).float().cpu()
        for (d, s, n), lg in zip(mb, logits):
            if d not in sums:
                sums[d] = torch.zeros(len(enc[d]["ids"]), lg.shape[-1])
            if merge == "mean":
                sums[d][s:s + n] += lg
                counts[d][s:s + n] += 1
            else:
                w = ner_align.merge_weights(n, merge)
                upd = w > best[d][s:s + n]
                idx = torch.from_numpy(np.nonzero(upd)[0])
                sums[d][s + idx] = lg[idx]
                best[d][s + idx] = w[idx]
    if merge == "mean":
        for d in sums:
            sums[d] /= torch.tensor(counts[d], dtype=torch.float32).unsqueeze(-1)
    return sums


@torch.no_grad()
def predict_spans(model, enc, doc_ids, cfg, id2tag, device, amp):
    w = cfg["window"]
    logits = doc_logits(model, enc, doc_ids, w["infer_len"], w["infer_stride"], w["merge"],
                        cfg["train"].get("infer_micro_batch") or cfg["train"]["micro_batch"] * 2, device, amp)
    pred = {}
    dec = getattr(model.head, "decode", None)
    for d in doc_ids:
        if len(enc[d]["ids"]) == 0:
            pred[d] = []
            continue
        lg = logits[d]
        if dec is not None:
            tag_ids = dec(lg.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        else:
            tag_ids = lg.argmax(-1).numpy()
        spans = ner_align.tags_to_spans(tag_ids, enc[d]["starts"], enc[d]["ends"], id2tag, cfg["decode"]["repair"])
        if cfg["decode"].get("snap_to_word"):              # optional decoding step, off by default
            spans = ner_align.snap_to_words(enc[d]["text"], spans)
        pred[d] = spans
    return pred


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def lr_factor(step, total, warmup, schedule):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1.0 - prog if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * prog))


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    check_config(cfg)
    name = cfg["name"]
    out = os.path.join(cfg["out_dir"], name)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.jsonl"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    tr = cfg["train"]
    seed = tr.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    t_start = time.time()
    deadline = t_start + 60 * tr["max_minutes"] if tr.get("max_minutes") else None

    # ---- data
    tok = ByteBPE.from_json(cfg["tokenizer"])
    texts, gold, fold_of, labels = ner_data.load_train(cfg["data_dir"], verbose=False)
    test_texts = ner_data.load_test(cfg["data_dir"])
    scheme = cfg["preprocess"]["tag_scheme"]
    tags = ner_align.tag_list(labels, scheme)
    tag2id, id2tag = {t: i for i, t in enumerate(tags)}, dict(enumerate(tags))
    json.dump(tags, open(os.path.join(out, "tags.json"), "w"))
    otok = ner_align.OffsetTokenizer(tok, split_punct=cfg["preprocess"]["split_punct"])
    enc, skipped = encode_docs(texts, gold, otok, tag2id, scheme)
    enc_test, _ = encode_docs(test_texts, None, otok, tag2id, scheme)
    train_ids, val_ids = ner_data.split_docs(list(texts), fold_of, cfg["split"])
    smoke = cfg.get("smoke")                     # {"steps": n}: real model + real window/micro-batch, tiny data
    if smoke:
        train_ids, val_ids = train_ids[:4], val_ids[:2]
        test_texts = dict(list(test_texts.items())[:2])
        enc_test = {d: enc_test[d] for d in test_texts}
        tr["epochs"] = 1
        log("SMOKE MODE", steps=smoke["steps"], train_docs=len(train_ids), val_docs=len(val_ids), test_docs=2)
    log("data", train_docs=len(train_ids), val_docs=len(val_ids), test_docs=len(test_texts),
        train_tokens=int(sum(len(enc[d]["ids"]) for d in train_ids)), overlap_skipped=skipped, n_tags=len(tags))

    # ---- model
    model, tmod, info = build_model(cfg, len(tags), tok.get_vocab_size())
    model.to(device)
    w = cfg["window"]
    assert max(w["train_len"], w["infer_len"]) <= model.backbone.max_len, \
        f"window longer than the backbone's max_seq_len {model.backbone.max_len}"
    params = count_params(model)
    params["backbone_total"] = sum(p.numel() for p in model.backbone.parameters())
    params["head"] = sum(p.numel() for p in model.head.parameters())
    log("model", **info, **params)

    teacher = None
    if cfg.get("kd"):
        kd = dict(cfg["kd"])
        teacher, tcfg, ttags = load_run(kd.pop("teacher_run"), device, tok.get_vocab_size())
        assert ttags == tags, "teacher and student must use the same labels and tag scheme"
        assert w["train_len"] <= teacher.backbone.max_len, "train window longer than the teacher's max_seq_len"
        for p in teacher.parameters():
            p.requires_grad_(False)
        kd_loss = techniques.get("kd").kd_loss
        t_params, s_params = count_params(teacher)["total"], params["total"]
        assert s_params < t_params, f"KD student ({s_params:,} params) must be smaller than the teacher ({t_params:,})"
        pass_weights = kd.pop("pass_class_weights", False)   # plumbing key: hand the CE class weights to kd_loss
        log("teacher", run=tcfg["name"], params=t_params, student_params=s_params, pass_class_weights=pass_weights)

    # ---- optimizer
    if hasattr(tmod, "param_groups"):
        groups = tmod.param_groups(model, tr["lr"])
    else:
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        groups = [{"params": bb_params, "lr": tr["lr"]},
                  {"params": list(model.head.parameters()), "lr": tr.get("head_lr") or tr["lr"]}]
    for g in groups:
        g["params"] = list(g["params"])
        g["base_lr"] = g["lr"]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=tr["weight_decay"], betas=tuple(tr.get("betas", (0.9, 0.999))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cw = class_weights(enc, train_ids, len(tags), tr.get("class_weights"), tr.get("o_weight"))
    if cw is not None:
        cw = cw.to(device)
        log("class_weights", weights={t: round(float(x), 3) for t, x in zip(tags, cw)})
    if teacher is not None and pass_weights:
        assert cw is not None, "kd.pass_class_weights needs train.class_weights (or o_weight) to be set"
        kd["weight"] = cw                                     # your kd_loss receives it as cfg["weight"]

    train_wins = make_windows(enc, train_ids, w["train_len"], w["train_stride"])
    rng = random.Random(seed)
    micro, accum = tr["micro_batch"], tr["batch_size"] // tr["micro_batch"]
    steps_per_epoch = math.ceil(len(batches(train_wins, micro, False, None)) / accum)
    total_steps = steps_per_epoch * tr["epochs"]
    warmup = int(tr["warmup_frac"] * total_steps)
    log("schedule", train_windows=len(train_wins), steps_per_epoch=steps_per_epoch, total_steps=total_steps,
        warmup_steps=warmup)

    best_f1, best_epoch, bad_epochs, step = -1.0, -1, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    tokens_seen = 0
    stop = False
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        if teacher is not None:
            teacher.eval()
        mbs = batches(train_wins, micro, True, rng)
        run_loss, n_loss, t_ep = 0.0, 0, time.time()
        for i in range(0, len(mbs), accum):
            group = mbs[i:i + accum]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total_steps, warmup, tr["schedule"])
            for mb in group:
                ids, tg = to_tensors(enc, mb, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(ids)
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(ids)
                logits = logits.float()
                if teacher is not None:
                    loss = kd_loss(logits, t_logits.float(), tg, **kd)
                elif hasattr(model.head, "loss"):
                    loss = model.head.loss(logits, tg)
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tg.reshape(-1),
                                           weight=cw, ignore_index=IGNORE)
                scaler.scale(loss / len(group)).backward()
                run_loss += loss.item()
                n_loss += 1
                tokens_seen += ids.numel()
            if tr["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if smoke and step >= smoke["steps"]:
                log("smoke: step limit reached", step=step)
                stop = True
                break
            if step % tr.get("log_every", 20) == 0:
                log("train", epoch=epoch, step=step, loss=round(run_loss / max(1, n_loss), 4),
                    lr=opt.param_groups[0]["lr"], tok_s=round(tokens_seen / (time.time() - t_train)))
            if deadline and time.time() > deadline:
                log("deadline reached: stopping training", step=step)
                stop = True
                break
        ep_loss = run_loss / max(1, n_loss)
        rec = dict(epoch=epoch, step=step, train_loss=round(ep_loss, 4), epoch_s=round(time.time() - t_ep, 1))
        if val_ids and (epoch % tr.get("eval_every_epochs", 1) == 0 or epoch == tr["epochs"] or stop):
            t_ev = time.time()
            sc = strict_scores({d: gold[d] for d in val_ids},
                               predict_spans(model, enc, val_ids, cfg, id2tag, device, amp), labels)
            rec.update(val_macro_f1=sc["macro_f1"], val_micro_f1=sc["micro_f1"], eval_s=round(time.time() - t_ev, 1))
            if sc["macro_f1"] > best_f1:
                best_f1, best_epoch, bad_epochs = sc["macro_f1"], epoch, 0
                save_model(model, os.path.join(out, "best_model.pt"))
                rec["saved"] = True
            else:
                bad_epochs += 1
        log("epoch", **rec)
        if not val_ids:
            save_model(model, os.path.join(out, "best_model.pt"))
            best_epoch = epoch
        if stop or (tr.get("patience") and bad_epochs >= tr["patience"]):
            break
    train_s = time.time() - t_train
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None

    # ---- best model -> val report + test predictions
    sd = torch.load(os.path.join(out, "best_model.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    summary = dict(name=name, phase1=info, technique=cfg["technique"], head=cfg["head"], kd=bool(cfg.get("kd")),
                   params=params, best_epoch=best_epoch, train_seconds=round(train_s, 1),
                   train_tokens=tokens_seen, peak_mem_GB=round(peak_gb, 2) if peak_gb else None)
    if val_ids:
        val_pred = predict_spans(model, enc, val_ids, cfg, id2tag, device, amp)
        val_sc = strict_scores({d: gold[d] for d in val_ids}, val_pred, labels)
        json.dump(val_sc, open(os.path.join(out, "val_scores.json"), "w"), indent=2)
        json.dump(ner_data.to_submission(val_pred), open(os.path.join(out, "val_predictions.json"), "w"))
        summary.update(val_macro_f1=val_sc["macro_f1"], val_macro_p=val_sc["macro_p"], val_macro_r=val_sc["macro_r"],
                       val_micro_f1=val_sc["micro_f1"])
    t_inf = time.time()
    test_pred = predict_spans(model, enc_test, list(test_texts), cfg, id2tag, device, amp)
    inf_s = time.time() - t_inf
    n_test_tok = sum(len(enc_test[d]["ids"]) for d in test_texts)
    summary.update(test_infer_seconds=round(inf_s, 1), test_infer_tok_s=round(n_test_tok / max(inf_s, 1e-9)),
                   test_spans=sum(len(v) for v in test_pred.values()))
    sub = ner_data.to_submission(test_pred)
    for d in test_texts:
        sub.setdefault(d, [])
    pred_path = os.path.join(out, "prediction.json")
    json.dump(sub, open(pred_path, "w"))
    zip_path = os.path.join(out, f"submission_{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    summary["total_seconds"] = round(time.time() - t_start, 1)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    log("done", **{k: v for k, v in summary.items() if k not in ("phase1",)}, zip=zip_path)


if __name__ == "__main__":
    main(sys.argv[1])
