#!/usr/bin/env python3
"""Fine-tune one Phase-1 model for abstractive summarization and write its Codabench submission (plumbing; declare it).

  python train_summ.py run_config.json

Pipeline: train.json -> clean (empty summaries, duplicate documents, test overlap removed) -> seeded validation hold-out
(split.val_docs) -> sentence-level ByteBPE (cached) -> source = head+tail selection of src_len tokens (summ_data) ->
target = reference truncated to tgt_len (whole sentences, + </s>) -> Phase-1 model WITH its LM head (gen_model) + YOUR
technique (techniques/<name>.py) -> token cross-entropy on the target tokens only (teacher forcing).

Source text (cfg["source"]):
  {"type": "document"}                                              the judgment itself (single-summary / stage-1 runs)
  {"type": "stage1", "train": f1, "val": f2, "test": f3}            stage 2 of the hierarchical method: the source is the
      concatenation of the stage-1 chunk summaries (files written by summ_infer.py --mode stage1 or by "after.stage1"
      below). Train documents missing from f1 are skipped (a stage-1 subset is allowed); val / test must be complete.

Checkpoint selection: lowest validation loss (teacher-forced, mean per target token), evaluated every
train.eval_every_steps and at the end of every epoch. Early stop: train.patience evaluations without improvement.

After training (best checkpoint): summaries for eval_generate.val_docs validation documents (scored with ROUGE-1/L and,
if eval_generate.bert_model is set, BERTScore) and for all test documents -> prediction.json (list format) + zip.
Optional "after": {"stage1": {"train_docs": N | "all", "decode": {...}}} also writes stage-1 chunk summaries of the
first N train documents, all validation and all test documents (stage1_train/val/test.json) for a stage-2 run.

KD (cfg["kd"] = {"teacher_run": <finished run dir>, ...your kd options}): the teacher (frozen) runs on the same documents
in its own input format; target tokens must be identical (same tgt_len / eos_on_truncated / target_trunc / source,
checked). YOUR techniques/kd.py kd_loss(student_logits (1, N, V), teacher_logits (1, N, V), labels (1, N), **opts) over
every target token replaces the cross-entropy. The student must have fewer parameters than the teacher (checked).
Sequence-level KD alternative: cfg["target_override"] = {id: text} file (e.g. teacher summaries of the train documents
from summ_infer.py) replaces the reference summaries of TRAIN documents only (validation keeps the references).

Outputs in <out_dir>/<name>/: config.json, log.jsonl, best_model.pt (trainable parameters + buffers only), val_refs.json,
val_generations.json, val_scores.json, test_summaries.json, prediction.json, submission_<name>.zip, summary.json.
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score_summ                      # noqa: E402
import summ_data                       # noqa: E402
import summ_infer                      # noqa: E402
import techniques                      # noqa: E402
from a1_tokenizer import ByteBPE       # noqa: E402
from inject import count_params        # noqa: E402

REQUIRED = {
    "format": ["src_len", "tgt_len", "eos_on_truncated"],
    "selection": ["strategy", "head_frac"],
    "split": ["val_docs"],
    "train": ["epochs", "batch_examples", "micro_examples", "lr", "weight_decay", "warmup_frac", "schedule",
              "grad_clip", "seed"],
}


def check_config(cfg, kind):
    missing = [f"{s}.{k}" for s, ks in REQUIRED.items() for k in ks if (cfg.get(s) or {}).get(k) is None]
    missing += [f"technique.name"] if (cfg.get("technique") or {}).get("name") is None else []
    if kind == "encdec":
        missing += ["format.start_token"] if cfg["format"].get("start_token") is None else []
    elif kind is not None:
        missing += ["format.sep_token"] if cfg["format"].get("sep_token") is None else []
    d = cfg.get("decode") or {}
    L = d.get("length") or {}
    if L.get("mode") is None:
        missing.append("decode.length.mode")
    elif L["mode"] == "ratio":
        missing += [f"decode.length.{k}" for k in ("ratio", "basis") if L.get(k) is None]
    assert not missing, "fill these config values first: " + ", ".join(missing)
    assert cfg["selection"]["strategy"] == "headtail", "selection.strategy: only 'headtail' is implemented"
    assert 0.0 <= cfg["selection"]["head_frac"] <= 1.0
    assert L["mode"] in ("ratio", "eos") and (L["mode"] == "eos" or L["basis"] in ("document", "input"))
    assert cfg["format"].get("target_trunc", "sentence") in ("sentence", "token")
    tr = cfg["train"]
    assert tr["batch_examples"] % tr["micro_examples"] == 0, "train.batch_examples must be a multiple of micro_examples"
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


def lr_factor(step, total, warmup, schedule):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 1.0 - prog if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * prog))


def micro_batches(exs, micro, shuffle, rng):
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].get("tgt", ())))   # similar target lengths together
    mbs = [order[i:i + micro] for i in range(0, len(order), micro)]
    if shuffle:
        rng.shuffle(mbs)
    return mbs


def check_lengths(model, fmt):
    bb = model.backbone
    S, Tt, P = fmt["src_len"], fmt["tgt_len"], bb.n_virtual_tokens
    if bb.kind == "encdec":
        assert S + P <= bb.max_len, f"src_len {S} + {P} virtual tokens > encoder length {bb.max_len}"
        assert Tt <= bb.max_tgt_len, f"tgt_len {Tt} > decoder length {bb.max_tgt_len} (backbone.max_seq_len_override)"
    else:
        assert P + S + Tt <= bb.max_len, \
            f"{P} virtual + src_len {S} + tgt_len {Tt} = {P + S + Tt} > context {bb.max_len} (separator included)"


def load_texts(cfg, tok, log, smoke):
    D, cache = cfg["data_dir"], cfg.get("cache_dir")
    train, val, test, rep = summ_data.load_splits(D, cfg["split"]["val_docs"], cfg["split"].get("seed", 0))
    src = cfg.get("source") or {"type": "document"}
    if src["type"] == "stage1":
        s1 = {k: json.load(open(src[k])) for k in ("train", "val", "test")}
        n0 = len(train)
        train = [d for d in train if d["id"] in s1["train"]]
        rep["train_with_stage1"] = len(train)
        rep["train_skipped_no_stage1"] = n0 - len(train)
        for k, docs in (("val", val), ("test", test)):
            miss = [d["id"] for d in docs if d["id"] not in s1[k]]
            assert not miss or smoke, f"stage-1 {k} file lacks {len(miss)} documents"
        srcdocs = {k: [{"id": d["id"], "document": " ".join(s1[k].get(d["id"], []))} for d in docs]
                   for k, docs in (("train", train), ("val", val), ("test", test))}
    else:
        srcdocs = {"train": train, "val": val, "test": test}
    if cfg.get("target_override"):
        ov = json.load(open(cfg["target_override"]))
        n0 = len(train)
        train = [{**d, "summary": ov[d["id"]]} for d in train if d["id"] in ov and ov[d["id"]].strip()]
        srcdocs["train"] = [d for d in srcdocs["train"] if d["id"] in {x["id"] for x in train}]
        rep["target_override"] = cfg["target_override"]
        rep["train_with_override"] = len(train)
        rep["train_skipped_no_override"] = n0 - len(train)
    if smoke:
        n = smoke.get("docs", 16)
        train, val, test = train[:n], val[:4], test[:4]
        keep = {d["id"] for d in train + val + test}
        srcdocs = {k: [d for d in v if d["id"] in keep] for k, v in srcdocs.items()}
    tag = src["type"] if src["type"] == "document" else \
        "stage1_" + summ_data.hashlib.md5(json.dumps(src, sort_keys=True).encode()).hexdigest()[:8]
    cp = lambda name: os.path.join(cache, f"{name}.pkl") if cache and not smoke else None
    enc = {k: summ_data.encode_sentences(tok, srcdocs[k], "document", cp(f"{tag}_{k}_src")) for k in srcdocs}
    tgt_tag = "override" if cfg.get("target_override") else "ref"
    tgt = {"train": summ_data.encode_sentences(tok, train, "summary", cp(f"{tgt_tag}_train_tgt")),
           "val": summ_data.encode_sentences(tok, val, "summary", cp("ref_val_tgt"))}
    return {"train": train, "val": val, "test": test}, srcdocs, enc, tgt, rep


@torch.no_grad()
def val_loss(model, exs, micro, kind, sep, start, device, amp):
    model.eval()
    tot, n = 0.0, 0
    for idx in micro_batches(exs, micro, False, None):
        b = summ_data.collate([exs[i] for i in idx], kind, sep, start, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg, lab = model.target_logits(b)
        tot += F.cross_entropy(lg.float(), lab, reduction="sum").item()
        n += lab.numel()
    model.train()
    return tot / max(1, n)


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    name, tr, fmt = cfg["name"], cfg["train"], cfg["format"]
    out = os.path.join(cfg["out_dir"], name)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.jsonl"))
    seed = tr.get("seed") or 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    t_start = time.time()
    deadline = t_start + 60 * tr["max_minutes"] if tr.get("max_minutes") else None
    smoke = cfg.get("smoke")          # {"steps": n, "docs": m}: real model, few documents and steps; not submittable

    check_config(cfg, None)
    tok = ByteBPE.from_json(cfg["tokenizer"])
    cfg["phase1_ckpt"] = summ_infer.resolve_ckpt(cfg["phase1_ckpt"])
    model, tmod, info = summ_infer.build_run_model(cfg, tok.get_vocab_size())
    kind = model.backbone.kind
    check_config(cfg, kind)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    if cfg.get("init_from_run"):
        init_run = cfg["init_from_run"]
        assert os.path.abspath(init_run) != os.path.abspath(out)
        icfg = json.load(open(os.path.join(init_run, "config.json")))
        for k in ("technique", "backbone"):
            assert (icfg.get(k) or {}) == (cfg.get(k) or {}), f"init_from_run: '{k}' differs from {init_run}"
        summ_infer.load_trainable(model, os.path.join(init_run, "best_model.pt"))
        info["init_from_run"] = init_run
    model.to(device)
    check_lengths(model, fmt)
    sep, start, eos = summ_infer.special_ids(tok, fmt, kind)

    docs, srcdocs, enc, tgt, drep = load_texts(cfg, tok, log, smoke)
    exs, est = {}, {}
    for s in ("train", "val"):
        exs[s], est[s] = summ_data.build_examples(srcdocs[s], enc[s], tgt[s], fmt, cfg["selection"], eos)
    exs["test"], est["test"] = summ_data.build_examples(srcdocs["test"], enc["test"], None, fmt, cfg["selection"], eos)
    log("data", **drep, **{f"{s}_{k}": v for s in est for k, v in est[s].items()}, kind=kind,
        src_len=fmt["src_len"], tgt_len=fmt["tgt_len"])
    params = count_params(model)
    log("model", **info, **params, max_len=model.backbone.max_len,
        max_tgt_len=model.backbone.max_tgt_len, virtual_tokens=model.backbone.n_virtual_tokens)

    teacher = None
    if cfg.get("kd"):
        kd = {k: v for k, v in cfg["kd"].items() if k != "teacher_run"}
        tdir = cfg["kd"]["teacher_run"]
        teacher, tcfg = summ_infer.load_run(tdir, device, tok.get_vocab_size())
        for p in teacher.parameters():
            p.requires_grad_(False)
        t_total = count_params(teacher)["total"]
        assert params["total"] < t_total, f"KD student ({params['total']:,}) must be smaller than the teacher ({t_total:,})"
        for k in ("tgt_len", "eos_on_truncated", "target_trunc"):
            assert tcfg["format"].get(k) == fmt.get(k), f"KD: format.{k} differs from the teacher ({tcfg['format'].get(k)})"
        assert (tcfg.get("source") or {"type": "document"}) == (cfg.get("source") or {"type": "document"}), "KD: source differs"
        t_kind = teacher.backbone.kind
        check_lengths(teacher, tcfg["format"])
        t_sep, t_start_id, _ = summ_infer.special_ids(tok, tcfg["format"], t_kind)
        t_exs, _ = summ_data.build_examples(srcdocs["train"], enc["train"], tgt["train"], tcfg["format"],
                                            tcfg["selection"], eos)
        t_by_id = {e["id"]: e for e in t_exs}
        kd_loss = techniques.get("kd").kd_loss
        log("teacher", run=tcfg["name"], kind=t_kind, params=t_total, student_params=params["total"])

    if hasattr(tmod, "param_groups"):
        groups = tmod.param_groups(model, tr["lr"])
    else:
        seen, ps = set(), []
        for p in model.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); ps.append(p)
        groups = [{"params": ps, "lr": tr["lr"]}]
    for g in groups:
        g["params"] = list(g["params"]); g["base_lr"] = g["lr"]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    extra_loss, on_step = getattr(tmod, "extra_loss", None), getattr(tmod, "on_step", None)

    rng = random.Random(seed)
    micro, accum = tr["micro_examples"], tr["batch_examples"] // tr["micro_examples"]
    steps_per_epoch = math.ceil(len(micro_batches(exs["train"], micro, False, None)) / accum)
    total = steps_per_epoch * tr["epochs"]
    warmup = int(tr["warmup_frac"] * total)
    im = tr.get("infer_micro_examples") or micro
    log("schedule", steps_per_epoch=steps_per_epoch, total_steps=total, warmup_steps=warmup,
        train_target_tokens=int(sum(len(e["tgt"]) for e in exs["train"])))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best, best_at, bad_evals, step, seen_tok, history = float("inf"), None, 0, 0, 0, []
    stop = False
    t_train = time.time()

    def evaluate(tag, rec):
        nonlocal best, best_at, bad_evals
        t_ev = time.time()
        vl = val_loss(model, exs["val"], im, kind, sep, start, device, amp)
        rec.update(at=tag, val_loss=round(vl, 4), eval_s=round(time.time() - t_ev, 1))
        if vl < best:
            best, best_at, bad_evals = vl, tag, 0
            summ_infer.save_trainable(model, os.path.join(out, "best_model.pt"))
            rec["saved_best"] = True
        else:
            bad_evals += 1
        history.append(rec)
        log("eval", **rec)

    eval_every = tr.get("eval_every_steps")
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        mbs = micro_batches(exs["train"], micro, True, rng)
        run_loss, n_tok = 0.0, 0
        for i in range(0, len(mbs), accum):
            group = mbs[i:i + accum]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, warmup, tr["schedule"])
            n_step_tok = sum(len(exs["train"][j]["tgt"]) for idx in group for j in idx)
            for idx in group:
                batch_ex = [exs["train"][j] for j in idx]
                b = summ_data.collate(batch_ex, kind, sep, start, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    lg, lab = model.target_logits(b)
                lg = lg.float()
                if teacher is not None:
                    tb = summ_data.collate([t_by_id[e["id"]] for e in batch_ex], t_kind, t_sep, t_start_id, device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        tlg, tlab = teacher.target_logits(tb)
                    assert torch.equal(tlab, lab), "KD: teacher / student target tokens differ"
                    loss = kd_loss(lg[None], tlg.float()[None], lab[None], **kd)
                else:
                    loss = F.cross_entropy(lg, lab)
                if extra_loss is not None:
                    loss = loss + extra_loss(model)
                scaler.scale(loss * lab.numel() / n_step_tok).backward()     # mean over the step's target tokens
                run_loss += loss.item() * lab.numel(); n_tok += lab.numel()
                seen_tok += b["ids"].numel() if "ids" in b else b["src"].numel() + b["dec_in"].numel()
            if tr["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tr["grad_clip"])
            scaler.step(opt); scaler.update()
            step += 1
            if on_step is not None:          # before zero_grad: your hook can still read p.grad (AdaLoRA importance)
                on_step(model, step, total)
            opt.zero_grad(set_to_none=True)
            if smoke and step >= smoke["steps"]:
                stop = True; break
            if step % (tr.get("log_every") or 50) == 0:
                el = time.time() - t_train
                log("train", epoch=epoch, step=step, of=total, loss=round(run_loss / max(1, n_tok), 4),
                    lr=opt.param_groups[0]["lr"], tok_s=round(seen_tok / el),
                    eta_min=round((total - step) * el / step / 60, 1))
            if eval_every and step % eval_every == 0 and step % steps_per_epoch != 0:
                evaluate(f"e{epoch}_step{step}", dict(epoch=epoch, step=step, train_loss=round(run_loss / max(1, n_tok), 4)))
                if tr.get("patience") and bad_evals >= tr["patience"]:
                    log("early stop", bad_evals=bad_evals); stop = True; break
            if deadline and time.time() > deadline:
                log("max_minutes reached: stopping inside the epoch", step=step); stop = True; break
        evaluate(f"e{epoch}_end", dict(epoch=epoch, step=step, train_loss=round(run_loss / max(1, n_tok), 4)))
        if stop or (tr.get("patience") and bad_evals >= tr["patience"]):
            break
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    del opt
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()

    # ---- generation with the best checkpoint ----
    summ_infer.load_trainable(model, os.path.join(out, "best_model.pt"))
    model.eval()
    dcfg = cfg["decode"]
    eg = cfg.get("eval_generate") or {}
    n_val = len(docs["val"]) if eg.get("val_docs") in (None, "all") else int(eg["val_docs"])
    vdocs = docs["val"][:n_val]
    refs = {d["id"]: d["summary"] for d in vdocs}
    json.dump(refs, open(os.path.join(out, "val_refs.json"), "w"), ensure_ascii=False)
    vsrc = {d["id"]: enc["val"][d["id"]] for d in vdocs}
    t0 = time.time()
    vgen = summ_infer.summarize_single(model, tok, cfg, vdocs, vsrc, dcfg, device, log) if vdocs else {}
    val_gen_s = time.time() - t0
    json.dump(vgen, open(os.path.join(out, "val_generations.json"), "w"), ensure_ascii=False)
    vsc = {}
    if vdocs:
        vsc = score_summ.score(vgen, refs, eg.get("bert_model"), "cuda" if amp else "cpu")
        json.dump(vsc, open(os.path.join(out, "val_scores.json"), "w"), indent=2)
        log("val generation", **vsc, seconds=round(val_gen_s, 1))
    t0 = time.time()
    tgen = summ_infer.summarize_single(model, tok, cfg, docs["test"], enc["test"], dcfg, device, log)
    test_gen_s = time.time() - t0
    json.dump(tgen, open(os.path.join(out, "test_summaries.json"), "w"), ensure_ascii=False)
    pred_path = os.path.join(out, "prediction.json")
    summ_infer.write_prediction(tgen, docs["test"], pred_path)
    zip_path = os.path.join(out, f"submission_{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")

    after = (cfg.get("after") or {}).get("stage1")
    s1_info = None
    if after:
        assert (cfg.get("source") or {"type": "document"})["type"] == "document", "stage-1 outputs need a document-source run"
        n_tr = len(docs["train"]) if after.get("train_docs") in (None, "all") else int(after["train_docs"])
        s1_info = {}
        for s, dd in (("train", docs["train"][:n_tr]), ("val", docs["val"]), ("test", docs["test"])):
            denc = enc[s] if s != "train" else {d["id"]: enc["train"][d["id"]] for d in dd}
            t0 = time.time()
            res = summ_infer.summarize_stage1(model, tok, cfg, dd, denc, after["decode"], device, log)
            p = os.path.join(out, f"stage1_{s}.json")
            json.dump(res, open(p, "w"), ensure_ascii=False)
            s1_info[s] = {"docs": len(res), "chunks": sum(map(len, res.values())), "seconds": round(time.time() - t0, 1)}
            log("stage1 written", split=s, path=p, **s1_info[s])

    summary = dict(name=name, phase1=info, kind=kind, technique=cfg["technique"], kd=bool(cfg.get("kd")),
                   format=fmt, selection=cfg["selection"], source=cfg.get("source") or {"type": "document"},
                   decode=dcfg, params=params, best_val_loss=round(best, 4), best_at=best_at, history=history,
                   val_scores=vsc, val_generated=len(vgen), train_seconds=round(train_s, 1),
                   train_tok_s=round(seen_tok / max(train_s, 1e-9)), peak_mem_GB=round(peak, 2) if peak else None,
                   val_gen_seconds=round(val_gen_s, 1), test_gen_seconds=round(test_gen_s, 1),
                   test_gen_s_per_doc=round(test_gen_s / max(1, len(docs["test"])), 3), stage1=s1_info,
                   data=drep, smoke=bool(smoke), total_seconds=round(time.time() - t_start, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    log("done", name=name, best_val_loss=round(best, 4), best_at=best_at, **{f"val_{k}": v for k, v in vsc.items()
        if k in ("rouge1_f1", "rougeL_f1", "bertscore_f1", "pred_words")}, zip=zip_path,
        train_min=round(train_s / 60, 1), trainable=params["trainable"])


if __name__ == "__main__":
    main(sys.argv[1])
