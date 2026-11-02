#!/usr/bin/env python3
"""Phase-6 summary generation (plumbing; declare it).

Modes
  single  one summary per document from its head+tail-selected source (the run's own format / selection).
  stage1  hierarchical stage 1: the document is packed into sentence chunks of <= src_len tokens and the model writes a
          summary for EVERY chunk (each chunk is a whole "document" to the model) -> {id: [chunk summary, ...]}.
          These files are the source of a stage-2 run (config source = {"type": "stage1", ...}).

Length control (decode["length"]):
  {"mode": "ratio", "ratio": r, "basis": "document" | "input", "min_words": a, "max_words": b, "allow_eos": bool}
      word budget = clip(round(r * words), a, b); words = words of the original document ("document") or of the text
      actually fed to the model ("input": the selected source / the chunk). Generation stops at the budget (and at
      </s> if allow_eos, but never before min_words).
  {"mode": "eos", "min_words": a, "max_words": b}
      stop at </s> (forbidden before min_words) or at max_words.
  Every row is also capped by the context: A/B/D max_len - virtual - src_len - 1 tokens, C the decoder length.
Other decode options: temperature (0 = greedy), top_k, top_p, repetition_penalty, no_repeat_ngram (token n-grams of
the generated text), batch_size, max_new_tokens (optional extra cap).

CLI (uses a finished run's best checkpoint):
  python summ_infer.py --run RUN_DIR --split val|test|train --mode single|stage1 --decode decode.json --out out.json
        [--limit N] [--shard i/n]     (--shard: process every n-th document starting at i; merge with --merge)
  python summ_infer.py --merge out.json part0.json part1.json ...
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_model                      # noqa: E402
import summ_data                      # noqa: E402
import techniques                     # noqa: E402
from a1_tokenizer import ByteBPE      # noqa: E402
from fastattn_ab import ngram_ban, sample_next   # noqa: E402


# ---------------------------------------------------------------------------
# model build / save / load (shared with train_summ.py)
# ---------------------------------------------------------------------------

def build_run_model(cfg, vocab_size, load_weights=True):
    model, spec, info = gen_model.build_model(cfg["phase1_ckpt"], cfg.get("backbone") or {}, vocab_size,
                                              load_weights=load_weights, fast_attention=cfg.get("fast_attention", True))
    for p in model.parameters():
        p.requires_grad_(True)
    tech = cfg["technique"]
    tmod = techniques.get(tech["name"])
    tmod.apply(model.backbone, **{k: v for k, v in tech.items() if k != "name"})
    return model, tmod, info


def save_trainable(model, path):
    """Only what training changed: trainable parameters + every buffer (technique buffers such as QLoRA codes /
    AdaLoRA masks). Frozen parameters are rebuilt from the Phase-1 checkpoint on load."""
    trainable = {n for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable or k in buffers}
    torch.save(sd, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_trainable(model, path):
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"{path}: unexpected keys {unexpected[:5]}"
    frozen = {n for n, p in model.named_parameters(remove_duplicate=False) if not p.requires_grad}
    bad = [k for k in missing if k not in frozen]
    assert not bad, f"{path}: missing trainable keys {bad[:5]}"


def resolve_ckpt(path):
    if os.path.exists(path):
        return path
    tail = os.path.join(*path.split(os.sep)[-2:])
    hits = sorted(glob.glob(os.path.join("/kaggle/input", "**", tail), recursive=True))
    assert hits, f"Phase-1 checkpoint {path} not found (looked for */{tail} under /kaggle/input)"
    return hits[0]


def load_run(run_dir, device, vocab_size, ckpt_name="best_model.pt"):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg["phase1_ckpt"] = resolve_ckpt(cfg["phase1_ckpt"])
    model, _, _ = build_run_model(cfg, vocab_size, load_weights=True)
    load_trainable(model, os.path.join(run_dir, ckpt_name))
    return model.to(device).eval(), cfg


def special_ids(tok, fmt, kind):
    sp = tok.special_ids
    sep = sp[fmt["sep_token"]] if kind != "encdec" else None
    start = None
    if kind == "encdec":
        st = fmt["start_token"]
        start = None if st == "last_src" else sp[st]
    return sep, start, tok.eot_id


# ---------------------------------------------------------------------------
# batched decoding
# ---------------------------------------------------------------------------

def context_cap(model, src_len):
    bb = model.backbone
    if bb.kind == "encdec":
        return bb.max_tgt_len
    return bb.max_len - bb.n_virtual_tokens - src_len - 1


def word_budget(L, words):
    if L["mode"] == "eos":
        return L.get("max_words")
    b = round(L["ratio"] * words)
    if L.get("min_words"):
        b = max(b, L["min_words"])
    if L.get("max_words"):
        b = min(b, L["max_words"])
    return max(1, b)


@torch.inference_mode()
def generate(model, tok, srcs, budgets, dcfg, sep_id, start_id, eos_id, device, log=None):
    """srcs: list of src_len-id lists; budgets: list of word budgets (None = unlimited). -> list[str] (input order)."""
    model.eval()
    bb = model.backbone
    L = dcfg["length"]
    S = len(srcs[0])
    cap = context_cap(model, S)
    if dcfg.get("max_new_tokens"):
        cap = min(cap, dcfg["max_new_tokens"])
    assert cap > 0, f"no room to generate: src_len {S} fills the context"
    allow_eos = L["mode"] == "eos" or L.get("allow_eos", False)
    min_words = L.get("min_words") or 0
    amp = device.type == "cuda"
    dtype = torch.float16 if amp else torch.float32
    eow = torch.tensor(tok.token_eow, device=device, dtype=torch.bool)
    has_bytes = torch.tensor([len(b) > 0 for b in tok.token_bytes], device=device, dtype=torch.bool)
    n_dec = len(tok.token_bytes)
    banned = torch.tensor([i for i in range(tok.n_special) if not (allow_eos and i == eos_id)], device=device)
    order = sorted(range(len(srcs)), key=lambda i: -(budgets[i] or 10 ** 9))
    out_text = [None] * len(srcs)
    bs = dcfg.get("batch_size") or 16
    t0 = time.time()
    for bi, s0 in enumerate(range(0, len(order), bs)):
        idx = order[s0:s0 + bs]
        B = len(idx)
        src = torch.tensor([srcs[i] + ([sep_id] if bb.kind != "encdec" else []) for i in idx], device=device)
        tgt_words = torch.tensor([budgets[i] if budgets[i] else 10 ** 9 for i in idx], device=device)
        max_new = cap
        words = torch.zeros(B, dtype=torch.long, device=device)
        in_word = torch.zeros(B, dtype=torch.bool, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        gen = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp), \
                gen_model.session(model, src, max_new, dtype, start_id=start_id) as (h, step):
            for t in range(max_new):
                logits = model.head(h).float()
                logits[~in_word, tok.eow_id] = -float("inf")       # bare </w> after a finished word never occurs
                logits[:, n_dec:] = -float("inf")
                if allow_eos and min_words:
                    logits[words < min_words, eos_id] = -float("inf")
                n = dcfg.get("no_repeat_ngram") or 0
                if n > 1 and gen:
                    ban = ngram_ban(torch.stack(gen, 1), n, logits.size(1))
                    ban &= ~(ban | torch.isinf(logits)).all(-1, keepdim=True)
                    logits = logits.masked_fill(ban, -float("inf"))
                rp = dcfg.get("repetition_penalty") or 1.0
                prev = torch.stack(gen, 1) if (rp != 1.0 and gen) else None
                nxt = sample_next(logits, dcfg.get("temperature") or 0, dcfg.get("top_k") or 0, dcfg.get("top_p") or 1.0,
                                  banned, prev, rp)
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
                gen.append(nxt)
                is_eos = nxt == eos_id
                open_ = in_word | has_bytes[nxt]
                words += (eow[nxt] & open_ & ~done & ~is_eos).long()
                in_word = open_ & ~eow[nxt] & ~is_eos
                done |= is_eos | (words >= tgt_words)
                if bool(done.all()) or t == max_new - 1:
                    break
                h = step(nxt[:, None])
        rows = torch.stack(gen, 1).tolist()
        for r, i in enumerate(idx):
            ids = rows[r]
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            text = " ".join(tok.decode(ids).split())
            if budgets[i]:
                text = " ".join(text.split()[:budgets[i]])
            out_text[i] = text
        if log and (bi + 1) % 10 == 0:
            log(f"generated {min(s0 + bs, len(order))}/{len(order)}", seconds=round(time.time() - t0, 1))
    return out_text


# ---------------------------------------------------------------------------
# document-level entry points
# ---------------------------------------------------------------------------

def words_of(tok, ids):
    return len(tok.decode([t for t in ids if t >= tok.n_special]).split())


def summarize_single(model, tok, cfg, docs, src_enc, dcfg, device, log=None):
    """docs: [{"id", "document"}], src_enc: {id: sentence ids of the run's source text} -> {id: summary}."""
    fmt, sel = cfg["format"], cfg["selection"]
    kind = model.backbone.kind
    sep, start, eos = special_ids(tok, fmt, kind)
    exs, _ = summ_data.build_examples(docs, src_enc, None, fmt, sel, eos)
    L = dcfg["length"]
    budgets = []
    for d, e in zip(docs, exs):
        w = len(d["document"].split()) if L.get("basis", "document") == "document" else words_of(tok, e["src"])
        budgets.append(word_budget(L, w))
    texts = generate(model, tok, [e["src"] for e in exs], budgets, dcfg, sep, start, eos, device, log)
    return {d["id"]: t for d, t in zip(docs, texts)}


def summarize_stage1(model, tok, cfg, docs, doc_enc, dcfg, device, log=None):
    """-> {id: [summary of chunk 0, chunk 1, ...]} (chunks = sentence packing of the document into src_len tokens)."""
    fmt = cfg["format"]
    S = fmt["src_len"]
    kind = model.backbone.kind
    sep, start, eos = special_ids(tok, fmt, kind)
    L = dcfg["length"]
    srcs, budgets, owner = [], [], []
    for d in docs:
        chunks = summ_data.pack_chunks(doc_enc[d["id"]], S)
        dw, total = len(d["document"].split()), sum(map(len, chunks))
        for c in chunks:
            cw = words_of(tok, c)
            # basis "document": the chunk's share of the document's word budget (ratio x chunk words, same thing)
            budgets.append(word_budget(L, cw if L.get("basis", "document") == "input" else dw * len(c) / max(1, total)))
            srcs.append(summ_data.pad_left(c, S))
            owner.append(d["id"])
    if log:
        log("stage1 chunks", docs=len(docs), chunks=len(srcs), per_doc=round(len(srcs) / max(1, len(docs)), 2))
    texts = generate(model, tok, srcs, budgets, dcfg, sep, start, eos, device, log)
    out = {d["id"]: [] for d in docs}
    for o, t in zip(owner, texts):
        out[o].append(t)
    return out


def write_prediction(summaries, test_docs, path):
    pred = [{"id": d["id"], "summary": summaries[d["id"]]} for d in test_docs]
    assert len(pred) == len(test_docs) and len({p["id"] for p in pred}) == len(pred)
    assert all(isinstance(p["summary"], str) for p in pred)
    json.dump(pred, open(path, "w"), ensure_ascii=False)
    return pred


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def source_docs(cfg, split_docs, split):
    """Documents whose 'document' field is the run's SOURCE text (stage-2 runs: joined stage-1 chunk summaries)."""
    src = cfg.get("source") or {"type": "document"}
    if src["type"] == "document":
        return split_docs
    s1 = json.load(open(src[split]))
    missing = [d["id"] for d in split_docs if d["id"] not in s1]
    assert not missing, f"stage-1 file {src[split]} lacks {len(missing)} {split} documents (e.g. {missing[:3]})"
    return [{**d, "document": " ".join(s1[d["id"]])} for d in split_docs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--mode", choices=["single", "stage1"], default="single")
    ap.add_argument("--decode", help="JSON file or inline JSON; default = the run config's decode block")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--data-dir")
    ap.add_argument("--zip", action="store_true", help="test split, single mode: also write submission zip next to --out")
    a = ap.parse_args()
    if a.merge:
        out, parts = a.merge[0], a.merge[1:]
        merged = {}
        for p in parts:
            merged.update(json.load(open(p)))
        json.dump(merged, open(out, "w"), ensure_ascii=False)
        print(f"merged {len(parts)} parts -> {out} ({len(merged)} documents)")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    tok = ByteBPE.from_json(resolve_ckpt(cfg["tokenizer"]) if not os.path.exists(cfg["tokenizer"]) else cfg["tokenizer"])
    model, cfg = load_run(a.run, device, tok.get_vocab_size())
    dcfg = cfg["decode"]
    if a.decode:
        dcfg = json.loads(a.decode) if a.decode.strip().startswith("{") else json.load(open(a.decode))
    data_dir = a.data_dir or cfg["data_dir"]
    train, val, test, _ = summ_data.load_splits(data_dir, cfg["split"]["val_docs"], cfg["split"]["seed"])
    docs = {"train": train, "val": val, "test": test}[a.split]
    if a.limit:
        docs = docs[:a.limit]
    i, n = map(int, a.shard.split("/"))
    docs = docs[i::n]
    log = lambda msg, **kv: print(json.dumps({"msg": msg, **kv}), flush=True)
    t0 = time.time()
    if a.mode == "stage1":
        enc = summ_data.encode_sentences(tok, docs, "document")
        res = summarize_stage1(model, tok, cfg, docs, enc, dcfg, device, log)
    else:
        sdocs = source_docs(cfg, docs, a.split)
        enc = summ_data.encode_sentences(tok, sdocs, "document")
        res = summarize_single(model, tok, cfg, docs, enc, dcfg, device, log)
    json.dump(res, open(a.out, "w"), ensure_ascii=False)
    log("done", docs=len(docs), out=a.out, seconds=round(time.time() - t0, 1))
    if a.zip and a.split == "test" and a.mode == "single" and n == 1:
        import zipfile
        pred_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), "prediction.json")
        write_prediction(res, docs, pred_path)
        zp = os.path.splitext(a.out)[0] + ".zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pred_path, arcname="prediction.json")
        log("zip", path=zp)


if __name__ == "__main__":
    main()
