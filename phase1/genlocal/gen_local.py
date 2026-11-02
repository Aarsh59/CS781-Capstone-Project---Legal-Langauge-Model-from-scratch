#!/usr/bin/env python3
"""Local generation for the decoder-only models A and B (CS781 capstone, Phase 1). Plumbing; declare it.

Loads a trained best.pt (its config travels inside the checkpoint), then for every decoding strategy:
  run_<tag>/prediction_<strategy>.json        1000 test continuations  {id: {"text": ...}}
  run_<tag>/val_generations_<strategy>.json   held-out prompts + references, for score_local.py
  run_<tag>/submission_<tag>_<strategy>.zip   prediction.json at the archive root (Codabench format)
Generation is the exact code path of the Kaggle runs (fastattn.generate_words / sample_next from the
arch12 notebook; sample_next is identical to the one C and D used), so A/B vs C/D compare like for like.

Optional --matched-loss: loss on the same validation windows as C/D's eval (seed 1234, 360 windows of
897 tokens), on positions 640..896 only (what C and D are scored on) and on the full window.

Decoding strategy keys: temperature, top_k, top_p, repetition_penalty, and optionally
  no_repeat_ngram (n > 1: never repeat an n-gram of token ids; ~1.2-1.3 tokens per word) and
  ngram_scope ("output": generated text only, the default | "all": prompt + generated text).
MBR (minimum Bayes risk) keys: mbr_samples (K sampled candidates per prompt, needs temperature > 0),
  mbr_utility ("rougeL" | "bleu" | "chrf": formula-only agreement, no pretrained model), mbr_include_greedy
  (also add the temperature-0 output with the same blocking as a candidate). Costs ~K x the generation time.
Beam search keys: beam_size (W beams; temperature is ignored), beam_length_penalty (score = sum log p / len**lp,
  default 1.0 = mean log-prob per token); combines with no_repeat_ngram / ngram_scope. Costs ~W x generation time.

Usage (from ~/capstone/phase1/genlocal):
  python gen_local.py --ckpt A=PATH/run_A/best.pt --ckpt B=PATH/run_B/best.pt \
      --tokenizer PATH/legal_bpe.json --test PATH/test.json --val PATH/val_prompts.json --out out_ab
  python gen_local.py ... --val-only --decoding my_sweep.json     # decoding sweep on val prompts only
  python gen_local.py ... --matched-loss --val-bin PATH/val.bin    # position-matched loss for A/B
"""
import argparse
import importlib
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastattn                                    # noqa: E402
from a1_tokenizer import ByteBPE                   # noqa: E402

# Same three strategies (your values) as the C/D run. Override with --decoding FILE.json.
DECODING = {
    "greedy":       dict(temperature=0.0, top_k=None, top_p=1.0,  repetition_penalty=1.0),
    "mild":         dict(temperature=0.7, top_k=50,   top_p=0.90, repetition_penalty=1.10),
    "conservative": dict(temperature=0.5, top_k=50,   top_p=0.90, repetition_penalty=1.10),
}


def log(path, msg, **kv):
    rec = {"t": round(time.time(), 1), "msg": msg, **kv}
    print(json.dumps(rec), flush=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def looping_share(texts):
    """share of outputs in which some 4-word sequence occurs 3+ times (same as the Kaggle runs)."""
    n = 0
    for t in texts:
        w = t.split()
        grams = {}
        for i in range(len(w) - 3):
            g = tuple(w[i:i + 4])
            grams[g] = grams.get(g, 0) + 1
        n += bool(grams) and max(grams.values()) >= 3
    return round(n / max(1, len(texts)), 3)


def mbr_select(cands, utility):
    """Minimum Bayes Risk selection with a FORMULA-ONLY agreement measure (no pretrained model, Phase 1 rule 7.6).
    cands: K lists of P strings (K candidates per prompt). For each prompt returns the candidate with the highest
    mean utility against the other K-1 candidates (the 'consensus' continuation)."""
    if utility == "rougeL":
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        u = lambda hyp, ref: sc.score(ref, hyp)["rougeL"].fmeasure
    elif utility in ("bleu", "chrf"):
        import sacrebleu
        f = sacrebleu.sentence_bleu if utility == "bleu" else sacrebleu.sentence_chrf
        u = lambda hyp, ref: f(hyp, [ref]).score / 100
    elif utility == "bertscore":
        # ABLATION ONLY: uses the pretrained roberta-large inside BERTScore to pick outputs. Not allowed for a
        # Phase 1 submission (Capstone Document 7.6); requires --allow-pretrained-utility and is never zipped
        # under a submittable name.
        return _mbr_bertscore(cands)
    else:
        raise ValueError(f"mbr_utility must be rougeL, bleu or chrf (formula-only), not {utility!r}")
    K, P = len(cands), len(cands[0])
    out, agree = [], []
    for p in range(P):
        texts = [cands[k][p] for k in range(K)]
        scores = [np.mean([u(texts[i], texts[j]) for j in range(K) if j != i]) for i in range(K)]
        best = int(np.argmax(scores))
        out.append(texts[best])
        agree.append(scores[best])
    return out, float(np.mean(agree))


def _mbr_bertscore(cands):
    """pairwise BERTScore F1 (roberta-large, bert_score defaults, as in score_local.py) among the K candidates."""
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", device="cuda" if torch.cuda.is_available() else "cpu", batch_size=64)
    K, P = len(cands), len(cands[0])
    pairs = [(p, i, j) for p in range(P) for i in range(K) for j in range(i + 1, K)]
    _, _, f1 = scorer.score([cands[i][p] for p, i, j in pairs], [cands[j][p] for p, i, j in pairs])
    m = np.zeros((P, K, K))
    for (p, i, j), f in zip(pairs, f1.tolist()):
        m[p, i, j] = m[p, j, i] = f                     # F1 is symmetric
    mean = m.sum(-1) / (K - 1)
    best = mean.argmax(-1)
    del scorer
    torch.cuda.empty_cache()
    return [cands[best[p]][p] for p in range(P)], float(mean.max(-1).mean())


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    arch = importlib.import_module(cfg["module"])
    model = arch.DecoderLM(arch.LMConfig(**cfg["model"]))
    model.load_state_dict(ck["model"])
    fastattn.patch_model(model)                    # fused attention + KV cache (same parameters)
    model.to(device).eval()
    info = dict(module=cfg["module"], step=ck.get("step"), tokens_M=round(ck.get("tokens", 0) / 1e6, 1),
                best_val=ck.get("best_val"), params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    return model, cfg, info


@torch.no_grad()
def matched_loss(model, val_bin, seq_len, prefix, device, amp_dtype, n_windows=360, mb=9, seed=1234):
    """Mean next-token loss on C/D's validation windows: positions prefix..seq_len (as C/D) and full window."""
    mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
    span = seq_len + 1
    starts = np.random.default_rng(seed).integers(0, len(mm) - span - 1, n_windows)
    tail, full = [], []
    for i in range(0, n_windows, mb):
        arr = torch.from_numpy(np.stack([mm[s:s + span] for s in starts[i:i + mb]]).astype(np.int64)).to(device)
        x, y = arr[:, :-1], arr[:, 1:]
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(x)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        nll = nll.view(y.shape)
        tail.append(nll[:, prefix - 1:].mean().item())   # predictions of tokens prefix..seq_len (C/D's targets)
        full.append(nll.mean().item())
    return float(np.mean(tail)), float(np.mean(full))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, help="TAG=path/to/best.pt (repeatable)")
    ap.add_argument("--tokenizer", required=True, help="legal_bpe.json the checkpoint was trained with")
    ap.add_argument("--test", default=None, help="test.json ({id: {'text': prompt}})")
    ap.add_argument("--val", default=None, help="val_prompts.json ({id: {'text', 'reference'}})")
    ap.add_argument("--n-val", type=int, default=100, help="first N val prompts (C/D used 100)")
    ap.add_argument("--out", default="out_ab")
    ap.add_argument("--decoding", default=None, help="JSON file {name: {temperature, top_k, top_p, repetition_penalty}}")
    ap.add_argument("--only", default=None, help="comma-separated strategy names to run")
    ap.add_argument("--val-only", action="store_true", help="skip the test set (decoding sweeps)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--target-words", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42, help="C/D runs used SEED = 42")
    ap.add_argument("--allow-pretrained-utility", action="store_true",
                    help="ABLATION ONLY: permit mbr_utility='bertscore' (pretrained roberta-large; not allowed for submission)")
    ap.add_argument("--matched-loss", action="store_true")
    ap.add_argument("--val-bin", default=None, help="val.bin for --matched-loss (same tokenizer as the model)")
    ap.add_argument("--prefix", type=int, default=640, help="--matched-loss: first scored target position")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.float16 if device.type == "cuda" else torch.float32   # fp16 as on the T4s
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no CUDA device, running on CPU (slow; fine only for a smoke test)")

    decoding = json.load(open(args.decoding)) if args.decoding else DECODING
    if any(d.get("mbr_utility") == "bertscore" for d in decoding.values()) and not args.allow_pretrained_utility:
        sys.exit("mbr_utility='bertscore' uses a pretrained model (not allowed for Phase 1 submissions); "
                 "pass --allow-pretrained-utility to run it as an ablation")
    if args.only:
        decoding = {k: decoding[k] for k in args.only.split(",")}
    tokenizer = ByteBPE.from_json(args.tokenizer)
    test = json.load(open(args.test, encoding="utf-8")) if (args.test and not args.val_only) else None
    if not args.val_only and test is None:
        sys.exit("--test is required unless --val-only")
    vp = json.load(open(args.val, encoding="utf-8")) if args.val else None
    vkeys = list(vp)[:args.n_val] if vp else []

    for spec in args.ckpt:
        tag, path = spec.split("=", 1)
        out = os.path.join(args.out, f"run_{tag}")
        os.makedirs(out, exist_ok=True)
        logf = os.path.join(out, "log.jsonl")
        model, cfg, info = load_model(os.path.expanduser(path), device)
        T = cfg["model"]["max_seq_len"]
        log(logf, "model", tag=tag, **info, seq_len=T, max_prompt=T - args.max_new_tokens)

        if args.matched_loss:
            assert args.val_bin, "--matched-loss needs --val-bin"
            tail, full = matched_loss(model, os.path.expanduser(args.val_bin), T, args.prefix, device, amp)
            log(logf, "matched_loss", tag=tag, positions=f"{args.prefix}-{T}", loss_tail=round(tail, 4),
                loss_full_window=round(full, 4))

        def run_gen(prompts, dec):
            bs = args.batch_size
            while True:
                try:
                    return fastattn.generate_words(
                        model, tokenizer, prompts, max_seq_len=T, target_words=args.target_words,
                        max_new_tokens=args.max_new_tokens, batch_size=bs, temperature=dec["temperature"],
                        top_k=dec.get("top_k"), top_p=dec.get("top_p"),
                        repetition_penalty=dec.get("repetition_penalty", 1.0), device=device, amp_dtype=amp,
                        no_repeat_ngram=dec.get("no_repeat_ngram", 0), ngram_scope=dec.get("ngram_scope", "output"))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bs //= 2
                    assert bs >= 1
                    print(f"OOM, retrying with batch size {bs}")

        mbr_cache = {}

        def decode(prompts, dec, name):
            """plain decoding, or MBR when dec has mbr_samples: K sampled candidates (+ optionally the
            temperature-0 output with the same blocking) and the consensus pick by mbr_utility."""
            if dec.get("beam_size"):
                assert not dec.get("mbr_samples"), "beam_size and mbr_samples are separate strategies"
                bs = args.batch_size
                while True:
                    try:
                        return fastattn.generate_beam(
                            model, tokenizer, prompts, max_seq_len=T, target_words=args.target_words,
                            max_new_tokens=args.max_new_tokens, batch_size=bs, beam_size=dec["beam_size"],
                            device=device, amp_dtype=amp, no_repeat_ngram=dec.get("no_repeat_ngram", 0),
                            ngram_scope=dec.get("ngram_scope", "output"),
                            length_penalty=dec.get("beam_length_penalty", 1.0))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        bs //= 2
                        assert bs >= dec["beam_size"], "out of memory even with one prompt per batch"
                        print(f"OOM, retrying beam search with batch size {bs}")
            k = int(dec.get("mbr_samples", 0))
            if not k:
                return run_gen(prompts, dec)
            assert dec["temperature"] > 0, "MBR needs sampled candidates (temperature > 0)"
            ckey = json.dumps({**{x: y for x, y in dec.items() if x != "mbr_utility"}, "_p": hash(tuple(prompts))},
                              sort_keys=True)
            if ckey in mbr_cache:                              # same candidates, different utility: reuse
                cands = mbr_cache[ckey]
            else:
                cands = []
                for i in range(k):
                    torch.manual_seed(args.seed * 1000 + i)
                    cands.append(run_gen(prompts, dec))
                if dec.get("mbr_include_greedy"):
                    cands.append(run_gen(prompts, dict(dec, temperature=0.0)))
                mbr_cache.clear()
                mbr_cache[ckey] = cands
            t0 = time.time()
            picked, agree = mbr_select(cands, dec.get("mbr_utility", "rougeL"))
            log(logf, "mbr", tag=tag, strategy=name, candidates=len(cands), utility=dec.get("mbr_utility", "rougeL"),
                mean_agreement=round(agree, 4), select_s=round(time.time() - t0, 1))
            return picked

        for name, dec in decoding.items():
            if test is not None:
                ids = list(test)
                torch.manual_seed(args.seed)
                t0 = time.time()
                preds = decode([test[k]["text"] for k in ids], dec, name)
                prediction = {k: {"text": p} for k, p in zip(ids, preds)}
                ppath = os.path.join(out, f"prediction_{name}.json")
                with open(ppath, "w", encoding="utf-8") as f:
                    json.dump(prediction, f, ensure_ascii=False, indent=2)
                zname = (f"DO_NOT_SUBMIT_{tag}_{name}.zip" if dec.get("mbr_utility") == "bertscore"
                         else f"submission_{tag}_{name}.zip")
                with zipfile.ZipFile(os.path.join(out, zname), "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(ppath, arcname="prediction.json")
                log(logf, "prediction written", tag=tag, strategy=name, decoding=dec, n=len(preds),
                    minutes=round((time.time() - t0) / 60, 2),
                    under_target_words=sum(len(p.split()) < args.target_words for p in preds),
                    looping_share=looping_share(preds))
            if vkeys:
                torch.manual_seed(args.seed)
                vpreds = decode([vp[k]["text"] for k in vkeys], dec, name)
                with open(os.path.join(out, f"val_generations_{name}.json"), "w", encoding="utf-8") as f:
                    json.dump({k: {"prediction": p, "reference": vp[k]["reference"]} for k, p in zip(vkeys, vpreds)},
                              f, ensure_ascii=False, indent=2)
                log(logf, "val_generations written", tag=tag, strategy=name, n=len(vkeys),
                    looping_share=looping_share(vpreds))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("done ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
