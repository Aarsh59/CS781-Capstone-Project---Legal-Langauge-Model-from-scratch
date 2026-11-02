#!/usr/bin/env python3
"""Tokenize the WHOLE Phase-1 legal corpus with your legal_bpe.json into train.bin / val.bin (plumbing; declare it).

  HF_TOKEN=... python build_data.py --tokenizer legal_bpe.json --out data
  python build_data.py --tokenizer legal_bpe.json --out data --local-dir corpus/   # files already downloaded

Cleaning: whitespace normalised, documents shorter than --min-words dropped, exact duplicates (hash of the first 2000
chars) dropped, and (unless --no-fertility-filter) non-Supreme-Court documents with fertility > --max-fertility
tokens/word dropped (garbled OCR, and in practice most non-English text with this tokenizer). Unlike Phase 1 there is no mixture / repetition: every remaining document
of every source is used once. A hash-chosen --val-percent of Supreme Court documents is held out as val.bin
(Supreme Court because the test prompts are ~93% SC).

Output (uint16 token ids, each document followed by </s>):
  data/train.bin   all sources concatenated (training samples random windows, so order does not matter)
  data/val.bin     held-out SC documents
  data/manifest.json   per-source docs / tokens / drops, tokenizer sha256, build time
Resumable: each source is written to data/parts/<source>.bin with a .done marker; finished sources are skipped
(cross-source duplicate removal then only covers the sources built in the current invocation).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_tokenizer                      # noqa: E402
from a1_tokenizer import ByteBPE         # noqa: E402

REPO = "Exploration-Lab/CS-781-Capstone"
FILES = {
    "supreme_court": "Phase_1_Pretraining/Train/train_supreme_court.json",
    "acts": "Phase_1_Pretraining/Train/train_acts.json",
    "high_courts": "Phase_1_Pretraining/Train/train_high_courts.json",
    "tribunals": "Phase_1_Pretraining/Train/train_tribunals.json",
}


def parse_line(line):
    line = line.strip().rstrip(",")
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def iter_texts(path):
    """Stream the "text" field of a one-record-per-line JSON array file (no full load into memory)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = parse_line(line)
            if rec and rec.get("text"):
                yield rec["text"]


SCRIPTS = [("devanagari", 0x0900, 0x097F), ("bengali", 0x0980, 0x09FF), ("gurmukhi", 0x0A00, 0x0A7F),
           ("gujarati", 0x0A80, 0x0AFF), ("odia", 0x0B00, 0x0B7F), ("tamil", 0x0B80, 0x0BFF), ("telugu", 0x0C00, 0x0C7F),
           ("kannada", 0x0C80, 0x0CFF), ("malayalam", 0x0D00, 0x0D7F), ("arabic_urdu", 0x0600, 0x06FF)]


def dominant_script(text, n=2000):
    """Most frequent script among the letters of the first n characters ("latin", one of SCRIPTS, or "other")."""
    counts = {}
    for ch in text[:n]:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x250:
            k = "latin"
        else:
            k = next((name for name, lo, hi in SCRIPTS if lo <= o <= hi), "other")
        counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def doc_hash(text):
    return int(hashlib.md5(text[:2000].encode("utf-8", errors="ignore")).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-dir", default=None, help="folder with the four train_*.json files (skip download)")
    ap.add_argument("--sources", nargs="+", default=list(FILES))
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--max-fertility", type=float, default=2.0)
    ap.add_argument("--no-fertility-filter", action="store_true",
                    help="keep every document regardless of tokens/word (non-English scripts and noisy OCR included)")
    ap.add_argument("--val-percent", type=int, default=2)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-docs", type=int, default=None, help="smoke tests only: docs per source")
    ap.add_argument("--dry-stats", action="store_true",
                    help="only count documents/tokens per script (kept vs dropped); writes no token files")
    args = ap.parse_args()

    t_all = time.time()
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)
    tok = ByteBPE.from_json(args.tokenizer)
    tok_sha = hashlib.sha256(open(args.tokenizer, "rb").read()).hexdigest()
    V, EOT = tok.get_vocab_size(), tok.eot_id
    assert V <= 65535, "uint16 storage needs vocab_size <= 65535"
    print(f"tokenizer {args.tokenizer}: vocab {V}, </s> id {EOT}, sha256 {tok_sha[:8]} (Phase-1 legal BPE: 38f51f14)")

    paths = {}
    for name in args.sources:
        if args.local_dir:
            paths[name] = os.path.join(args.local_dir, os.path.basename(FILES[name]))
        else:
            from huggingface_hub import hf_hub_download
            paths[name] = hf_hub_download(repo_id=REPO, filename=FILES[name], repo_type="dataset",
                                          token=os.environ.get("HF_TOKEN"))
        print(f"{name}: {paths[name]} ({os.path.getsize(paths[name]) / 1e9:.2f} GB)", flush=True)

    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"sources": {}}
    seen = set()
    report = {}
    val_part = os.path.join(args.out, "parts", "val_sc.bin")
    for name in args.sources:
        part = os.path.join(args.out, "parts", f"{name}.bin")
        done = part + ".done"
        if os.path.exists(done) and not args.dry_stats:
            print(f"[{name}] already built, skipping", flush=True)
            continue
        stats = Counter()
        t0 = time.time()

        def cleaned():
            for raw in iter_texts(paths[name]):
                stats["docs_read"] += 1
                if args.max_docs and stats["docs_read"] > args.max_docs:
                    return
                text = " ".join(raw.split())
                scr = dominant_script(text)
                stats[f"script_read_{scr}"] += 1
                scripts.append(scr)
                if text.count(" ") + 1 < args.min_words:
                    stats["dropped_short"] += 1
                    continue
                h = doc_hash(text)
                if h in seen:
                    stats["dropped_duplicate"] += 1
                    continue
                seen.add(h)
                if name == "supreme_court" and h % 100 < args.val_percent:
                    stats["held_out_val"] += 1
                    yield ("val", text)
                    continue
                yield ("train", text)

        texts, kinds, scripts, kept_scripts = [], [], [], []

        def feed():
            for kind, text in cleaned():
                kinds.append(kind)
                kept_scripts.append(scripts[-1])          # script of the document being yielded
                yield text

        mode = "wb" if name == "supreme_court" else "ab"     # val.bin holds SC docs only; a rebuild starts it fresh
        out_path, val_path_ = (os.devnull, os.devnull) if args.dry_stats else (part, val_part)
        with open(out_path, "wb") as out, open(val_path_, mode) as vout, tok.pool(args.workers) as pool:
            for i, (arr, n_words) in enumerate(pool.imap(a1_tokenizer.encode_array_worker, feed(), chunksize=16)):
                kind, scr = kinds[i], kept_scripts[i]
                if (not args.no_fertility_filter and kind == "train" and name != "supreme_court"
                        and len(arr) / max(1, n_words) > args.max_fertility):
                    stats["dropped_fertility"] += 1
                    stats[f"script_dropped_fertility_{scr}"] += 1
                    stats[f"tokens_dropped_fertility_{scr}"] += len(arr)
                    continue
                if args.dry_stats:
                    stats[f"script_kept_{scr}"] += 1
                    stats[f"tokens_kept_{scr}"] += len(arr) + 1
                    stats["docs_kept"] += kind == "train"
                    continue
                stats[f"script_kept_{scr}"] += 1
                stats[f"tokens_kept_{scr}"] += len(arr) + 1
                arr = np.append(arr, np.uint16(EOT)).astype(np.uint16)
                (vout if kind == "val" else out).write(arr.tobytes())
                stats["val_tokens" if kind == "val" else "train_tokens"] += len(arr)
                stats["docs_kept"] += kind == "train"
                if stats["docs_kept"] and stats["docs_kept"] % 100_000 == 0 and kind == "train":
                    el = time.time() - t0
                    print(f"  [{name}] {stats['docs_kept']:,} docs | {stats['train_tokens'] / 1e6:,.0f}M tokens | "
                          f"{stats['train_tokens'] / el / 1e6:.2f}M tok/s", flush=True)
        stats["minutes"] = round((time.time() - t0) / 60, 1)
        if args.dry_stats:
            report[name] = dict(stats)
            print(f"[{name}] (dry) {dict(stats)}", flush=True)
            continue
        manifest["sources"][name] = dict(stats)
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        open(done, "w").write("ok")
        print(f"[{name}] {dict(stats)}", flush=True)
    if args.dry_stats:
        json.dump(report, open(os.path.join(args.out, "script_stats.json"), "w"), indent=2)
        print("dry run: per-script counts written to", os.path.join(args.out, "script_stats.json"))
        return

    # final train.bin / val.bin
    train_path, val_path = os.path.join(args.out, "train.bin"), os.path.join(args.out, "val.bin")
    with open(train_path, "wb") as out:
        for name in args.sources:
            with open(os.path.join(args.out, "parts", f"{name}.bin"), "rb") as f:
                shutil.copyfileobj(f, out, length=64 << 20)
    shutil.copyfile(val_part, val_path)
    n_train = os.path.getsize(train_path) // 2
    n_val = os.path.getsize(val_path) // 2
    probe = np.memmap(train_path, dtype=np.uint16, mode="r")
    assert int(probe[: min(len(probe), 50_000_000)].max()) < V, "token id >= vocab: tokenizer mismatch"
    manifest.update(tokenizer=os.path.basename(args.tokenizer), tokenizer_sha256=tok_sha, vocab_size=V, eot_id=EOT,
                    dtype="uint16", train_tokens=n_train, val_tokens=n_val, whitespace_normalised=True,
                    min_words=args.min_words, max_fertility=None if args.no_fertility_filter else args.max_fertility,
                    fertility_filter=not args.no_fertility_filter, val_percent=args.val_percent,
                    build_minutes_total=round((time.time() - t_all) / 60, 1))
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"train.bin {n_train:,} tokens ({n_train * 2 / 1e9:.1f} GB) | val.bin {n_val:,} tokens")
    print("parts/ can be deleted to free disk once you are happy with train.bin")


if __name__ == "__main__":
    main()
