#!/usr/bin/env python3
"""Download Phase_4_CJPE/ and small reference samples of the Phase-1 corpus (plumbing; declare it).

Token lookup order: Kaggle secret "HF" -> HF_TOKEN environment variable -> hidden prompt.
The token is used for these requests only: never printed, never written to disk.

  python fetch_data.py                 # -> ./data/Phase_4_CJPE/... and ./data/ref_samples/<source>.jsonl

Reference samples: for each Phase-1 source (supreme_court, acts, high_courts, tribunals) a few byte ranges at seeded
random offsets of the one-record-per-line corpus file are downloaded (HTTP Range) and only complete records are kept.
They are used to fit a unigram source classifier for the CJPE documents (analyze_cjpe.py), like the Phase-1 test check.
"""
import argparse
import getpass
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request

REPO = "https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/resolve/main/"
FILES = ["Phase_4_CJPE/Train/train.json", "Phase_4_CJPE/Train/dev.json", "Phase_4_CJPE/Test/test.json"]
SOURCES = ("supreme_court", "acts", "high_courts", "tribunals")
RANGES_PER_SOURCE, RANGE_BYTES, SEED = 12, 2_000_000, 7


def get_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF")
    except Exception:
        pass
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if not sys.stdin.isatty():
        raise SystemExit("No Hugging Face token: on Kaggle add a secret with the LABEL 'HF' (Add-ons -> Secrets) and tick "
                         "it for this notebook; locally set HF_TOKEN or run this script in a terminal.")
    return getpass.getpass("Hugging Face token (input hidden): ").strip()


def get(url, token, rng=None):
    h = {"Authorization": f"Bearer {token}"}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        hint = "check the token / dataset access" if e.code in (401, 403) else e.reason
        raise SystemExit(f"download failed for {url.split('/main/')[-1]}: HTTP {e.code} ({hint})")


def parse_line(line):
    line = line.strip().rstrip(",")
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--skip-ref", action="store_true", help="only the CJPE files (no corpus samples; for training sessions)")
    args = ap.parse_args()
    token = get_token()
    try:
        for rel in FILES:
            dst = os.path.join(args.out, rel)
            if os.path.exists(dst):
                print("exists:", dst); continue
            data, _ = get(REPO + rel, token)
            json.loads(data)                                   # validate before saving
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)
            print(f"saved {dst} ({len(data) / 1e6:.1f} MB)")
        rnd = random.Random(SEED)
        for s in ([] if args.skip_ref else SOURCES):
            dst = os.path.join(args.out, "ref_samples", f"{s}.jsonl")
            if os.path.exists(dst):
                print("exists:", dst); continue
            url = REPO + f"Phase_1_Pretraining/Train/train_{s}.json"
            _, hdr = get(url, token, (0, 0))
            size = int(re.search(r"/(\d+)$", hdr.get("Content-Range", hdr.get("content-range", ""))).group(1))
            recs = []
            for _ in range(RANGES_PER_SOURCE):
                a = rnd.randrange(0, max(1, size - RANGE_BYTES))
                chunk, _ = get(url, token, (a, a + RANGE_BYTES - 1))
                lines = chunk.decode("utf-8", errors="ignore").split("\n")[1:-1]   # drop the partial first / last line
                recs += [r["text"] for r in map(parse_line, lines) if r and r.get("text")]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                for t in recs:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            print(f"saved {dst}: {len(recs)} records from {RANGES_PER_SOURCE} ranges of a {size / 1e9:.2f} GB file")
    finally:
        del token


if __name__ == "__main__":
    main()
