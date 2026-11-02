#!/usr/bin/env python3
"""Download Phase_6_Summ/ (plumbing; declare it).

Token lookup order: Kaggle secret "HF" -> HF_TOKEN environment variable -> hidden prompt.
The token is used for these requests only: never printed, never written to disk.

  python fetch_data.py                 # -> ./data/Phase_6_Summ/{Train/train.json,Test/test.json}
"""
import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/resolve/main/"
FILES = ["Phase_6_Summ/Train/train.json", "Phase_6_Summ/Test/test.json"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
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
    finally:
        del token


if __name__ == "__main__":
    main()
