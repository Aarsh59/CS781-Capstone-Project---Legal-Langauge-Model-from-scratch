#!/usr/bin/env python3
"""Download Phase_2_LNER/ from the gated Hugging Face dataset (plumbing; declare it).

Token lookup order: Kaggle secret "HF" -> HF_TOKEN environment variable -> hidden prompt.
The token is used for these requests only: never printed, never written to disk.

  python fetch_data.py                      # -> ./data/Phase_2_LNER/...
  python fetch_data.py --out /kaggle/working/data
"""
import argparse
import getpass
import json
import os
import urllib.error
import urllib.request

REPO = "https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/resolve/main/"
FILES = [
    "Phase_2_LNER/Train/fold1-text.json",
    "Phase_2_LNER/Train/fold1-gold.json",
    "Phase_2_LNER/Train/fold2-text.json",
    "Phase_2_LNER/Train/fold2-gold.json",
    "Phase_2_LNER/Train/labels.txt",
    "Phase_2_LNER/Test/fold3-text.json",
]


def get_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF")
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token (input hidden): ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()
    if all(os.path.exists(os.path.join(args.out, rel)) for rel in FILES):
        print("all Phase-2 files already present in", args.out)
        return
    token = get_token()
    try:
        for rel in FILES:
            dst = os.path.join(args.out, rel)
            if os.path.exists(dst):
                print("exists:", dst)
                continue
            req = urllib.request.Request(REPO + rel, headers={"Authorization": f"Bearer {token}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                hint = "check the token / dataset access" if e.code in (401, 403) else e.reason
                raise SystemExit(f"download failed for {rel}: HTTP {e.code} ({hint})")
            if rel.endswith(".json"):
                json.loads(data)                       # validate before saving
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
            print(f"saved {dst} ({len(data) / 1e6:.1f} MB)")
    finally:
        del token


if __name__ == "__main__":
    main()
