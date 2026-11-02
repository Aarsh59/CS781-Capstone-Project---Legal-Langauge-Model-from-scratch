#!/usr/bin/env python3
"""Download the Phase 1 test set (test.json) from the gated Hugging Face dataset.

The token is read from the HF_TOKEN environment variable if set, otherwise asked for at a hidden
prompt. It is used for this one request only: never printed, never written to disk.

  python3 fetch_test.py                       # -> ~/capstone/phase1/ab_inputs/test.json
  python3 fetch_test.py --out some/dir/test.json
"""
import argparse
import getpass
import json
import os
import urllib.error
import urllib.request

URL = ("https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/resolve/main/"
       "Phase_1_Pretraining/Test/test.json")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/capstone/phase1/ab_inputs/test.json"))
args = ap.parse_args()

token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token (input hidden): ").strip()
req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {token}"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
except urllib.error.HTTPError as e:
    raise SystemExit(f"download failed: HTTP {e.code} ({'check the token / dataset access' if e.code in (401, 403) else e.reason})")
finally:
    del token

test = json.loads(data)                               # validate before saving
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "wb") as f:
    f.write(data)
first = next(iter(test.values()))
print(f"saved {args.out}: {len(test)} prompts, ~{len(first['text'].split())} words in the first one")
