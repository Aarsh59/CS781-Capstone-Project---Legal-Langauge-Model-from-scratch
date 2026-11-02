#!/usr/bin/env python3
"""Build the Kaggle notebook ~/capstone/capstone_phase6_summ.ipynb from the files in this folder (plumbing).
Every .py file becomes a %%writefile cell (your techniques/*.py and experiments.py are editable there too).
    python build_notebook_summ.py [--out PATH]"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working/phase6"

PLUMBING = ["a1_tokenizer.py"] + [f"phase1_models/{f}" for f in (
    "rope.py", "arch1.py", "arch3.py", "attention3.py", "non_causal_self_attention.py", "cross_attention.py",
    "encoder.py", "decoder4.py", "enc_dec.py", "prefix_attention.py", "prefix_lm.py",
    "fastattn_ab.py", "fastattn_cd.py")] + [
    "fetch_data.py", "summ_data.py", "gen_model.py", "inject.py", "techniques/__init__.py", "techniques/full_ft.py",
    "summ_infer.py", "score_summ.py", "train_summ.py", "run_many.py"]
YOURS = ["techniques/lora.py", "techniques/lorma.py", "techniques/qlora.py", "techniques/adalora.py",
         "techniques/adapters.py", "techniques/prefix_tuning.py", "techniques/prompt_tuning.py",
         "techniques/partial_ft.py", "techniques/bitfit.py", "techniques/kd.py", "experiments.py"]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def writefile(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return code(f"%%writefile {WORK}/{rel}\n" + f.read())


INTRO = """
# CS781 Capstone — Phase 6: Abstractive Legal Summarization on 2×T4

**Kaggle settings:** GPU T4 ×2, Internet **on**, secret **`HF`** or **`HF_TOKEN`** (ticked). **Add Input:** your private
Phase-1 model datasets (A, B2, C+D, optionally B3 = `ctxext/best_newctx.pt`) and, from session 2 on, earlier outputs of
this notebook (either account) so finished runs (teachers, stage-1 files) are reused.

**Plumbing (declare it):** head+tail source selection (`src_len` tokens) → target = reference truncated to `tgt_len`
(+ `</s>`) → Phase-1 LM with its tied head + your technique → token cross-entropy on the summary. A/B/D read
`[source][sep][summary]` (D bidirectional over source + sep), C encodes the source and decodes the summary. Hierarchical:
a finished run writes stage-1 chunk summaries (`after.stage1`), a stage-2 run (`source.type = "stage1"`) learns
chunk summaries → reference. Checkpoint = lowest validation loss on a seeded hold-out of train.json; summaries for
validation (ROUGE-1/L + BERTScore roberta-large, local proxy) and test → `prediction.json` (list format) + zip.

**Your parts (course rule 9):** `techniques/*.py`, `kd.py`, and every value in `experiments.py` (and the decoding lists in
Stage 7). Outputs: `/kaggle/working/phase6/runs/<name>/`, zips in `/kaggle/working/submissions/`.
"""

SETUP = f"""
import os, sys, json, glob, time, hashlib, subprocess, shutil, inspect
SESSION_START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rouge-score", "bert-score"], check=True)
WORK = "{WORK}"
for d in ("phase1_models", "techniques", "configs", "runs", "data", "cache", "sweeps"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
import torch
N_GPUS = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])
"""

DATA = """
# Phase-6 files from the gated HF dataset (Kaggle secret "HF"), Phase-1 inputs
from kaggle_secrets import UserSecretsClient
_hf, _errs = None, {}
for _label in ("HF", "HF_TOKEN"):
    try:
        _hf = UserSecretsClient().get_secret(_label)
        if _hf:
            print(f"Hugging Face token: Kaggle secret '{_label}'"); break
    except Exception as e:
        _errs[_label] = type(e).__name__
if not _hf:
    raise SystemExit(f"Could not read a Kaggle secret 'HF' or 'HF_TOKEN' ({_errs}): Add-ons -> Secrets, tick it, rerun.")
subprocess.run([sys.executable, "fetch_data.py", "--out", os.path.join(WORK, "data")], check=True,
               env=dict(os.environ, HF_TOKEN=_hf.strip()))
del _hf
DATA_DIR = os.path.join(WORK, "data", "Phase_6_Summ")

def find_all(pattern):
    return sorted(glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True))

def find_one(pattern):
    hits = find_all(pattern)
    if len(hits) > 1:
        print(f"  note: {len(hits)} matches for {pattern}, using the first: {hits}")
    return hits[0] if hits else None

def sha8(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]

TOKS = find_all("legal_bpe.json")
assert TOKS, "legal_bpe.json not found under /kaggle/input (Add Input: your Phase-1 outputs)"
tok_hashes = {t: sha8(t) for t in TOKS}
assert len(set(tok_hashes.values())) == 1, f"different legal_bpe.json files attached: {tok_hashes}"
TOKENIZER = TOKS[0]
print("tokenizer sha256:", tok_hashes[TOKENIZER], "(Phase-1 legal BPE was 38f51f14)")
PHASE1 = {"A": find_one("run_A/best.pt"), "B2": find_one("run_B2/best.pt"), "B3": find_one("ctxext/best_newctx.pt"),
          "C": find_one("run_C/best.pt"), "D": find_one("run_D/best.pt")}
for k, v in PHASE1.items():
    if v is None:
        print(f"Phase-1 {k}: NOT FOUND"); continue
    ck = torch.load(v, map_location="cpu", weights_only=False, mmap=True)
    c = ck["config"]
    lim = c["model"]["max_seq_len"] if "model" in c else (c["encoder"]["max_seq_len"], c["decoder"]["max_seq_len"])
    print(f"Phase-1 {k}: {v}\\n    tag={c.get('tag')} module={c['module']} kind={c.get('kind', 'decoder')} max_len={lim}")
    del ck
import summ_data
_tr, _va, _te, DATA_REPORT = summ_data.load_splits(DATA_DIR, 0, 0)
print(DATA_REPORT)
del _tr, _va, _te
"""

CONFIG = """
# Experiment matrix = experiments.py (YOUR values). RUN = what this session trains, in priority order.
import importlib, experiments, train_summ, techniques
importlib.reload(experiments)
EXPERIMENTS = experiments.build(PHASE1, DATA_DIR, TOKENIZER, os.path.join(WORK, "runs"), os.path.join(WORK, "cache"))
ACCOUNT = None                      # 1 / 2 (this account's share, experiments.SPLIT) or "all"
assert ACCOUNT in (1, 2, "all"), "set ACCOUNT = 1 or 2 (or 'all') in this cell"
RUN = experiments.SPLIT[1] + experiments.SPLIT[2] if ACCOUNT == "all" else experiments.SPLIT[ACCOUNT]

def find_run(name):
    # a finished run (summary.json + best_model.pt) here or in an attached earlier output of this notebook
    for d in [os.path.join(WORK, "runs", name)] + sorted(glob.glob(f"/kaggle/input/**/runs/{name}", recursive=True)):
        if all(os.path.exists(os.path.join(d, f)) for f in ("summary.json", "best_model.pt", "config.json")):
            if not json.load(open(os.path.join(d, "summary.json"))).get("smoke"):
                return d
    return None

def resolve_dependencies(name, cfg, queued_before):
    # kd_teacher / stage1_from: a run queued earlier in this session, or a finished (attached) run
    for key in ("kd_teacher", "stage1_from"):
        dep = cfg.pop(key, None)
        if not dep:
            continue
        here = os.path.join(WORK, "runs", dep)
        d = here if dep in queued_before else find_run(dep)
        assert d, f"{name}: {key} {dep} is neither queued before it nor a finished attached run"
        if key == "kd_teacher":
            cfg["kd"]["teacher_run"] = d
        else:
            cfg["source"] = {"type": "stage1", **{s: os.path.join(d, f"stage1_{s}.json") for s in ("train", "val", "test")}}
            cfg["wait_for_run"] = d
    return cfg

DONE = {n: find_run(n) for n in RUN if find_run(n)}
if DONE:
    print("already finished (reused, not trained again):", DONE)
CFG_PATHS = []
for i, name in enumerate(RUN):
    if name in DONE:
        continue
    cfg = resolve_dependencies(name, EXPERIMENTS[name], [n for n in RUN[:i] if n not in DONE])
    train_summ.check_config(cfg, None)
    path = os.path.join(WORK, "configs", f"{name}.json")
    json.dump(cfg, open(path, "w"), indent=2)
    CFG_PATHS.append(path)
    f_ = cfg["format"]
    print(f"{name:24s} | {cfg['model_key']:3s} | {cfg['technique']['name']:13s} | lr {cfg['train']['lr']} | "
          f"src {f_['src_len']} tgt {f_['tgt_len']} | source {cfg['source']['type']} | kd {bool(cfg.get('kd'))} | "
          f"length {cfg['decode']['length']}")
print(f"{len(CFG_PATHS)} runs queued")
"""

SMOKE = """
# Real-model smoke test BEFORE the long runs: every queued config unchanged (real checkpoint, technique, lengths,
# micro-batch) on 48 train documents, 6 optimizer steps, validation/test generation on 4 documents, save -> reload ->
# zip. Reports peak memory, training speed and generation seconds per document. Outputs: runs_smoke/ (deleted ckpts).
RUN_SMOKE = True
if RUN_SMOKE and CFG_PATHS:
    smoke_paths = []
    for path in CFG_PATHS:
        c = json.load(open(path))
        c["smoke"] = {"steps": 6, "docs": 48}
        c["out_dir"] = os.path.join(WORK, "runs_smoke")
        c["eval_generate"] = {**(c.get("eval_generate") or {}), "val_docs": 4, "bert_model": None}
        if c.get("after"):
            c["after"]["stage1"] = {**c["after"]["stage1"], "train_docs": 4}
        if c.get("kd") and c["kd"]["teacher_run"].startswith(os.path.join(WORK, "runs")):
            c["kd"]["teacher_run"] = os.path.join(WORK, "runs_smoke", os.path.basename(c["kd"]["teacher_run"].rstrip("/")))
        if c["source"]["type"] == "stage1" and c["source"]["train"].startswith(os.path.join(WORK, "runs")):
            d = os.path.join(WORK, "runs_smoke", os.path.basename(os.path.dirname(c["source"]["train"])))
            c["source"] = {"type": "stage1", **{s: os.path.join(d, f"stage1_{s}.json") for s in ("train", "val", "test")}}
        if c["technique"]["name"] == "adalora":
            c["technique"].update(t_init=0, t_final=0, delta_t=1)
        if c.get("wait_for_run", "").startswith(os.path.join(WORK, "runs")):
            c["wait_for_run"] = os.path.join(WORK, "runs_smoke", os.path.basename(c["wait_for_run"].rstrip("/")))
        sp = os.path.join(WORK, "configs", f"smoke_{c['name']}.json")
        json.dump(c, open(sp, "w"), indent=2)
        smoke_paths.append(sp)
    t0 = time.time()
    sp_proc = subprocess.run([sys.executable, "run_many.py", "--gpus", ",".join(str(i) for i in range(max(1, N_GPUS))),
                              "--every", "60", *smoke_paths])
    print(f"\\nsmoke test: {(time.time() - t0) / 60:.1f} min")
    for path in CFG_PATHS:
        c = json.load(open(path))
        sm = os.path.join(WORK, "runs_smoke", c["name"], "summary.json")
        if not os.path.exists(sm):
            print(f"  FAIL {c['name']:24s} -> {os.path.join(WORK, 'runs_smoke', c['name'] + '.stdout')}"); continue
        s_ = json.load(open(sm))
        print(f"  OK   {c['name']:24s} peak {s_['peak_mem_GB']} GB | trainable {s_['params']['trainable']:,} "
              f"({s_['params']['trainable_pct']}%) | {s_['train_tok_s']} tok/s | gen {s_['test_gen_s_per_doc']} s/doc")
    assert sp_proc.returncode == 0, "smoke test failed: fix the runs marked FAIL before launching the full runs"
    # ---- projected schedule from the MEASURED smoke speeds (upper bound: every example = src_len + tgt_len tokens) ----
    est, deps = {}, {}
    for path in CFG_PATHS:
        c = json.load(open(path)); s_ = json.load(open(os.path.join(WORK, "runs_smoke", c["name"], "summary.json")))
        f_, t_ = c["format"], c["train"]
        n_tr = DATA_REPORT["train"] - c["split"]["val_docs"]
        tok_ep = n_tr * (f_["src_len"] + f_["tgt_len"])
        train_min = t_["epochs"] * tok_ep / max(1, s_["train_tok_s"]) / 60
        steps = t_["epochs"] * -(-n_tr // t_["batch_examples"])
        n_eval = steps // (t_.get("eval_every_steps") or 10 ** 9) + t_["epochs"]
        eval_min = n_eval * c["split"]["val_docs"] * (f_["src_len"] + f_["tgt_len"]) / (3 * max(1, s_["train_tok_s"])) / 60
        gen_min = ((c.get("eval_generate") or {}).get("val_docs", 0) + DATA_REPORT["test"]) * s_["test_gen_s_per_doc"] / 60
        est[c["name"]] = dict(train=train_min, total=train_min + eval_min + gen_min + 2, cap=t_.get("max_minutes"))
        deps[c["name"]] = [os.path.basename(d.rstrip("/")) for d in
                           ([c["kd"]["teacher_run"]] if c.get("kd") else []) + ([c["wait_for_run"]] if c.get("wait_for_run") else [])]
    t_setup = (time.time() - SESSION_START) / 60 + 5
    limit = 12 * 60 - 20
    free = [t_setup] * max(1, N_GPUS); end, queue = {}, [json.load(open(p))["name"] for p in CFG_PATHS]
    while queue:
        g = min(range(len(free)), key=lambda i: free[i])
        ready = [n for n in queue if all(d in end or d not in est for d in deps[n])]
        n = ready[0] if ready else queue[0]
        start = max([free[g]] + [end[d] for d in deps[n] if d in end])
        end[n] = start + est[n]["total"]; free[g] = end[n]; queue.remove(n)
        cap = est[n]["cap"]
        warn = ""
        if cap and cap < est[n]["train"]:
            warn += f"  !! max_minutes {cap} < estimated training {est[n]['train']:.0f} min: training would be cut short"
        if cap and start + cap + 10 > limit:
            warn += f"  !! start {start:.0f} + max_minutes {cap} + 10 > {limit}: the scheduler would SKIP this run"
        print(f"  {n:24s} GPU{g} start {start / 60:4.1f} h  end {end[n] / 60:4.1f} h  (train ~{est[n]['train']:.0f} min, "
              f"total ~{est[n]['total']:.0f} min, cap {cap}){warn}")
    print(f"PROJECTED: all runs done ~{max(end.values()) / 60:.1f} h into the session (+ Stage 7 sweep ~4 min/run); "
          f"limit {limit / 60:.1f} h. Smoke speeds are measured on 6 steps (slightly pessimistic).")
    for ck in glob.glob(os.path.join(WORK, "runs_smoke", "*", "best_model.pt")):
        os.remove(ck)
    print("SMOKE TEST PASSED -> run the next cell")
"""

LAUNCH = """
# Full runs: both GPUs in parallel (shared queue; a KD run waits for its teacher, a stage-2 run for its stage-1 run).
# Runs that cannot finish before the session limit (train.max_minutes + margin) are not started.
SESSION_HOURS = 12
STOP_AT = SESSION_START + SESSION_HOURS * 3600 - 20 * 60
proc = subprocess.run([sys.executable, "run_many.py", "--gpus", ",".join(str(i) for i in range(max(1, N_GPUS))),
                       "--every", "600", "--stop-at", str(STOP_AT), *CFG_PATHS])
print("exit code", proc.returncode)
"""

SWEEP = """
# Stage 7 — decoding sweep on finished runs (YOUR decode settings, course rule 9). Each entry generates validation
# summaries with summ_infer.py (both GPUs: half of the documents each), scores them (ROUGE-1/L + BERTScore) and, if
# "test": True, also writes a test submission zip. Results: sweeps/sweep_scores.jsonl.
# YOUR length caps per Phase-1 model (2026-11-02). Every finished run of this account (or attached) is re-decoded with
# its OWN decode settings, only decode.length.max_words replaced; validation = the run's full hold-out split.
MAX_WORDS = {"A": 210, "B2": 210, "C": 180, "D": 225}
VAL_LIMIT = None                    # None = all validation documents of the run's split (500); an int = first N
SWEEP = []                          # {"run", "tag", "decode", "test", "val_limit"}; extra manual entries can be appended
for _name, (_mk, _) in experiments.MATRIX.items():
    _rd = find_run(_name)
    if not _rd:
        print(f"sweep: {_name} not finished / not attached here -> skipped"); continue
    _dec = json.loads(json.dumps(json.load(open(os.path.join(_rd, "config.json")))["decode"]))
    _dec["length"]["max_words"] = MAX_WORDS[_mk]
    SWEEP.append({"run": _name, "tag": f"maxw{MAX_WORDS[_mk]}", "decode": _dec, "test": True, "val_limit": VAL_LIMIT})
print(f"{len(SWEEP)} sweep entries:", [(s_["run"], s_["tag"]) for s_ in SWEEP])
def _gen(run_dir, split, dec, out, limit=None):
    gpus = list(range(max(1, N_GPUS)))
    procs, parts = [], []
    for g in gpus:
        part = f"{out}.part{g}"
        cmd = [sys.executable, "summ_infer.py", "--run", run_dir, "--split", split, "--mode", "single",
               "--decode", json.dumps(dec), "--out", part, "--shard", f"{g}/{len(gpus)}"]
        if limit:
            cmd += ["--limit", str(limit)]
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
        parts.append(part)
    assert all(p.wait() == 0 for p in procs), f"generation failed for {out}"
    subprocess.run([sys.executable, "summ_infer.py", "--merge", out, *parts], check=True)

import score_summ
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for s in SWEEP:
    rd = find_run(s["run"])
    if not rd:
        print(f"run {s['run']} not found -> skipped"); continue
    _split = json.load(open(os.path.join(rd, "config.json")))["split"]
    base = os.path.join(WORK, "sweeps", f"{s['run']}__{s['tag']}")
    t0 = time.time()
    _gen(rd, "val", s["decode"], base + "_val.json", s.get("val_limit"))
    refs = json.load(open(os.path.join(rd, "val_refs.json"))) if not s.get("val_limit") else None
    if refs is None or len(refs) < len(json.load(open(base + "_val.json"))):
        _tr, _va, _te, _ = summ_data.load_splits(DATA_DIR, _split["val_docs"], _split.get("seed", 0))
        gen_ids = set(json.load(open(base + "_val.json")))
        refs = {d["id"]: d["summary"] for d in _va if d["id"] in gen_ids}
    preds = json.load(open(base + "_val.json"))
    res = {"run": s["run"], "tag": s["tag"], "decode": s["decode"],
           **score_summ.score(preds, refs, "roberta-large", "cuda"), "val_gen_min": round((time.time() - t0) / 60, 1)}
    if s.get("test"):
        _gen(rd, "test", s["decode"], base + "_test.json")
        _tr, _va, _te, _ = summ_data.load_splits(DATA_DIR, 0, 0)
        import summ_infer
        summ_infer.write_prediction(json.load(open(base + "_test.json")), _te, base + "_prediction.json")
        import zipfile
        zp = f"/kaggle/working/submissions/submission_{s['run']}__{s['tag']}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(base + "_prediction.json", arcname="prediction.json")
        res["zip"] = zp
    print(json.dumps(res))
    with open(os.path.join(WORK, "sweeps", "sweep_scores.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\\n")
"""

COLLECT = """
# Collect every finished run's zip + a results table (facts only)
os.makedirs("/kaggle/working/submissions", exist_ok=True)
rows = []
for sp in sorted(glob.glob(os.path.join(WORK, "runs", "*", "summary.json"))):
    s = json.load(open(sp))
    if s.get("smoke"):
        continue
    d = os.path.dirname(sp)
    for z in glob.glob(os.path.join(d, "submission_*.zip")):
        shutil.copy(z, "/kaggle/working/submissions/")
    v = s.get("val_scores") or {}
    rows.append((s["name"], s["phase1"].get("tag"), s["technique"]["name"], s["kd"], s["source"]["type"],
                 s["params"]["total"], s["params"]["trainable_pct"], s["best_val_loss"], v.get("bertscore_f1"),
                 v.get("rouge1_f1"), v.get("rougeL_f1"), v.get("pred_words"), v.get("ref_words"), v.get("rep4"),
                 round(s["train_seconds"] / 60, 1), s["peak_mem_GB"], s["test_gen_s_per_doc"]))
hdr = ("run", "phase1", "technique", "kd", "source", "params", "trainable%", "val_loss", "BERTScore", "R1", "RL",
       "words", "ref_words", "rep4", "train_min", "peak_GB", "gen_s/doc")
print("\\t".join(hdr))
for r in rows:
    print("\\t".join(str(x) for x in r))
print(sorted(os.listdir("/kaggle/working/submissions")))
"""


def build(out):
    cells = [md(INTRO), md("## Stage 0 — setup"), code(SETUP), md("## Stage 1 — plumbing files (declare)")]
    cells += [writefile(f) for f in PLUMBING]
    cells += [md("## Stage 2 — YOUR files (course rule 9): techniques, KD loss, experiment values")]
    cells += [writefile(f) for f in YOURS]
    cells += [md("## Stage 3 — data + Phase-1 inputs"), code(DATA),
              md("## Stage 4 — experiment queue (set ACCOUNT)"), code(CONFIG),
              md("## Stage 5 — real-model smoke test"), code(SMOKE),
              md("## Stage 6 — full runs (2 GPUs)"), code(LAUNCH),
              md("## Stage 7 — decoding sweep (your settings)"), code(SWEEP),
              md("## Stage 8 — collect zips + table"), code(COLLECT)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "capstone_phase6_summ.ipynb"))
    build(ap.parse_args().out)
