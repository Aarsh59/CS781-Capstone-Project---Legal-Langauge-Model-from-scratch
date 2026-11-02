#!/usr/bin/env python3
"""Phase 3 artifact discovery, integrity audit, recomputed metrics and analysis bundle (plumbing; declare it).

Read-only on every existing artifact. Writes only under artifact_audit/ (this folder):
  experiment_inventory.csv, integrity_issues.csv, ensembles.json, submissions.csv,
  metrics/ (per_run_summary.csv, per_class.csv, per_document.csv, confusion/<run>.csv, pairwise_disagreement.csv),
  bundle/ (dev_gold.csv, test_meta.csv, label_vocab.json, hard_predictions_{dev,test}.csv,
           probs/<run>_{dev,test}.csv, evaluation/ copy of rr_metrics.py, README.md)
Original metrics (summary.json, Codabench scores) are kept in their own columns, never replaced by recomputed ones.

  python artifact_audit/audit_predictions.py
"""
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)
import rr_data                                   # noqa: E402
from rr_ensemble import vote                     # noqa: E402
from rr_metrics import rr_scores                 # noqa: E402

H = os.path.expanduser
SOURCES = {   # campaign -> run folder (each run: config/summary/log/dev+test predictions/probs)
    "campaign3_allep": H("~/kaggle_outputs/allep/phase3/runs"),
    "campaign2": H("~/kaggle_outputs/notebooka3fd7ebab3/phase3/runs"),
}
C2_CONFIGS = H("~/kaggle_outputs/notebooka3fd7ebab3/phase3/configs")
C2_ENSEMBLES = H("~/kaggle_outputs/notebooka3fd7ebab3/phase3/ensembles")
C1_ZIPS = H("~/capstone/Phase3 Submissions")
OTHER_ZIP_DIRS = [H("~/Downloads"), H("~/capstone/Phase3 Submissions/dump")]
ENS_EXP = os.path.join(P3, "ens_exp_out")
ENS_FAM = os.path.join(P3, "ens_family", "out")
OUT = HERE
DATA = os.path.join(P3, "data")

# Codabench scores as transcribed from the user's screenshots (codabench_phase3_scores.md): name -> (score, id)
CODABENCH = {
    "B_lorma": (.4559, 991111), "B_lora_lr2e4": (.4522, 991107), "B_rslora_r16": (.4517, 991135), "KD": (.4501, 991125),
    "B_lora_r16": (.4461, 991110), "B_rslora": (.4460, 991130), "B_qlora": (.4055, 991117), "B_lora": (.4015, 991104),
    "B_lora_lr5e5": (.3937, 991108), "B_lora_ctx_local": (.3927, 991105), "B_lora_inv_freq": (.3868, 991106),
    "B_prompt_p16": (.3816, 991116), "D_run": (.3801, 991123), "B_prefix_tuning": (.3793, 991112), "B_adalora_r4": (.3636, 991073),
    "B_adalora": (.3632, 991066), "B_ia3": (.3623, 991081), "B_prompt_tuning": (.3552, 991114), "A_run": (.3502, 991065),
    "C_run": (.3108, 991121), "B_rslora_r16_lr2e4": (.4498, 991343), "B_rslora_r16_lr5e5": (.4445, 991344),
    "KD_seed123": (.4407, 991351), "B_lora_ctx_wide": (.4269, 991339), "B_lora_seed123": (.4147, 991340),
    "B_lora_unweighted": (.4009, 991342), "B_lora_crf": (.3968, 991337), "ENS_top3": (.4458, 991346),
    "ENS_weighted": (.4424, 991353), "ENS_top5": (.4373, 991349), "ENS_top4": (.4360, 991347), "ENS_recall": (.4309, 991345),
    "ENS_candidate": (.4475, 991395), "B_lora_ctx_wide_allep": (.4616, 991973), "KD_allep": (.4496, 991970),
    "KD_seed123_allep": (.4461, 991969), "B_lorma_allep": (.4386, 991972), "H_anchor_w2.0+top4net": (.4659, 992031),
    "G2_anchor+B_lorma_allep+KD_seed123_allep": (.4652, 992028), "C1_one_rep_per_family_bestdev": (.4634, 992032),
}
ISSUES = []


def issue(kind, artifact, detail, severity="warning"):
    ISSUES.append({"severity": severity, "kind": kind, "artifact": artifact, "detail": detail})


def sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ canonical data
labels = rr_data.load_labels(DATA)
dev = {D: rr_data.load_domain(DATA, D, "dev") for D in ("CL", "IT")}
tst = {D: rr_data.load_domain(DATA, D, "test") for D in ("CL", "IT")}
gold = {d["id"]: d["labels"] for D in dev for d in dev[D]}
dom = {d["id"]: D for D in dev for d in dev[D]}
tdom = {d["id"]: D for D in tst for d in tst[D]}
nsent = {k: len(v) for k, v in gold.items()}
tnsent = {d["id"]: len(d["sentences"]) for D in tst for d in tst[D]}
docs, tdocs = list(gold), list(tnsent)


def check_hard(name, split, P):
    ref = nsent if split == "dev" else tnsent
    ok = True
    if set(P) != set(ref):
        issue("doc_ids", name, f"{split}: missing {sorted(set(ref) - set(P))[:3]} extra {sorted(set(P) - set(ref))[:3]}", "error"); ok = False
    else:
        bad = [k for k in ref if len(P[k]) != ref[k]]
        if bad:
            issue("sentence_count", name, f"{split}: {len(bad)} documents with wrong sentence counts", "error"); ok = False
        unk = {l for v in P.values() for l in v} - set(labels)
        if unk:
            issue("label_vocab", name, f"{split}: unknown labels {sorted(unk)}", "error"); ok = False
    return ok


def check_probs(name, split, pr, run_labels):
    ref = nsent if split == "dev" else tnsent
    if pr.get("labels") != labels:
        if sorted(pr.get("labels", [])) == sorted(labels):
            issue("class_order", name, f"{split} probs use a different class order (would need remapping)", "error")
        else:
            issue("class_order", name, f"{split} probs class set differs from label_vocab", "error")
        return False
    if run_labels is not None and run_labels != labels:
        issue("class_order", name, "labels.json differs from label_vocab", "error"); return False
    if set(pr["probs"]) != set(ref) or any(len(pr["probs"][k]) != ref[k] for k in ref):
        issue("prob_alignment", name, f"{split} probs doc ids / sentence counts differ", "error"); return False
    rows = [r for k in pr["probs"] for r in pr["probs"][k]]
    if any(len(r) != len(labels) for r in rows):
        issue("prob_shape", name, f"{split} probs not {len(labels)} columns", "error"); return False
    if not all(math.isfinite(x) and x >= 0 for r in rows for x in r):
        issue("prob_values", name, f"{split} probs contain negative / non-finite values (logits?)", "error"); return False
    dev1 = max(abs(sum(r) - 1) for r in rows)
    if dev1 > 1e-3:
        issue("prob_values", name, f"{split} probs rows do not sum to 1 (max deviation {dev1:.3g}); not softmax", "error"); return False
    return True


# ------------------------------------------------------------------ discover runs
inv, HARD, PROBS = [], {}, {}           # HARD[name] = {"dev": {...}, "test": {...}}; PROBS[name] = {"dev": pr, "test": pr}
for campaign, root in SOURCES.items():
    if not os.path.isdir(root):
        issue("missing_source", campaign, f"{root} not found", "error"); continue
    for n in sorted(os.listdir(root)):
        rd = os.path.join(root, n)
        if not os.path.isdir(rd):
            continue
        f = {k: os.path.join(rd, v) for k, v in dict(config="config.json", summary="summary.json", log="log.jsonl", labels="labels.json",
             dev_pred="dev_predictions.json", test_pred="prediction.json", dev_probs="dev_probs.json", test_probs="test_probs.json",
             ckpt="best_model.pt").items()}
        missing = [k for k, p in f.items() if not os.path.exists(p)]
        cfg = json.load(open(f["config"])) if "config" not in missing else {}
        sm = json.load(open(f["summary"])) if "summary" not in missing else {}
        run_labels = json.load(open(f["labels"])) if "labels" not in missing else None
        HARD[n], PROBS[n] = {}, {}
        status = []
        for split, key in (("dev", "dev_pred"), ("test", "test_pred")):
            if key in missing:
                status.append(f"no {split} predictions"); continue
            P = json.load(open(f[key]))
            if check_hard(n, split, P):
                HARD[n][split] = P
        for split, key in (("dev", "dev_probs"), ("test", "test_probs")):
            if key in missing:
                continue
            pr = json.load(open(f[key]))
            if check_probs(n, split, pr, run_labels):
                PROBS[n][split] = pr
        if "dev" in PROBS[n] and PROBS[n]["dev"]["decode"] == "argmax" and "dev" in HARD[n]:
            am = {k: [labels[max(range(len(labels)), key=r.__getitem__)] for r in PROBS[n]["dev"]["probs"][k]] for k in docs}
            if am != HARD[n]["dev"]:
                issue("probs_vs_predictions", n, "argmax of dev_probs does not reproduce dev_predictions", "error")
        epochs = [json.loads(l) for l in open(f["log"])] if "log" not in missing else []
        ep = [e for e in epochs if e.get("msg") == "epoch"]
        t, tr, ctx, bb = cfg.get("technique", {}), cfg.get("train", {}), cfg.get("context", {}), cfg.get("backbone", {})
        base_name = n[:-6] if n.endswith("_allep") else n
        inv.append({
            "experiment_name": n, "experiment_id": f"{campaign}:{n}", "campaign": campaign,
            "model_family": "", "backbone": os.path.basename(os.path.dirname(cfg.get("phase1_ckpt", ""))),
            "architecture": (sm.get("phase1") or {}).get("module", ""), "training_method": t.get("name", "") + (" + KD" if cfg.get("kd") else ""),
            "seed": tr.get("seed", ""), "learning_rate": tr.get("lr", ""), "weight_decay": tr.get("weight_decay", ""),
            "rank": t.get("r", ""), "alpha": t.get("alpha", ""), "target_modules": " ".join(t.get("targets", [])) if t.get("targets") else "",
            "context_window": f"len {ctx.get('window_len')} L/T/R {ctx.get('left_tokens')}/{ctx.get('target_tokens')}/{ctx.get('right_tokens')}" if ctx else "",
            "training_epochs": tr.get("epochs", ""), "epochs_run": len(ep), "best_epoch": sm.get("best_epoch", ""),
            "early_stopping": f"patience {tr.get('patience')}" if cfg else "",
            "dev_macro_f1": sm.get("dev_macro_f1", ""), "dev_cl_f1": sm.get("dev_CL_macro_f1", ""), "dev_it_f1": sm.get("dev_IT_macro_f1", ""),
            "dev_macro_precision": sm.get("dev_macro_p", ""), "dev_macro_recall": sm.get("dev_macro_r", ""),
            "test_score_official": CODABENCH.get(n, ("",))[0], "codabench_id": CODABENCH.get(n, ("", ""))[1],
            "dev_predictions_path": f["dev_pred"] if "dev" in HARD[n] else "", "test_predictions_path": f["test_pred"] if "test" in HARD[n] else "",
            "dev_probabilities_path": f["dev_probs"] if "dev" in PROBS[n] else "", "test_probabilities_path": f["test_probs"] if "test" in PROBS[n] else "",
            "prediction_type": ("CRF Viterbi labels; probs = pre-decoding emission softmax" if cfg.get("head", {}).get("name") == "mlp_crf"
                                else "argmax labels; probs = softmax of the same logits") if cfg else "",
            "config_path": f["config"] if cfg else "", "checkpoint_path": f["ckpt"] if "ckpt" not in missing else "",
            "ensemble_members": "", "artifact_status": "complete" if not [m for m in missing if m != "ckpt"] else "missing: " + ",".join(missing),
            "notes": ("checkpoint not downloaded (Kaggle output only); " if "ckpt" in missing else "")
                     + ("same config as campaign-1/2 run " + base_name + " except patience None" if n.endswith("_allep") else ""),
        })

# campaign 1: only test zips locally; configs reconstructed from the notebook builder (C1 dict + campaign-1 BASE)
sys.path.insert(0, P3)
import build_notebook_rr as B                     # noqa: E402
g = {"os": os, "json": json, "glob": glob, "DATA_DIR": "data", "TOKENIZER": "tok", "WORK": "/kaggle/working/phase3", "copy": __import__("copy")}
exec(B.CONFIG.split("EXPERIMENTS, RUN = {}, []")[0], g)
C1 = g["C1"]
for n in g["C1_RUN"]:
    z = os.path.join(C1_ZIPS, f"submission_{n}.zip")
    e = g["deep_update"](g["copy"].deepcopy(g["BASE"]), {k: v for k, v in g["copy"].deepcopy(C1[n]).items() if k != "model"})
    e["train"]["patience"] = 2                                                          # campaign-1 setting
    t, tr, ctx = e["technique"], e["train"], e["context"]
    HARD[n], PROBS[n] = {}, {}
    if os.path.exists(z):
        P = json.loads(zipfile.ZipFile(z).read("prediction.json"))
        if check_hard(n, "test", P):
            HARD[n]["test"] = P
    else:
        issue("missing_artifact", n, "campaign-1 submission zip not found", "error")
    issue("missing_artifact", n, "campaign-1 run folder (dev predictions, probabilities, summary, log, checkpoint) not downloaded; only the test zip exists locally", "info")
    inv.append({"experiment_name": n, "experiment_id": f"campaign1:{n}", "campaign": "campaign1", "model_family": "",
                "backbone": {"A": "run_A", "B": "run_B2", "C": "run_C", "D": "run_D"}[C1[n]["model"]], "architecture": "",
                "training_method": t["name"] + (" + KD" if e.get("kd") else ""), "seed": tr["seed"], "learning_rate": tr["lr"],
                "weight_decay": tr["weight_decay"], "rank": t.get("r", ""), "alpha": t.get("alpha", ""),
                "target_modules": " ".join(t.get("targets", [])) if t.get("targets") else "",
                "context_window": f"len {ctx['window_len']} L/T/R {ctx['left_tokens']}/{ctx['target_tokens']}/{ctx['right_tokens']}",
                "training_epochs": tr["epochs"], "epochs_run": "", "best_epoch": "", "early_stopping": "patience 2",
                "dev_macro_f1": "", "dev_cl_f1": "", "dev_it_f1": "", "dev_macro_precision": "", "dev_macro_recall": "",
                "test_score_official": CODABENCH.get(n, ("",))[0], "codabench_id": CODABENCH.get(n, ("", ""))[1],
                "dev_predictions_path": "", "test_predictions_path": z if "test" in HARD[n] else "", "dev_probabilities_path": "",
                "test_probabilities_path": "", "prediction_type": "CRF Viterbi labels" if e["head"]["name"] == "mlp_crf" else "argmax labels",
                "config_path": "reconstructed from build_notebook_rr.py (C1 + BASE, patience 2)", "checkpoint_path": "", "ensemble_members": "",
                "artifact_status": "test predictions only", "notes": "dev predictions / probabilities never downloaded"})

# families (user's definition, by technique type)
for r in inv:
    tm = r["training_method"]
    r["model_family"] = ("F3_KD" if "KD" in tm else "F4_other_backbones" if r["backbone"] not in ("run_B2", "") else
                         "F2_prompt_prefix_ia3" if tm.split()[0] in ("prompt_tuning", "prefix_tuning", "ia3") else "F1_lora_type_adapters")

# ------------------------------------------------------------------ ensembles: recover, reconstruct, verify
ENS = {}


def add_ens(name, method, members, weights, tie, dev_saved, test_saved, rebuild, source, notes="", regenerated_dev=False):
    rec = {"name": name, "method": method, "members": members, "weights": weights, "tie_breaking": tie,
           "class_specific_adjustments": "none" if "routing" not in method else method, "min_votes": "none", "routing": "routing" in method,
           "source": source, "notes": notes, "codabench": CODABENCH.get(name)}
    ok = {}
    for split, saved in (("dev", dev_saved), ("test", test_saved)):
        if saved is None:
            ok[split] = "no saved predictions"; continue
        try:
            re = rebuild(split)
        except KeyError:
            miss = [m for m in members if split not in HARD.get(m, {})]
            ok[split] = f"cannot reconstruct: members without local {split} predictions: {miss}"; continue
        if re is None:
            ok[split] = "cannot reconstruct"; continue
        diff = sum(a != b for k in saved for a, b in zip(saved[k], re[k]))
        ok[split] = "reconstruction matches saved predictions" if diff == 0 else f"MISMATCH on {diff} sentences"
        if diff:
            issue("ensemble_mismatch", name, f"{split}: reconstruction differs on {diff} sentences", "error")
    if regenerated_dev:
        ok["dev"] = "no saved dev predictions; dev regenerated from member predictions (not an independent check)"
    rec["verification"] = ok
    ENS[name] = rec
    HARD[name] = {s: p for s, p in (("dev", dev_saved), ("test", test_saved)) if p is not None}


for e in sorted(os.listdir(C2_ENSEMBLES)) if os.path.isdir(C2_ENSEMBLES) else []:
    d = os.path.join(C2_ENSEMBLES, e); sm = json.load(open(os.path.join(d, "summary.json")))
    mem = [os.path.basename(m.rstrip("/")) for m in sm["members"]]
    add_ens(e, "weighted hard vote (rr_ensemble.vote)", mem, sm["weights"], "earliest listed member",
            json.load(open(os.path.join(d, "dev_predictions.json"))), json.load(open(os.path.join(d, "prediction.json"))),
            lambda s, mem=mem, w=sm["weights"]: vote([HARD[m][s] for m in mem], w), "campaign2 ensembles (Kaggle output)",
            "members are campaign-1 runs: dev reconstruction impossible locally (their dev predictions were not downloaded)")
if os.path.exists(os.path.join(ENS_EXP, "best_candidate.json")):
    bc = json.load(open(os.path.join(ENS_EXP, "best_candidate.json")))
    rules = [tuple(x.split("<-")) for x in bc["config"].split("; ")]
    v7 = ["B_lora_crf", "B_lora_ctx_wide", "B_lora_seed123", "B_lora_unweighted", "B_rslora_r16_lr2e4", "B_rslora_r16_lr5e5", "KD_seed123"]

    def rebuild_cand(s):
        base = vote([HARD[m][s] for m in v7], [1.0] * 7)
        out = {k: list(v) for k, v in base.items()}
        for lab, m in rules:
            for k in out:
                for i, x in enumerate(HARD[m][s][k]):
                    if x == lab:
                        out[k][i] = lab
        return out
    add_ens("ENS_candidate", "7-run hard vote + class routing (label <- model overrides, applied in order)", v7 + [m for _, m in rules],
            [1.0] * 7, "earliest listed member; later routing rule wins", json.load(open(os.path.join(ENS_EXP, "best_candidate_dev_predictions.json"))),
            json.load(open(os.path.join(ENS_EXP, "best_candidate_prediction.json"))), rebuild_cand, "ens_exp_out", "routing rules: " + bc["config"])
fam = json.load(open(os.path.join(ENS_FAM, "summary.json"))) if os.path.exists(os.path.join(ENS_FAM, "summary.json")) else None
if fam:
    for g_, v in fam["best_per_group"].items():
        if not v.get("zip"):
            continue
        name = v["method"]; mtxt = v["members"]
        if "|" in mtxt:
            mem = mtxt.split("|")[0].split(); w = [float(x) for x in mtxt.split("weights")[1].split(",")]
        else:
            mem = mtxt.split(); w = [1.0] * len(mem)
        tpath = os.path.join(ENS_FAM, "candidates", f"{name}_prediction.json")
        if not os.path.exists(tpath):
            issue("missing_artifact", name, "listed in ens_family/out/summary.json but its candidate files were removed "
                  "(probability ensemble did not beat hard voting on dev; dropped per the study's rule)", "info")
            continue
        tp = json.load(open(tpath))
        if name.startswith("C1"):
            def rb(s, mem=mem):        # family vote with representatives, families ordered by best dev (anchor family first)
                return vote([HARD[m][s] for m in mem], [1.0] * len(mem))
        else:
            def rb(s, mem=mem, w=w):
                return vote([HARD[m][s] for m in mem], w)
        add_ens(name, "anchored weighted hard vote" if not name.startswith("C1") else "one representative per family, equal votes",
                mem, w, "earliest listed member (anchor first)", rb("dev"), tp, rb, "ens_family/out",
                "dev predictions regenerated from member predictions (the study saved test predictions only)", regenerated_dev=True)

# ------------------------------------------------------------------ other zips: schema check + duplicates
zip_rows = []
for zdir in [C1_ZIPS] + OTHER_ZIP_DIRS + [os.path.join(ENS_FAM, "candidates"), ENS_EXP]:
    for z in sorted(glob.glob(os.path.join(zdir, "*.zip"))):
        try:
            zf = zipfile.ZipFile(z); names = zf.namelist()
            P = json.loads(zf.read("prediction.json")) if "prediction.json" in names else None
        except Exception as ex:
            zip_rows.append({"zip": z, "phase3_valid": False, "detail": f"unreadable: {ex}", "hash": ""}); continue
        if P is None or not isinstance(P, dict) or set(P) != set(tnsent):
            zip_rows.append({"zip": z, "phase3_valid": False, "detail": "not a Phase-3 prediction (doc ids differ: other phase or malformed)", "hash": ""}); continue
        ok = all(len(P[k]) == tnsent[k] for k in tnsent) and {l for v in P.values() for l in v} <= set(labels)
        zip_rows.append({"zip": z, "phase3_valid": ok, "detail": "ok" if ok else "sentence counts / labels invalid", "hash": sha(P),
                         "root_prediction_json": names == ["prediction.json"]})
by_hash = {}
for r in zip_rows:
    if r["hash"]:
        by_hash.setdefault(r["hash"], []).append(os.path.basename(r["zip"]))
for h_, zs in by_hash.items():
    r = [x for x in zip_rows if x["hash"] == h_]
    for x in r:
        x["identical_to"] = " | ".join(z for z in zs if z != os.path.basename(x["zip"]))
# identical predictions despite different configurations (runs)
run_hash = {n: sha(HARD[n]["test"]) for n in HARD if "test" in HARD[n]}
groups = {}
for n, h_ in run_hash.items():
    groups.setdefault(h_, []).append(n)
for h_, ns in groups.items():
    if len(ns) > 1:
        issue("identical_predictions", " | ".join(ns), "identical test predictions from different experiment names", "info")
for a, b in itertools.combinations([n for n in HARD if "dev" in HARD[n] and n in [r["experiment_name"] for r in inv]], 2):
    if HARD[a]["dev"] == HARD[b]["dev"]:
        issue("identical_predictions", f"{a} | {b}", "identical dev predictions" + (" and test differs on "
              + str(sum(x != y for k in tnsent for x, y in zip(HARD[a]["test"][k], HARD[b]["test"][k]))) + " sentences"
              if "test" in HARD[a] and "test" in HARD[b] else ""), "info")

# ------------------------------------------------------------------ recomputed metrics (separate from originals)
M = os.path.join(OUT, "metrics"); os.makedirs(os.path.join(M, "confusion"), exist_ok=True)
gD = {D: {k: v for k, v in gold.items() if dom[k] == D} for D in ("CL", "IT")}
devnames = [n for n in HARD if "dev" in HARD[n]]
rec_rows, pc_rows, pd_rows = [], [], []
for n in devnames:
    P = HARD[n]["dev"]; s = rr_scores(gold, P, labels)
    cl = rr_scores(gD["CL"], {k: P[k] for k in gD["CL"]}, labels)["macro_f1"]; it = rr_scores(gD["IT"], {k: P[k] for k in gD["IT"]}, labels)["macro_f1"]
    orig = next((r for r in inv if r["experiment_name"] == n), {})
    rec_rows.append({"experiment": n, "recomputed_dev_macro_f1": s["macro_f1"], "recomputed_CL": cl, "recomputed_IT": it,
                     "recomputed_macro_P": s["macro_p"], "recomputed_macro_R": s["macro_r"], "accuracy": s["micro_f1"],
                     "labels_in_macro_average": len(s["per_label"]), "original_dev_macro_f1": orig.get("dev_macro_f1", ""),
                     "matches_original": (orig.get("dev_macro_f1", "") == "" or round(float(orig["dev_macro_f1"]), 4) == round(s["macro_f1"], 4))})
    if orig.get("dev_macro_f1", "") != "" and round(float(orig["dev_macro_f1"]), 4) != round(s["macro_f1"], 4):
        issue("metric_mismatch", n, f"recomputed {s['macro_f1']:.4f} != saved {orig['dev_macro_f1']}", "error")
    for l in labels:
        pl = s["per_label"].get(l)
        pc_rows.append({"experiment": n, "label": l, "precision": pl["p"] if pl else "", "recall": pl["r"] if pl else "",
                        "f1": pl["f1"] if pl else "", "support": pl["support"] if pl else 0, "in_macro_average": bool(pl)})
    for k in docs:
        pd_rows.append({"experiment": n, "document": k, "domain": dom[k], "sentences": nsent[k],
                        "doc_macro_f1": rr_scores({k: gold[k]}, {k: P[k]}, labels)["macro_f1"]})
    with open(os.path.join(M, "confusion", f"{n}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gold \\ predicted"] + labels)
        cm = {(a, b): 0 for a in labels for b in labels}
        for k in docs:
            for a, b in zip(gold[k], P[k]):
                cm[(a, b)] += 1
        for a in labels:
            w.writerow([a] + [cm[(a, b)] for b in labels])
pw = []
for a, b in itertools.combinations(sorted(HARD), 2):
    row = {"run_a": a, "run_b": b}
    for split, ref in (("dev", docs), ("test", tdocs)):
        if split in HARD[a] and split in HARD[b]:
            tot = sum(len(HARD[a][split][k]) for k in ref)
            d = sum(x != y for k in ref for x, y in zip(HARD[a][split][k], HARD[b][split][k]))
            row[f"{split}_disagree"], row[f"{split}_disagree_pct"] = d, round(100 * d / tot, 2)
    pw.append(row)
for fn, rows in (("per_run_summary.csv", rec_rows), ("per_class.csv", pc_rows), ("per_document.csv", pd_rows), ("pairwise_disagreement.csv", pw)):
    with open(os.path.join(M, fn), "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ outputs: inventory, issues, ensembles, submissions, bundle
for name, e in ENS.items():
    inv.append({"experiment_name": name, "experiment_id": f"ensemble:{name}", "campaign": e["source"], "model_family": "ensemble",
                "training_method": e["method"], "ensemble_members": " ".join(e["members"]) + f" | weights {e['weights']}",
                "test_score_official": (e["codabench"] or ("",))[0], "codabench_id": (e["codabench"] or ("", ""))[1],
                "artifact_status": "; ".join(f"{k}: {v}" for k, v in e["verification"].items()), "notes": e["notes"]})
cols = ["experiment_name", "experiment_id", "campaign", "model_family", "backbone", "architecture", "training_method", "seed", "learning_rate",
        "weight_decay", "rank", "alpha", "target_modules", "context_window", "training_epochs", "epochs_run", "best_epoch", "early_stopping",
        "dev_macro_f1", "dev_cl_f1", "dev_it_f1", "dev_macro_precision", "dev_macro_recall", "test_score_official", "codabench_id",
        "dev_predictions_path", "test_predictions_path", "dev_probabilities_path", "test_probabilities_path", "prediction_type", "config_path",
        "checkpoint_path", "ensemble_members", "artifact_status", "notes"]
with open(os.path.join(OUT, "experiment_inventory.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in inv:
        w.writerow({c: r.get(c, "") for c in cols})
with open(os.path.join(OUT, "integrity_issues.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["severity", "kind", "artifact", "detail"]); w.writeheader(); w.writerows(ISSUES)
json.dump(ENS, open(os.path.join(OUT, "ensembles.json"), "w"), indent=1)
with open(os.path.join(OUT, "submissions.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["experiment_or_ensemble", "codabench_score", "codabench_id", "source"])
    for n, (s, i) in sorted(CODABENCH.items(), key=lambda x: -x[1][0]):
        w.writerow([n, s, i, "transcribed from user's Codabench screenshots (codabench_phase3_scores.md)"])
with open(os.path.join(OUT, "zip_audit.csv"), "w", newline="") as f:
    keys = list(dict.fromkeys(k for r in zip_rows for k in r)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(zip_rows)

BD = os.path.join(OUT, "bundle"); os.makedirs(os.path.join(BD, "probs"), exist_ok=True); os.makedirs(os.path.join(BD, "evaluation"), exist_ok=True)
with open(os.path.join(BD, "dev_gold.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["split", "doc_id", "domain", "sentence_index", "gold_label", "sentence_text"])
    for D in dev:
        for d in dev[D]:
            for i, (s, l) in enumerate(zip(d["sentences"], d["labels"])):
                w.writerow(["dev", d["id"], D, i, l, s])
with open(os.path.join(BD, "test_meta.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["split", "doc_id", "domain", "sentence_index", "sentence_text"])     # no test labels exist locally
    for D in tst:
        for d in tst[D]:
            for i, s in enumerate(d["sentences"]):
                w.writerow(["test", d["id"], D, i, s])
json.dump({"labels": labels, "index": {l: i for i, l in enumerate(labels)}, "source": "data/Phase_3_RR/Train/label_vocab.json"},
          open(os.path.join(BD, "label_vocab.json"), "w"), indent=1)
for split, ref in (("dev", docs), ("test", tdocs)):
    names = sorted(n for n in HARD if split in HARD[n])
    with open(os.path.join(BD, f"hard_predictions_{split}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "doc_id", "sentence_index"] + names)
        for k in ref:
            for i in range(len(HARD[names[0]][split][k])):
                w.writerow([split, k, i] + [HARD[n][split][k][i] for n in names])
for n, pr in PROBS.items():
    for split, ref in (("dev", docs), ("test", tdocs)):
        if split in pr:
            with open(os.path.join(BD, "probs", f"{n}_{split}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["split", "doc_id", "sentence_index", "decode"] + [f"p_{l}" for l in labels])
                for k in ref:
                    for i, r in enumerate(pr[split]["probs"][k]):
                        w.writerow([split, k, i, pr[split]["decode"]] + [f"{x:.6g}" for x in r])
shutil.copy(os.path.join(P3, "rr_metrics.py"), os.path.join(BD, "evaluation", "rr_metrics.py"))
shutil.copy(os.path.join(P3, "rr_ensemble.py"), os.path.join(BD, "evaluation", "rr_ensemble.py"))

# summary for the report
n_runs = len([r for r in inv if not r["experiment_id"].startswith("ensemble:")])
summ = {"experiments": n_runs, "ensembles": len(ENS),
        "dev_predictions": len([n for n in HARD if "dev" in HARD[n] and not n in ENS]),
        "test_predictions": len([n for n in HARD if "test" in HARD[n] and not n in ENS]),
        "with_probabilities": len([n for n in PROBS if "dev" in PROBS[n]]),
        "hard_only": len([n for n in HARD if n not in ENS and not PROBS.get(n, {}).get("dev")]),
        "issues": {s: sum(i["severity"] == s for i in ISSUES) for s in ("error", "warning", "info")},
        "zips": {"phase3_valid": sum(bool(r.get("phase3_valid")) for r in zip_rows), "other": sum(not r.get("phase3_valid") for r in zip_rows)},
        "ensemble_verification": {k: v["verification"] for k, v in ENS.items()}}
json.dump(summ, open(os.path.join(OUT, "audit_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
print("errors:", [i for i in ISSUES if i["severity"] == "error"][:10])
