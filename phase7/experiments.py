"""PCR experiment matrix: YOUR values (course rule 9), transcribed from your Phase-7 config (2026-11-06).
BASE = your COMMON block; pcr_config / matrix / SPLIT / KD are yours unchanged.

build(phase1, data_dir, tokenizer, out_dir, cache_dir) -> {run name: config dict}
Spec 13.2: >= 6 substantially different experiments, >= 5 different techniques from {full_ft, partial_ft, qlora,
lorma (compulsory), prefix_tuning, prompt_tuning, ia3}, + 1 KD run (counts as one of the 6).
Field meanings: train_pcr.py docstring.
"""
import copy

BASE = {
    "fast_attention": True,
    "backbone": {"prefix": None, "encdec_mode": None, "truncate_layers": None, "dropout": None},
    "chunking": {"strategy": "even", "max_chunks": 6, "chunk_len": 896, "stride": 640},
    "chunk_pool": "mean",
    "doc_pool": "attn",
    "proj_dim": 256,
    "similarity": "cosine",
    "temperature": 0.05,
    "negatives": {"source": "bm25", "per_positive": 2, "bm25_top": 100, "in_batch": True},
    "technique": {"name": None},
    "kd": None,
    "train": {},
}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def pcr_config(technique, lr, epochs=2, head_lr=3e-4, changes=None, tech_opts=None):
    c = {
        "technique": {"name": technique, **(copy.deepcopy(tech_opts) if tech_opts else {})},
        "train": {
            "epochs": epochs, "batch_pairs": 8, "micro_docs": 2, "lr": lr, "head_lr": head_lr,
            "weight_decay": 0.01, "warmup_frac": 0.05, "schedule": "cosine", "grad_clip": 1.0, "seed": 42,
            "log_every": 20, "infer_micro_docs": 4, "max_minutes": 250, "eval_every_steps": None, "patience": 1,
            "eval_at_start": True,
        },
    }
    if changes:
        deep_update(c, changes)
    return c


LORMA_OPTS = {"r": 16, "alpha": 32, "targets": ["attn", "ffn"]}
QLORA_OPTS = {"r": 16, "alpha": 32, "targets": ["attn", "ffn"], "block_size": 64}


def matrix():
    """-> {name: (model key, overrides)}; model keys: A, B2, B3, C, D."""
    return {
        # P1: Full fine-tuning
        "P1": ("B2", pcr_config("full_ft", lr=1e-5, epochs=2)),
        # P2: Partial fine-tuning
        "P2": ("B2", pcr_config("partial_ft", lr=2e-5, epochs=2, tech_opts={"k": 6})),
        # P3: QLoRA
        "P3": ("B2", pcr_config("qlora", lr=1e-4, epochs=2, tech_opts=QLORA_OPTS)),
        # P4: LoRMA - KD teacher
        "P4": ("B2", pcr_config("lorma", lr=5e-5, epochs=2, tech_opts=LORMA_OPTS)),
        # P5: Prefix tuning
        "P5": ("B2", pcr_config("prefix_tuning", lr=2e-4, epochs=2, tech_opts={"n_prefix": 32})),
        # P6: LoRMA + mean document pooling
        "P6": ("B2", pcr_config("lorma", lr=5e-5, epochs=2, tech_opts=LORMA_OPTS, changes={"doc_pool": "mean"})),
        # P7: LoRMA + random negatives
        "P7": ("B2", pcr_config("lorma", lr=5e-5, epochs=2, tech_opts=LORMA_OPTS,
                                changes={"negatives": {"source": "random", "per_positive": 2, "bm25_top": None,
                                                       "in_batch": True}})),
    }


SPLIT = {
    1: ["P1", "P3", "P5", "P7"],
    2: ["P2", "P4", "P6", "P8_kd"],
}

KD_NAME = "P8_kd"
KD_TEACHER = "P4"          # P4 is queued before P8_kd on account 2
KD_STUDENT = ("B2", pcr_config("full_ft", lr=2e-5, epochs=2,
                               changes={"backbone": {"truncate_layers": 9}, "kd": {"temperature": 2.0, "alpha": 0.5}}))


def make_cfg(name, model_key, overrides, phase1, data_dir, tokenizer, out_dir, cache_dir):
    assert model_key, f"{name}: set its model key (A / B2 / B3 / C / D) in experiments.py"
    c = deep_update(copy.deepcopy(BASE), overrides)
    c.update(name=name, phase1_ckpt=phase1[model_key], data_dir=data_dir, tokenizer=tokenizer, out_dir=out_dir,
             cache_dir=cache_dir, model_key=model_key)
    assert c["phase1_ckpt"], f"{name}: no checkpoint for model {model_key} (attach it)"
    return c


def build(phase1, data_dir, tokenizer, out_dir, cache_dir):
    return {n: make_cfg(n, m, o, phase1, data_dir, tokenizer, out_dir, cache_dir) for n, (m, o) in matrix().items()}


def build_kd(teacher_run, phase1, data_dir, tokenizer, out_dir, cache_dir, name=KD_NAME):
    m, o = copy.deepcopy(KD_STUDENT)
    o["kd"]["teacher_run"] = teacher_run
    return make_cfg(name, m, o, phase1, data_dir, tokenizer, out_dir, cache_dir)
