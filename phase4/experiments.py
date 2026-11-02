"""CJPE experiment matrix: YOUR values (course rule 9), transcribed from your Phase-4 specification (2026-10-19).

build(phase1, data_dir, tokenizer, out_dir, cache_dir, steps_per_epoch) -> {run name: config dict}
Field meanings: docstring of train_cjpe.py. Values marked "(Phase 3)" were not in the Phase-4 specification and are
carried over unchanged from your Phase-3 BASE / technique options; change them here if you want something else.
"""
import copy

BASE = {
    "fast_attention": True,
    "backbone": {"prefix": None, "encdec_mode": None, "truncate_layers": None,
                 "dropout": 0.0},                                    # (Phase 3)
    "chunking": {"strategy": "even", "max_chunks": 4, "chunk_len": None, "stride": None},   # chunk_len None = model limit
    "chunk_pool": "mean",                                            # (Phase 3 pooling: mean over the chunk's tokens)
    "doc_pool": "attn",
    "head": {"name": "mlp", "hidden": 256, "dropout": 0.1},          # (Phase 3)
    "technique": {"name": "full_ft"},
    "kd": None,
    "select_on": "dev_all",
    "train": {
        "epochs": 1,                    # >= 1 full epoch per main experiment; a 2nd epoch only for the strongest later
        "batch_docs": 8,                # effective batch: 8 documents per optimizer step (your range 4-8)
        "micro_docs": 2,                # documents per forward pass (memory; checked by the Kaggle smoke stage)
        "lr": 5e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_frac": 0.06,            # "short warm-up" (Phase 3 value)
        "schedule": "cosine",
        "grad_clip": 1.0,
        "class_weights": None,
        "seed": 42,                     # (Phase 3)
        "log_every": 50,
        "infer_micro_docs": 8,
        "max_minutes": None,            # no mid-epoch cut: every main run completes its epoch
    },
}

LORMA = {"name": "lorma", "r": 8, "alpha": 16, "targets": ["attn", "ffn"]}           # alpha / targets (Phase 3)
ADALORA_PHASE3 = {"alpha": 16, "targets": ["attn", "ffn"], "t_final": 100, "delta_t": 10,
                  "beta1": 0.85, "beta2": 0.85, "orth_reg": 0.01}                     # (Phase 3)


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def matrix(steps_per_epoch):
    """-> {name: (model key, overrides)}; model keys: A, B2, B3, C, D."""
    return {
        # E1: primary reference (also C2, the "even" condition of the chunking ablation)
        "E1_B2_full": ("B2", {}),
        "E2_B3_full": ("B3", {}),                                      # chunk_len = 1024 (B3's limit)
        "E3_A_partial": ("A", {"technique": {"name": "partial_ft", "k": 8}}),   # A has 17 blocks: top 8 train, lower 9 frozen
        "E4_C_adapters": ("C", {"backbone": {"encdec_mode": "encoder"},
                                "technique": {"name": "adapters", "bottleneck": 64},
                                "train": {"lr": 1e-4}}),
        "E5_D_bitfit": ("D", {"backbone": {"prefix": 640}, "technique": {"name": "bitfit"}, "train": {"lr": 1e-4}}),
        "E6_B2_lorma": ("B2", {"technique": dict(LORMA), "train": {"lr": 2e-4}}),
        "E7_B2_adalora": ("B2", {"technique": {"name": "adalora", "r": 16, "target_rank": 4,
                                               "t_init": max(1, int(0.10 * steps_per_epoch)), **ADALORA_PHASE3},
                                 "train": {"lr": 1e-4}}),
        # chunking ablations on B2 (C2 = E1)
        "C1_B2_headtail": ("B2", {"chunking": {"strategy": "headtail"}}),
        "C3_B2_overlap": ("B2", {"chunking": {"strategy": "overlap", "stride": 640}}),
        # class weighting on B2 (if compute permits)
        "W1_B2_full_weighted": ("B2", {"train": {"class_weights": [0.847, 1.220]}}),
    }


# E8 (KD) is built after the main runs: the teacher is the finished run with the highest dev macro-F1.
KD_STUDENT = ("B2", {"backbone": {"truncate_layers": 12}, "technique": {"name": "full_ft"},
                     "kd": {"temperature": 2.0, "alpha": 0.5}, "train": {"lr": 5e-5, "head_lr": 1e-4}})


def make_cfg(name, model_key, overrides, phase1, data_dir, tokenizer, out_dir, cache_dir):
    c = deep_update(copy.deepcopy(BASE), overrides)
    c.update(name=name, phase1_ckpt=phase1[model_key], data_dir=data_dir, tokenizer=tokenizer, out_dir=out_dir,
             cache_dir=cache_dir, model_key=model_key)
    assert c["phase1_ckpt"], f"{name}: no checkpoint for model {model_key}"
    return c


def build(phase1, data_dir, tokenizer, out_dir, cache_dir, steps_per_epoch):
    return {n: make_cfg(n, m, o, phase1, data_dir, tokenizer, out_dir, cache_dir) for n, (m, o) in matrix(steps_per_epoch).items()}


def build_kd(teacher_run, phase1, data_dir, tokenizer, out_dir, cache_dir, name="E8_KD_B2s12"):
    m, o = copy.deepcopy(KD_STUDENT)
    o["kd"]["teacher_run"] = teacher_run
    return make_cfg(name, m, o, phase1, data_dir, tokenizer, out_dir, cache_dir)
