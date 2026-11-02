"""LSI experiment matrix: YOUR values (course rule 9), transcribed from your Phase-5 config (2026-10-24).

build(phase1, data_dir, tokenizer, out_dir, cache_dir, steps_per_epoch) -> {run name: config dict}
SPLIT = which runs each Kaggle account trains (ACCOUNT = 1 / 2); KD (L7) runs on account 1 after its teacher L1.
Transcription notes: epochs = 2 for every run (your BASE; the per-run "epochs: 5" of the first draft were dropped after
the time estimate — change per run if wanted); head "type": "mlp" -> heads.py name "mlp".
"""
import copy

BASE = {
    "fast_attention": True,
    "backbone": {"prefix": None, "encdec_mode": None, "truncate_layers": None, "dropout": 0.0},
    "chunking": {"strategy": "overlap", "max_chunks": 8, "chunk_len": None, "stride": 640},
    "chunk_pool": "mean",
    "doc_pool": "attn",
    "head": {"name": "mlp", "hidden": 256, "dropout": 0.1},
    "technique": {"name": "full_ft"},
    "kd": None,
    "select_on": "dev_excl_test",
    "save_best_on": ["dev_excl_test", "dev_all"],
    "train": {
        "epochs": 2, "batch_docs": 8, "micro_docs": 2, "lr": 2e-5, "head_lr": 1e-4,
        "weight_decay": 0.01, "warmup_frac": 0.06, "schedule": "cosine", "grad_clip": 1.0,
        "pos_weight": "sqrt_inv_freq", "pos_weight_max": 5.0,
        "seed": 42, "log_every": 50, "infer_micro_docs": 8, "max_minutes": None, "eval_every_steps": None,
    },
}


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
        "L1_B2_full": ("B2", {"technique": {"name": "full_ft"}, "train": {"lr": 2e-5, "head_lr": 1e-4}}),
        "L2_A_partial": ("A", {"technique": {"name": "partial_ft", "k": 6}, "train": {"lr": 3e-5, "head_lr": 1e-4}}),
        "L3_B3_qlora": ("B3", {"technique": {"name": "qlora", "r": 16, "alpha": 32, "targets": ["attn", "ffn"], "block_size": 64},
                               "train": {"lr": 1e-4, "head_lr": 1e-4}}),
        "L4_B2_lorma": ("B2", {"technique": {"name": "lorma", "r": 16, "alpha": 32, "targets": ["attn", "ffn"]},
                               "train": {"lr": 5e-5, "head_lr": 1e-4}}),
        "L5_C_adapters": ("C", {"backbone": {"encdec_mode": "encoder"},
                                "chunking": {"strategy": "overlap", "max_chunks": 8, "chunk_len": 640, "stride": 448},
                                "technique": {"name": "adapters", "bottleneck": 64}, "train": {"lr": 5e-5, "head_lr": 1e-4}}),
        "L6_D_prefix": ("D", {"backbone": {"prefix": 640}, "technique": {"name": "prefix_tuning", "n_prefix": 32},
                              "train": {"lr": 5e-4, "head_lr": 1e-4}}),
        "L8_B2_adalora": ("B2", {"technique": {"name": "adalora", "r": 16, "alpha": 32, "targets": ["attn", "ffn"], "target_rank": 8,
                                               "t_init": 100, "t_final": 1000, "delta_t": 10, "beta1": 0.85, "beta2": 0.85,
                                               "orth_reg": 0.1},
                                 "train": {"lr": 5e-5, "head_lr": 1e-4}}),
    }


# Two accounts balanced with the measured LSI token counts x Phase-4 speeds (about 10.3 h each). L7 (KD) is in account
# 1's queue: it waits for L1 (its teacher) and starts on the first free GPU after L1 finishes.
SPLIT = {1: ["L1_B2_full", "L4_B2_lorma", "L7_B2_kd", "L6_D_prefix"],
         2: ["L3_B3_qlora", "L8_B2_adalora", "L5_C_adapters", "L2_A_partial"]}

# L7 — B2 teacher (finished run L1_B2_full) -> 12-block B2 student
KD_NAME = "L7_B2_kd"
KD_TEACHER = "L1_B2_full"
KD_STUDENT = ("B2", {"backbone": {"truncate_layers": 12}, "technique": {"name": "full_ft"},
                     "kd": {"temperature": 2.0, "alpha": 0.5}, "train": {"lr": 2e-5, "head_lr": 1e-4}})


def make_cfg(name, model_key, overrides, phase1, data_dir, tokenizer, out_dir, cache_dir):
    c = deep_update(copy.deepcopy(BASE), overrides)
    c.update(name=name, phase1_ckpt=phase1[model_key], data_dir=data_dir, tokenizer=tokenizer, out_dir=out_dir,
             cache_dir=cache_dir, model_key=model_key)
    assert c["phase1_ckpt"], f"{name}: no checkpoint for model {model_key}"
    return c


def build(phase1, data_dir, tokenizer, out_dir, cache_dir, steps_per_epoch):
    return {n: make_cfg(n, m, o, phase1, data_dir, tokenizer, out_dir, cache_dir) for n, (m, o) in matrix(steps_per_epoch).items()}


def build_kd(teacher_run, phase1, data_dir, tokenizer, out_dir, cache_dir, name=KD_NAME):
    m, o = copy.deepcopy(KD_STUDENT)
    o["kd"]["teacher_run"] = teacher_run
    return make_cfg(name, m, o, phase1, data_dir, tokenizer, out_dir, cache_dir)
