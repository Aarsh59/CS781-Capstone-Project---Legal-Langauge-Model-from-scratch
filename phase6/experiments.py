"""Phase-6 experiment matrix: YOUR values (course rule 9), as written by you (2026-11-02); epochs per run updated by you the same day.
max_minutes per run = safety caps added by Claude for the 12 h session scheduler (not a training choice): ~1.4 x the
estimated run time; edit freely.
Field meanings: train_summ.py / summ_infer.py docstrings."""
import copy

# ============================================================
# SHARED BASE CONFIGURATION
# ============================================================

BASE = {
    "fast_attention": True,

    "backbone": {
        "dropout": 0.0,
        "max_seq_len_override": None,
    },

    "technique": {
        "name": "full_ft",
    },

    "format": {
        "src_len": 640,
        "tgt_len": 256,
        "eos_on_truncated": True,
        "target_trunc": "sentence",
        "sep_token": "</s>",
        "start_token": None,
    },

    "selection": {
        "strategy": "headtail",
        "head_frac": 0.5,
    },

    "source": {
        "type": "document",
    },

    "split": {
        "val_docs": 500,
        "seed": 42,
    },

    "train": {
        "epochs": 5,
        "batch_examples": 8,
        "micro_examples": 1,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_frac": 0.06,
        "schedule": "cosine",
        "grad_clip": 1.0,
        "seed": 42,
        "eval_every_steps": 250,
        "patience": 3,
        "max_minutes": 600,
        "log_every": 50,
    },

    "decode": {
        "length": {
            "mode": "ratio",
            "ratio": 0.25,
            "basis": "document",
            "min_words": 20,
            "max_words": 180,
            "allow_eos": True,
        },
        "temperature": 0.3,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram": 3,
        "batch_size": 16,
    },

    "eval_generate": {
        "val_docs": 100,
        "bert_model": "roberta-large",
    },

    "kd": None,
    "after": None,
}


# ============================================================
# TECHNIQUE TARGETS
# ============================================================

ATTENTION_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]

ATTENTION_FFN_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "ffn.net.0",
    "ffn.net.2",
]


# ============================================================
# EXPERIMENT MATRIX
# ============================================================

MATRIX = {

    # R1: B2 + LoRMA
    "R1_B2_LoRMA": (
        "B2",
        {
            "technique": {
                "name": "lorma",
                "r": 8,
                "alpha": 16,
                "targets": ATTENTION_FFN_TARGETS,
            },
            "train": {
                "lr": 2e-4,
                "epochs": 10,
                "max_minutes": 160,   # safety cap set by Claude (scheduling only; edit freely)
            },
        },
    ),

    # R2: A + LoRA
    "R2_A_LoRA": (
        "A",
        {
            "technique": {
                "name": "lora",
                "r": 8,
                "alpha": 16,
                "targets": ATTENTION_TARGETS,
            },
            "train": {
                "lr": 1e-4,
                "epochs": 10,
                "max_minutes": 135,   # safety cap set by Claude (scheduling only; edit freely)
            },
        },
    ),

    # R3: B2 + QLoRA
    "R3_B2_QLoRA": (
        "B2",
        {
            "technique": {
                "name": "qlora",
                "r": 8,
                "alpha": 16,
                "targets": ATTENTION_TARGETS,
                "block_size": 64,
            },
            "train": {
                "lr": 1e-4,
                "epochs": 10,
                "max_minutes": 270,   # safety cap set by Claude (scheduling only; edit freely)
            },
        },
    ),

    # R4: C + Houlsby Adapters
    "R4_C_Houlsby": (
        "C",
        {
            "format": {
                "src_len": 384,
                "tgt_len": 256,
                "eos_on_truncated": True,
                "target_trunc": "sentence",
                "sep_token": None,
                "start_token": "last_src",
            },
            "technique": {
                "name": "adapters",
                "bottleneck": 64,
            },
            "train": {
                "lr": 1e-4,
                "epochs": 12,
                "max_minutes": 100,   # safety cap set by Claude (scheduling only; edit freely)
            },
        },
    ),

    # R5: D + Prefix Tuning
    "R5_D_Prefix": (
        "D",
        {
            "format": {
                "src_len": 624,
                "tgt_len": 256,
                "eos_on_truncated": True,
                "target_trunc": "sentence",
                "sep_token": "</s>",
                "start_token": None,
            },
            "technique": {
                "name": "prefix_tuning",
                "n_prefix": 16,
            },
            "train": {
                "lr": 5e-4,
                "epochs": 10,
                "max_minutes": 190,   # safety cap set by Claude (scheduling only; edit freely)
            },
        },
    ),

    # R6: B2 + FULL FINE-TUNING
    "R6_B2_FullFT": (
        "B2",
        {
            "technique": {
                "name": "full_ft",
            },
            "train": {
                "lr": 2e-5,
                "epochs": 8,
                "max_minutes": 130,   # safety cap set by Claude (scheduling only; edit freely)
                "batch_examples": 8,
                "micro_examples": 1,
                "weight_decay": 0.01,
                "warmup_frac": 0.06,
                "schedule": "cosine",
            },
        },
    ),

    # R7: B2 KNOWLEDGE DISTILLATION  (teacher: R1_B2_LoRMA, student: B2 with 12 layers)
    "R7_B2_KD": (
        "B2",
        {
            "backbone": {
                "truncate_layers": 12,
            },
            "technique": {
                "name": "full_ft",
            },
            "kd_teacher": "R1_B2_LoRMA",
            "kd": {
                "temperature": 2.0,
                "alpha": 0.5,
            },
            "train": {
                "lr": 2e-5,
                "epochs": 10,
                "max_minutes": 200,   # safety cap set by Claude (scheduling only; edit freely)
                "batch_examples": 8,
                "micro_examples": 1,
                "weight_decay": 0.01,
                "warmup_frac": 0.06,
                "schedule": "cosine",
            },
        },
    ),
}


# ============================================================
# KAGGLE ACCOUNT ALLOCATION
# ============================================================

SPLIT = {
    1: [
        "R1_B2_LoRMA",
        "R3_B2_QLoRA",
        "R7_B2_KD",
    ],
    2: [
        "R2_A_LoRA",
        "R4_C_Houlsby",
        "R5_D_Prefix",
        "R6_B2_FullFT",
    ],
}


# ============================================================
# CONFIG BUILDER
# ============================================================

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict) and k not in (
            "technique", "kd", "after"
        ):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def build(phase1, data_dir, tokenizer, out_dir, cache_dir):
    out = {}
    for name, (model_key, over) in MATRIX.items():
        c = deep_update(copy.deepcopy(BASE), over)
        assert phase1.get(model_key), (
            f"{name}: Phase-1 model {model_key} not found"
        )
        c.update(
            name=name,
            model_key=model_key,
            phase1_ckpt=phase1[model_key],
            data_dir=data_dir,
            tokenizer=tokenizer,
            out_dir=out_dir,
            cache_dir=cache_dir,
        )
        out[name] = c
    return out
