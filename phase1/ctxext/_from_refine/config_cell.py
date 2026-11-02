# configuration
SEED = 42

# ---------------- session budget (plumbing) ----------------
SESSION_HOURS = 12.0        # Kaggle hard limit
SAFETY_MIN = 20             # stop everything this long before the limit
GEN_RESERVE_MIN = 10        # training-only run: final eval + checkpoint save. Set 45 if DECODING is not empty

# ---------------- inputs ----------------
OUTPUT_DIR = "/kaggle/working/cs781_phase1"
B_TOKENIZER_SHA256 = "38f51f147c54fa95b86f6d14fa521b6edc7cbe2655b66bf1325c5f13e60dd030"  # legal_bpe.json of the A/B run
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
TEST_FILE = "Phase_1_Pretraining/Test/test.json"
VOCAB_SIZE = 32_000


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================================
# YOUR CHOICES (course policy: model design + hyperparameters are yours). Fill every None.
# =====================================================================================
TAG = "B2"                           # name of this run: run_<TAG>/, submission_<TAG>_<strategy>.zip
INIT_FROM = "auto"                   # None = train from scratch | "auto" = warm start (weights only, fresh optimiser and
                                     # schedule) from an attached run_B/best.pt | or an explicit path to a best.pt
RESUME = None                        # path to a last.pt of THIS run (continue an interrupted run, same schedule)

MODEL = {                            # your B architecture as trained before (arch3 = GQA + RoPE); change it if you want
    "module": "arch3",
    "model": dict(vocab_size=VOCAB_SIZE, d_model=768, n_layers=18, n_heads=12, n_kv_heads=4, d_ff=3456,
                  max_seq_len=896, dropout=0.1, pos_emb_type="rope", norm_first=True, attention_scale=0.125),
    "init_std": 0.02,
}
TRAIN = dict(
    batch_tokens=16384,              # GLOBAL batch in tokens, split over the GPUs: batch_tokens // max_seq_len sequences
                                     # must be divisible by 2 (B's run: 8192 -> 9 seqs, one T4, micro-batch 9 fitted)
    lr=3e-4, min_lr=3e-5, warmup_steps=1000, schedule="cosine",   # "cosine" | "linear" | "constant"
    weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, grad_clip=1.0,  # B's previous values
    max_tokens=None,                 # optional token cap (global tokens); None = the schedule runs to the time deadline
    # plumbing
    eval_every_min=30, eval_windows=360, log_every=50, compile=True, fast_attention=True, seed=SEED,
)
GEN = dict(                          # settings shared by every decoding strategy
    checkpoint="best",               # "best" (lowest val loss) or "last"
    target_words=128, max_new_tokens=256, batch_size=32,
    n_val_prompts=100,               # held-out SC prompts -> val_generations_<name>.json for score_local.py
    seed=SEED,
)
DECODING = {}                        # {} = TRAINING ONLY (generate locally from best.pt with genlocal/gen_local.py).
                                     # To generate on Kaggle instead, add strategies and set GEN_RESERVE_MIN = 45, e.g.
                                     # "greedy": dict(temperature=0.0, top_k=None, top_p=1.0, repetition_penalty=1.0),
                                     # optional keys: no_repeat_ngram (n > 1, token n-grams), ngram_scope ("output" | "all")
if DECODING and GEN_RESERVE_MIN < 30:
    print(f"WARNING: {len(DECODING)} decoding strategies but GEN_RESERVE_MIN = {GEN_RESERVE_MIN}: generation may be skipped")
print("Mode:", "train + generate on Kaggle" if DECODING else "TRAINING ONLY (no generation on Kaggle)")

# ---------------- data folder (plumbing) ----------------
_cands = sorted({os.path.dirname(h) for h in glob.glob("/kaggle/input/**/legal_bpe.json", recursive=True)
                 if all(os.path.exists(os.path.join(os.path.dirname(h), f)) for f in ("train.bin", "val.bin"))})
assert _cands, "no data attached: add an input with legal_bpe.json + train.bin + val.bin (prebuilt data or the A/B output)"
_hash = {d: _sha256(os.path.join(d, "legal_bpe.json")) for d in _cands}
for d in _cands:
    print(f"data candidate: {d}  tokenizer sha256 {_hash[d][:12]}{'  (= B tokenizer)' if _hash[d] == B_TOKENIZER_SHA256 else ''}")
_match = [d for d in _cands if _hash[d] == B_TOKENIZER_SHA256]
if INIT_FROM:
    assert _match, "INIT_FROM needs B's tokenizer, but no attached data folder has it: attach the A/B run's output"
DATA_DIR = (_match or _cands)[0]
TOKENIZER_PATH = os.path.join(DATA_DIR, "legal_bpe.json")
print("Data dir:", DATA_DIR)

if INIT_FROM == "auto":
    _ck = sorted(glob.glob("/kaggle/input/**/run_B/best.pt", recursive=True))
    assert _ck, "INIT_FROM = 'auto' but no run_B/best.pt is attached (add the A/B run's output as an input)"
    INIT_FROM = _ck[0]
if INIT_FROM:
    assert os.path.exists(INIT_FROM), INIT_FROM
    print("Warm start from:", INIT_FROM)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def missing(d, prefix=""):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += missing(v, f"{prefix}{k}.")
        elif v is None and k not in ("attention_scale", "init_std", "max_tokens", "top_k"):
            out.append(prefix + k)
    return out

todo = missing(MODEL, "MODEL.") + missing(TRAIN, "TRAIN.")
assert not todo, "Fill these in the configuration cell before running: " + ", ".join(todo)
print(f"Configuration loaded. Run {TAG}: {MODEL['module']}, {N_GPUS} GPU(s)")