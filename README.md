# Legal Language Models from Scratch (CS781 Capstone, IIT Kanpur)

An in-house legal language-model system built for the CS781 *Legal Language Modelling
Challenge*, a multi-phase Codabench competition. Every model is pretrained from scratch on
Indian legal text (no pretrained weights or external corpora allowed) and then adapted to a
series of downstream legal NLP tasks with full fine-tuning, parameter-efficient methods and
knowledge distillation. All training ran on free Kaggle GPUs (2× T4).

## Phase 1: Pretraining

A custom byte-level BPE tokenizer (32k vocabulary, written from scratch) was trained on the
legal corpus, and four architecturally different models were pretrained on a Supreme-Court-heavy
data mix built with DSIR-style importance selection over a 3.8B-token cleaned corpus.

| Model | Architecture | Params |
|---|---|---|
| A | Decoder-only, multi-head attention | 151.7M |
| B / B2 | Decoder-only, grouped-query attention (12 → 4 KV heads) + RoPE; B2 = B trained further to ~1.3B tokens | 148.6M |
| C | Encoder–decoder (10 + 10 layers, GQA, RoPE) | 202.7M |
| D | Prefix-LM (bidirectional prefix, causal continuation) | 162.3M |

B2 is the main backbone for every later phase. On the test task (continue a 512-word legal
prompt with 128 words), minimum Bayes risk (MBR) decoding over 8 samples with n-gram blocking
reached **0.842 BERTScore F1**.

## Downstream phases

| Phase | Task | Metric | Best result |
|---|---|---|---|
| 2 | Legal named-entity recognition | strict span macro-F1 | 0.531 (vote ensemble + span snapping) |
| 3 | Rhetorical-role labelling (13 classes) | macro-F1 | 0.466 (anchored hard-vote ensemble) |
| 4 | Court judgment prediction (long documents) | macro-F1 | 0.808 leaderboard (ENS05; threshold tuned on test-overlapping dev) / 0.801 (held-out-selected re-chunked ensemble) |
| 5 | Statute identification (100-label multi-label) | macro-F1 | 0.380 (B2 full FT) |
| 6 | Abstractive judgment summarization | BERTScore F1 | 0.839 (B2 full FT) |
| 7 | Prior case retrieval | Micro-F1@7 | 0.333 (B2 LoRMA bi-encoder, random negatives) |

Each phase compares at least five adaptation techniques across the Phase-1 models, chosen from:
full and partial fine-tuning, BitFit, LoRA, rsLoRA, QLoRA (4-bit NF4), AdaLoRA, LoRMA, IA³,
bottleneck adapters, prefix tuning, prompt tuning, and knowledge distillation into smaller
students. Long documents are handled with chunked encoders (overlapping windows + attention
pooling) for classification, and head+tail source selection with KV-cached generation for
summarization.

Detailed per-phase documentation (problem, data, method, every experiment and its score, how to run) is in
[`PHASES.md`](PHASES.md); additional notes for Phases 1–6 are in [`CAPSTONE_SUMMARY_P1_P5.md`](CAPSTONE_SUMMARY_P1_P5.md).

## Repository layout

| Path | Contents |
|---|---|
| `phase1/models/` | Model code: attention variants, RoPE, decoder, encoder–decoder, prefix-LM, fast attention with KV cache |
| `phase1/notebooks/` | Pretraining notebooks (A/B, C/D, B2 refinement) |
| `phase1/ctxext/`, `capstone_b2_ctx*.ipynb` | B3: context extension of B2 from 896 to 1024 tokens |
| `phase1/genlocal/`, `phase1/tools/` | Decoding experiments (sampling, beam, MBR) and local scoring |
| `phaseN/` (2–7) | Per-phase training, inference and ensembling scripts |
| `phaseN/techniques/` | Fine-tuning / PEFT / distillation implementations for that phase |
| `phaseN/phase1_models/` | Copy of the Phase-1 model code so each phase runs on its own |
| `capstone_phaseN_*.ipynb` | Kaggle notebooks that run each phase end to end |

## Reproducing

The notebooks are written for Kaggle (2× T4) and expect the competition data and the Phase-1
checkpoints as attached Kaggle datasets. Data, checkpoints and submission files are not included
in this repository.
