# CS781 Capstone: per-phase documentation

> File lists in each phase section describe that phase's folder. In this repository the Phase-1 code is split across `phase1/models/`, `phase1/notebooks/`, `phase1/genlocal/` and `phase1/tools/`; phases 2–7 match the folder names directly.

One section per phase: problem, data, method, experiments and results, and how to run it.

All later phases start from the Phase-1 models trained from scratch (A, B/B2, C, D) and use the same tokenizer: an own byte-level BPE, 32k vocabulary. Model weights, checkpoints and datasets are not included.

| Phase | Task | Metric | Best submission | Test score |
|---|---|---|---|---|
| 1 | Legal LM pretraining (continue 512-word prompts by 128 words) | BERTScore F1 | B2 (GQA + RoPE decoder, 148.6M) + MBR decoding (chrF) | 0.8422 |
| 2 | Legal NER | strict span macro-F1 | weighted span-vote ensemble of 5 B full-FT runs + word-boundary snapping | 0.5313 |
| 3 | Rhetorical role prediction | macro-F1 | anchored hard-vote ensemble (H_anchor) of B2 PEFT / KD runs | 0.4659 |
| 4 | Court judgment prediction | macro-F1 | ENS05: C3 (B2 full FT, overlapping chunks) + E8_long (KD student), tuned threshold | 0.8082 |
| 5 | Statute identification (multi-label) | macro-F1 | L1: B2 full FT | 0.3804 |
| 6 | Abstractive summarisation | BERTScore F1 | R6: B2 full FT, max 210 words | 0.8389 |
| 7 | Prior case retrieval | Micro-F1@7 | P7: B2 LoRMA bi-encoder, random negatives | 0.3325 |

All experiments were run on Kaggle (2× NVIDIA T4), one notebook per phase (Save & Run All). The notebooks write every `.py` file of their folder with `%%writefile`, so each notebook runs on its own once the Phase-1 checkpoints and `legal_bpe.json` are attached as inputs.

## Contents
1. [Phase 1: Legal Language Model Pretraining](#phase-1-legal-language-model-pretraining)
2. [Phase 2: Legal Named Entity Recognition (L-NER)](#phase-2-legal-named-entity-recognition-l-ner)
3. [Phase 3: Rhetorical Role Prediction (RR)](#phase-3-rhetorical-role-prediction-rr)
4. [Phase 4: Court Judgment Prediction (CJPE)](#phase-4-court-judgment-prediction-cjpe)
5. [Phase 5: Legal Statute Identification (LSI)](#phase-5-legal-statute-identification-lsi)
6. [Phase 6: Abstractive Summarization of Legal Judgments](#phase-6-abstractive-summarization-of-legal-judgments)
7. [Phase 7: Prior Case Retrieval (PCR)](#phase-7-prior-case-retrieval-pcr)

---

## Phase 1: Legal Language Model Pretraining

Folder: [`phase1/`](phase1/)

### Problem statement
Pretrain a legal language model from scratch on the provided legal corpus.
- Model size 50M–300M parameters. No pretrained weights or external corpora.
- At least four substantially different architectures.
- Test task: continue a 512-word legal prompt with 128 words (1,000 test prompts, about 93% Supreme Court text).
- Metric: BERTScore F1 (secondary metrics: ROUGE-L F1, BLEU-4).

### Data and tokenizer
- **Corpus:** `Exploration-Lab/CS-781-Capstone`, `Phase_1_Pretraining/Train`. Four sources: Supreme Court (SC), Acts, High Courts (HC), Tribunals.
- **Tokenizer:** own byte-level BPE (`a1_tokenizer.py`, the A1 tokenizer extended for legal text).
  - 32,000 tokens: 32 special tokens, 256 byte tokens, 1 end-of-word marker and 31,711 learned merges.
  - Legal pre-tokenisation: letter runs, digit groups of at most 3, and punctuation runs are merged separately.
  - Incremental BPE training (pair counts, inverted index, lazy max-heap).
  - Tokenizer training text: 200M characters, mix SC 0.40 / Acts 0.30 / HC 0.15 / Tribunals 0.15.
- **Pretraining data** (`train.bin` / `val.bin`, about 1.0B training tokens):
  - Documents are whitespace-normalised. Documents under 100 words are dropped.
  - SC is capped at a 70% share (at most 4 epochs). Acts are capped at 8% (at most 2 epochs).
  - HC and Tribunal text fill the rest. Candidates are ranked by a DSIR-style unigram SC-likeness score, and the most SC-like are kept.
  - Validation: 2% of SC held out, plus 300 held-out prompt/continuation pairs for local scoring.

### Architectures (all written from scratch, starting from the A3 decoder code)
| Model | Architecture | Configuration | Parameters |
|---|---|---|---|
| A | Decoder-only, multi-head attention, sinusoidal positions | d 768, 17 layers, 12 heads, d_ff 3328, context 896 | 151.7M |
| B | Decoder-only, grouped-query attention (12 query / 4 KV heads), RoPE | d 768, 18 layers, d_ff 3456, context 896 | 148.6M |
| C | Encoder–decoder (GQA, RoPE) | 10 + 10 layers, d 768, kv 4, d_ff 3456, source 640 / target 256, decoder start = last source token | 202.7M |
| D | Prefix-LM (bidirectional over the first 640 tokens, causal after) | 20 layers, context 896 | 162.3M |
| **B2** | Model B, warm-started from B's best checkpoint and trained further | same as B | 148.6M |

All models use pre-norm, dropout 0.1, attention scale 0.125, init std 0.02, and the LM head tied to the token embedding.

### Training
| Run | Hardware | Batch (tokens/step) | LR schedule | Tokens | Best val loss |
|---|---|---|---|---|---|
| A ‖ B (`train_lm.py`, one model per T4) | Kaggle 2×T4 | 8,064 (9 × 896) | AdamW (0.9, 0.95), wd 0.1, lr 3e-4 → 3e-5 cosine, warmup 1000, clip 1.0 | max 350M (B: 350M in 43,403 steps) | B 2.5469 |
| C ‖ D (`train_lm.py`) | Kaggle 2×T4 | 8,192 | lr 2.5e-4 (D 3e-4), warmup 1500 | max 375M (C ~700M window tokens / ~200M target tokens; D ~367M) | — |
| **B2** (`train_ddp.py`, DDP over both T4s) | Kaggle 2×T4 | 16,128 | lr 3e-4 → 3e-5 cosine, warmup 1000, same optimiser as B | +971.7M (60,247 steps; about 1.32B in total) | **2.2394** (B: 2.547 on the same 360 windows) |

Fused SDPA attention and a KV cache (`fastattn.py`) wrap the model code for training and generation. A notebook check confirms they give the same outputs as the original forward pass.

### Decoding
`gen_local.py` generates 128-word continuations from a checkpoint. It supports:
- greedy decoding;
- temperature / top-p sampling;
- n-gram blocking;
- beam search;
- minimum Bayes risk (MBR) decoding: K sampled candidates (plus the greedy one), choosing the candidate with the highest average similarity to the others (chrF, BLEU or ROUGE-L utility).

### Results (Codabench test BERTScore F1)
| Model + decoding | Test BERTScore |
|---|---|
| A greedy | 0.7848 |
| A mild sampling | 0.8216 |
| C greedy / conservative / mild | 0.7992 / 0.8203 / 0.8248 |
| D greedy / conservative / mild | 0.8047 / 0.8218 / 0.8281 |
| B near-greedy (t 0.1, top-p 0.95) | 0.8131 |
| B greedy | 0.8124 |
| B mild sampling | 0.8317 |
| B mild + 4-gram blocking | 0.8323 |
| B greedy + 4-gram blocking | 0.8338 |
| B2 MBR (K 8, t 0.3, 3-gram blocking, + greedy), BLEU utility | 0.8411 |
| B2 MBR, ROUGE-L utility | 0.8407 |
| **B2 MBR, chrF utility (leaderboard entry)** | **0.8422** |
| B2 MBR, BERTScore utility (ablation only, not on the leaderboard) | 0.8431 |

**Local validation (300 held-out prompts, BERTScore roberta-large):**
- B2 sweep of 17 strategies: MBR chrF 0.8427, MBR BLEU 0.8424, MBR ROUGE-L 0.8422.
- B2 plain sampling: t 0.5 + 3-gram 0.8410, t 0.3 + 3-gram 0.8406 (B with t 0.3 + 3-gram: 0.8382).
- B2 beam search (W 4): 3-gram 0.8367, 4-gram 0.8358.
- Position-matched validation loss (positions 640–896): B2 2.119, B 2.436, D ~2.78, C ~3.1, A 3.71.

**Best submission:** B2 + MBR decoding (8 samples at temperature 0.3, 3-gram blocking, greedy candidate included, chrF utility): **0.8422**.

### Analysis
_To be written._

### Files
| File | Purpose |
|---|---|
| `a1_tokenizer.py` | Byte-level BPE tokenizer (train / encode / decode / save) |
| `arch3.py` | Model B / B2 (GQA + RoPE decoder) |
| `fastattn.py` | Fused SDPA attention, KV cache, sampling / beam / n-gram-blocked generation |
| `train_lm.py` | Single-GPU training + prediction (used for A, B) |
| `train_ddp.py` | Multi-GPU (DDP) training (used for B2) |
| `gen_local.py` | Generation from a checkpoint, decoding sweeps, MBR |
| `tools/fetch_test.py`, `tools/score_local.py` | Download the test prompts; local BERTScore / ROUGE-L / BLEU-4 |
| `notebooks/capstone_phase1_arch12.ipynb` | Kaggle notebook: tokenizer + data build, train A and B |
| `notebooks/capstone_phase1_refine.ipynb` | Kaggle notebook: B2 refinement (DDP) |

### How to run
1. **Kaggle (GPU T4 ×2, Internet on, secret `HF` = Hugging Face token):** run `capstone_phase1_arch12.ipynb` with Save & Run All. It builds the tokenizer and data, then trains A and B. Then run `capstone_phase1_refine.ipynb` with the first notebook's output attached, to train B2.
2. **Generation (local or Kaggle):**
   `python gen_local.py --ckpt B2=PATH/run_B2/best.pt --decoding decoding.json`
   - `--val-only` runs a sweep on the validation prompts.
   - `python tools/score_local.py DIR` scores validation generations.
3. **Submission:** a zip containing `prediction.json` at the root.

Model weights are not included.

---

## Phase 2: Legal Named Entity Recognition (L-NER)

Folder: [`phase2/`](phase2/)

### Problem statement
Predict character-level entity spans and their labels in Indian legal judgments.
- Metric: strict span macro-F1 over the 12 entity labels. A span counts only if start, end and label all match.
- Requirements:
  - all four Phase-1 models;
  - at least 6 submissions;
  - at least 5 fine-tuning techniques, including LoRMA (compulsory);
  - at least 1 knowledge-distillation run.

### Data
- Training: 70 documents (fold1 + fold2), 8,393 gold spans, about 6.7k tokens per document (466k tokens in total).
- Test: 35 documents.
- Validation: random 15% of the training documents (seed 42).

### Method
1. **Character ↔ token alignment** (`ner_align.py`): ByteBPE token offsets are mapped exactly back to characters.
   - Optional punctuation splitting (`split_punct`) raises the alignment ceiling from 0.9866 to 0.9945 macro-F1.
   - Tag scheme: BIO.
2. **Windows:** documents are cut into overlapping windows (A/B/D 768 tokens, stride 512; C 512, stride 384). Logits of overlapping windows are merged with the "center" rule.
3. **Backbone** (`backbones.py`): any Phase-1 model with its LM head removed.
   - C runs in encoder mode.
   - D uses a 640-token bidirectional prefix.
   - Students can be truncated to their first k blocks.
4. **Head** (`heads.py`): `linear`, `mlp` (GELU) or `mlp_crf` (MLP + CRF).
5. **Loss:** token cross-entropy with inverse-square-root class weights.
6. **Decoding:** BIO → spans (strict repair). Optional snapping of span boundaries to word boundaries (`snap_spans.py` / `--snap`).
7. **Training defaults:** AdamW, lr 2e-5 (head 1e-4), batch 16 windows, cosine schedule with 6% warm-up, wd 0.01, clip 1.0, up to 8 epochs, early stopping (patience 2) on validation macro-F1.
8. **Ensembling** (`ensemble.py --mode vote`):
   - Span-level weighted vote over member submissions: a span is kept if its summed member weight ≥ `min_votes`.
   - Optional per-label vote thresholds.
   - Overlapping spans are resolved by higher vote, then longer span.

### Fine-tuning techniques (`techniques/`)
| Technique | Settings used |
|---|---|
| Full fine-tuning | all parameters |
| LoRMA (multiplicative low-rank on the layer output) | r 8, alpha 16, targets q_proj / v_proj, lr 2e-4 |
| LoRA | r 8, alpha 16, targets q_proj / v_proj, lr 1e-4 |
| Partial fine-tuning | top k = 2 blocks + final LayerNorm, CRF head |
| BitFit | bias terms only, lr 5e-5 |
| Knowledge distillation | Hinton loss alpha·T²·KL + (1−alpha)·CE |

### Experiments and results (Codabench test strict macro-F1)
**Round 1** (one run per model/technique):

| Run | Model | Technique | Test |
|---|---|---|---|
| B_full | B | full FT | 0.4494 |
| B_lorma | B | LoRMA | 0.2066 |
| D_peft | D | BitFit | 0.1423 |
| A_lora | A | LoRA | 0.1350 |
| C_full | C (encoder) | full FT | 0.0936 |
| C_lorma | C (encoder) | LoRMA | 0.0556 |
| B_peft | B | partial FT (k 2) + CRF head | submitted (score not recorded) |

**Round 2** (one-factor sweep on B full FT):

| Run | Change vs B_full | Test |
|---|---|---|
| B_lr5e5 | lr 5e-5 | 0.4918 |
| B_batch8 | batch 8 | 0.4719 |
| B_lr3e5 | lr 3e-5 | 0.4690 |
| B_longer | 12 epochs | 0.4651 |
| B_wd005 | weight decay 0.05 | 0.4526 |
| B_window896 | 896-token windows | 0.4351 |
| B_batch32 | batch 32 | 0.4066 |
| B_unweighted | no class weights | 0.3968 |
| B_lr1e5 | lr 1e-5 | 0.3836 |

**Round 3:**

| Run | Change | Test |
|---|---|---|
| B_lr5e5_batch8 | lr 5e-5, batch 8 | **0.5033** (best single run) |
| B_lr5e5_seed123 | seed 123 | 0.5006 |
| B_lr5e5_head2e4 | head lr 2e-4 | 0.4984 |
| B_lr5e5_longer | longer training | 0.4934 |
| B_lr7e5 | lr 7e-5 | 0.4924 |
| B_lr1e4 | lr 1e-4 | 0.4909 |

**Knowledge distillation:**

| Run | Teacher → student | Settings | Test |
|---|---|---|---|
| KD_B_to_C (attempt 1) | B full FT → C encoder | — | 0.0000 (every token predicted O; val F1 0 for 3 epochs) |
| KD_B_to_C (attempt 2) | B_lr5e5 → C encoder | T 2.0, alpha 0.3 | 0.0000 (all O) |
| KD_B_to_Bsmall | B_lr5e5 (148.6M) → first 9 blocks of B (86.6M) | T 2.0, alpha 0.3, class-weighted CE, no early stopping | 0.3745 |

**Ensembles:** span vote over the five round-3 B runs (B_lr5e5_batch8, seed123, head2e4, lr7e5, lr1e4). Recipes are in `ensemble_recipes/`.

| Ensemble | Rule | Test |
|---|---|---|
| B_vote5_min2 | equal weights, ≥ 2 votes | 0.5181 |
| B_vote5_min3 | equal weights, ≥ 3 votes | 0.5269 |
| B_vote5_min3 + word snapping | as above, spans snapped to word boundaries | 0.5288 |
| S3 | + per-label thresholds: counsel labels 2, CASENO 4 | 0.5297 |
| S5 | + counsel 2, APP 2 | 0.5294 |
| S9 | S3 + strip trailing punctuation | 0.5260 |
| S6 | + counsel 2, CASENO 4, APP 2 | 0.5308 |
| **Final best** | one of the later vote variants (S8 / S10 / S11 / S11p; recipes in `ensemble_recipes/`). The exact zip was not recorded: **fill in** | **0.5313** |

Word snapping on validation: +0.015 to +0.034 macro-F1 for every member; vote5_min3 val 0.4352 → 0.4548.
Validation set: 10 documents. Its ranking of ensembles did not match the Codabench ranking.

### Analysis
_To be written._

### Files
| File | Purpose |
|---|---|
| `ner_data.py`, `ner_align.py`, `ner_metrics.py` | Data loading, char↔token alignment, strict span metrics |
| `backbones.py`, `phase1_models/`, `inject.py` | Phase-1 model wrappers and module-surgery helpers |
| `heads.py`, `techniques/` | Task heads and fine-tuning techniques (+ `kd.py`) |
| `train_ner.py`, `run_many.py` | Train one config / a queue of configs over 2 GPUs |
| `check_pipeline.py` | Oracle check of the whole prediction path (gold → tags → spans) |
| `ensemble.py`, `snap_spans.py` | Vote / probability ensembles; word-boundary snapping |
| `error_examples.py` | Per-label boundary / label / spurious / missed error examples |
| `configs/` | Config template and field reference |
| `ensemble_recipes/` | Members, weights and thresholds of every submitted ensemble |
| `capstone_phase2_lner.ipynb` | Kaggle notebook (round-1 configuration) |

### How to run
1. **Kaggle (GPU T4 ×2, secret `HF`):** attach the Phase-1 outputs (`run_A/B2/C/D/best.pt`, `legal_bpe.json`), then Save & Run All `capstone_phase2_lner.ipynb`.
2. **Locally:**
   - `python fetch_data.py` downloads the data.
   - `python train_ner.py config.json` trains one run (see `configs/template.json`).
3. **Ensemble from submission zips:**
   - `python ensemble.py --name ENS --runs a.zip b.zip ... --data data --mode vote --weights 3 3 2 1 1 --min-votes 5 --snap --out out`
   - Add `--label-min-votes` for per-label thresholds.

Model weights are not included.

---

## Phase 3: Rhetorical Role Prediction (RR)

Folder: [`phase3/`](phase3/)

### Problem statement
Assign each sentence of a legal judgment one of 13 rhetorical-role labels, such as Fact, Issue, ArgumentPetitioner, Statute, PrecedentReliedUpon, RatioOfTheDecision and RulingByPresentCourt.
- Data covers two domains: CL and IT.
- Metric: macro-F1 over labels.
- Allowed techniques: LoRA, QLoRA, AdaLoRA, rsLoRA, LoRMA (compulsory), prefix tuning, prompt tuning, IA³, and KD (mandatory).

### Data
- Training: 80 documents, 16,845 sentences (PrecedentOverruled has only 3 training sentences).
- Dev: 10 documents (5 CL, 5 IT), 2,142 sentences, 11 of the 13 labels present.
- Test: 10 documents.

### Method
1. **Sentence windows** (`rr_data.py`): each sentence is encoded with the ByteBPE.
   - Each target sentence (up to 128 tokens) is packed with 64 tokens of left and 64 of right context.
   - Up to 4 target sentences per window.
2. **Encoding:** the backbone is a Phase-1 model (B2 unless noted). Each target sentence's tokens are mean-pooled.
3. **Head:** MLP (hidden 256, dropout 0.1); optionally an MLP + CRF head.
4. **Loss:** cross-entropy with inverse-square-root class weights.
5. **Training:** batch 16, cosine schedule, up to 8 epochs. Early stopping on dev macro-F1 (patience 2) in campaigns 1–2; no early stopping in campaign 3.
6. **Ensembling** (`rr_ensemble.py`, `ens_family/`): sentence-level weighted hard majority vote. Ties go to the earliest-listed member (the "anchor").

### Fine-tuning techniques (`techniques/`)
| Technique | Settings used (r 8 / alpha 16, targets attention + FFN linears, lr 1e-4 unless noted) |
|---|---|
| LoRA | h = W0 x + (alpha/r) B A x |
| LoRMA | multiplicative low-rank update on the layer output |
| rsLoRA | scale alpha / √r |
| QLoRA | NF4 4-bit frozen weights (block 64) + LoRA, implemented from scratch |
| AdaLoRA | SVD-form update, importance-based rank pruning (target rank 2 or 4), orthogonality penalty |
| IA³ | learned scaling vectors on keys, values and the FFN inner activation |
| Prompt tuning | 8 or 16 soft prompt tokens |
| Prefix tuning | 8 learned key/value prefixes per layer |
| KD | B2 LoRA teacher → B2 with 12 of 18 layers, LoRA student, T 2, alpha 0.5 |

### Experiments and results (Codabench test macro-F1)
**Campaign 1** (20 runs, early stopping):

| Run | Model | Technique / change | Test |
|---|---|---|---|
| B_lorma | B2 | LoRMA r8, lr 2e-4 | 0.4559 |
| B_lora_lr2e4 | B2 | LoRA r8, lr 2e-4 | 0.4522 |
| B_rslora_r16 | B2 | rsLoRA r16 / alpha 32 | 0.4517 |
| KD | B2 (12 of 18 layers) | LoRA student, teacher B_lora, T 2, alpha 0.5 | 0.4501 |
| B_lora_r16 | B2 | LoRA r16 / alpha 32 | 0.4461 |
| B_rslora | B2 | rsLoRA r8 | 0.4460 |
| B_qlora | B2 | QLoRA r8, NF4 block 64 | 0.4055 |
| B_lora | B2 | LoRA r8, lr 1e-4 | 0.4015 |
| B_lora_lr5e5 | B2 | LoRA r8, lr 5e-5 | 0.3937 |
| B_lora_ctx_local | B2 | LoRA r8, no left/right context | 0.3927 |
| B_lora_inv_freq | B2 | LoRA r8, inverse-frequency class weights | 0.3868 |
| B_prompt_p16 | B2 | prompt tuning, 16 tokens | 0.3816 |
| D_run | D | LoRA r8, prefix 640 | 0.3801 |
| B_prefix_tuning | B2 | prefix tuning, 8 prefixes | 0.3793 |
| B_adalora_r4 | B2 | AdaLoRA r8 → target rank 4 | 0.3636 |
| B_adalora | B2 | AdaLoRA r8 → target rank 2 | 0.3632 |
| B_ia3 | B2 | IA³ | 0.3623 |
| B_prompt_tuning | B2 | prompt tuning, 8 tokens | 0.3552 |
| A_run | A | LoRA r8 | 0.3502 |
| C_run | C (encoder) | LoRA r8, window 512 | 0.3108 |

**Campaign 2** (single-factor changes):

| Run | Change | Test | Counterpart |
|---|---|---|---|
| B_rslora_r16_lr2e4 | lr 2e-4 | 0.4498 | 0.4517 |
| B_rslora_r16_lr5e5 | lr 5e-5 | 0.4445 | 0.4517 |
| KD_seed123 | seed 123 | 0.4407 | 0.4501 |
| B_lora_ctx_wide | context 128 each side | 0.4269 | 0.4015 |
| B_lora_seed123 | seed 123 | 0.4147 | 0.4015 |
| B_lora_unweighted | no class weights | 0.4009 | 0.4015 |
| B_lora_crf | MLP + CRF head | 0.3968 | 0.4015 |

**Campaign 2 ensembles** (hard vote):
- ENS_top3: 0.4458
- ENS_weighted (3, 3, 2, 1, 1): 0.4424
- ENS_top5: 0.4373
- ENS_top4: 0.4360
- ENS_recall: 0.4309
- ENS_candidate (7-run vote + per-label routing, `ens_experiments.py`): 0.4475 (dev 0.5274)

**Campaign 3** (all 27 runs retrained without early stopping, suffix `_allep`; only runs whose predictions changed noticeably were submitted):

| Run | Best epoch / epochs | Dev all / CL / IT | Test |
|---|---|---|---|
| B_lora_ctx_wide_allep | 4 / 8 | 0.5109 / 0.4658 / 0.5333 | 0.4616 |
| KD_allep | 8 / 8 | 0.4858 / 0.3783 / 0.5292 | 0.4496 |
| KD_seed123_allep | 5 / 8 | 0.4916 / 0.4016 / 0.5343 | 0.4461 |
| B_lorma_allep | 2 / 8 | 0.4781 / 0.3443 / 0.5266 | 0.4386 |

**Anchored ensembles** (`ens_family/`; anchor = B_lora_ctx_wide_allep):

| Submission | Members (weights) | Dev all / CL / IT | Test |
|---|---|---|---|
| **H_anchor_w2.0+top4net** | anchor ×2; KD_seed123_allep, B_rslora_r16_allep, B_lorma_allep, B_rslora_r16_lr5e5_allep ×1 | 0.5251 / 0.4459 / 0.5495 | **0.4659** (best, leaderboard) |
| G2 | anchor, B_lorma_allep, KD_seed123_allep ×1 | 0.5320 / 0.4509 / 0.5670 | 0.4652 |
| C1 (one per family) | anchor, KD_seed123_allep, D_run_allep, B_prompt_p16_allep ×1 | 0.5177 / 0.4370 / 0.5505 | 0.4634 |

Seed replicas: B_lora 0.4015 vs 0.4147; KD 0.4501 vs 0.4407.

### Analysis
_To be written._

### Files
| File | Purpose |
|---|---|
| `rr_data.py`, `rr_metrics.py` | Sentence windows with context; macro-F1 and per-label metrics |
| `backbones.py`, `phase1_models/`, `inject.py`, `heads.py`, `techniques/` | Backbone wrappers, heads, techniques (incl. AdaLoRA hooks, QLoRA, IA³, prompt/prefix tuning) |
| `train_rr.py`, `run_many.py`, `check_pipeline_rr.py` | Training, multi-GPU queue, oracle pipeline check |
| `rr_ensemble.py` | Weighted sentence-level vote |
| `ens_experiments.py` | Campaign-2 routing ensemble experiment |
| `ens_family/` | Audit (steps 1–2) and family / anchored ensembles (steps 3–9) that produced H_anchor |
| `capstone_phase3_rr.ipynb` | Kaggle notebook |

### How to run
1. **Kaggle (2×T4, secret `HF`):** attach the Phase-1 outputs, then Save & Run All the notebook.
2. **Locally:** `python train_rr.py config.json`.
3. **Ensembles from finished run folders:** `cd ens_family && python step12_audit.py && python step3to9_ensembles.py`. The input run directory is set in `common.py`.

Model weights are not included.

---

## Phase 4: Court Judgment Prediction (CJPE)

Folder: [`phase4/`](phase4/)

### Problem statement
Predict from the full text of a Supreme Court judgment whether the appeal was accepted (1) or rejected (0).
- Metric: macro-F1.
- Allowed techniques: full FT, partial FT, layer freezing, BitFit, LoRMA (compulsory), AdaLoRA, adapters, and KD (mandatory).
- All four Phase-1 models must be used.

### Data
- Train: 37,387 documents (59% / 41%). Exact duplicates are removed, leaving 31,653 distinct texts.
- Dev: 3,505 rows (2,511 distinct). Test: 1,517 documents.
- Length: median about 2.7k (train) / 3.5k (dev, test) ByteBPE tokens. About 4% of dev/test documents fit in 1,024 tokens.
- All 1,517 test documents also appear in the released dev set with labels. The course TA confirmed this overlap is intentional. Three dev subsets are therefore scored:
  - **dev_all**: the released dev set;
  - **dev_excl_test**: 994 distinct dev documents not in test;
  - **dev_in_test**: the test documents, equal to the Codabench score.

### Method (chunked document classifier)
1. **Chunking** (`cjpe_data.py`): each document is split into up to `max_chunks` chunks of the model's context length. Three strategies:
   - `even`: evenly spaced;
   - `headtail`: head + tail;
   - `overlap`: 896-token windows with stride 640.
2. **Encoding:** each chunk is encoded by the Phase-1 backbone and mean-pooled.
3. **Pooling and head:** attention pooling over the chunks gives one document vector, followed by an MLP head (hidden 256, dropout 0.1, 2 outputs).
4. **Training:** cross-entropy loss, 1 epoch, 8 documents per step (micro-batch 2).
   - AdamW: lr 5e-5 (head 1e-4), wd 0.01, warmup 6%, cosine schedule, clip 1.0.
   - Checkpoint selected on dev_all.
5. **Post-processing** (no retraining; saved probabilities only):
   - threshold tuning (`cjpe_select.py`);
   - probability-averaging ensembles;
   - inference-time re-chunking (`predict_cjpe.py`), e.g. 8 overlapping chunks;
   - dual-split ensemble optimisation (`dual_split_sweep.py`).

### Experiments (`experiments.py`)
| Run | Model | Technique / change | Trainable params | Train min |
|---|---|---|---|---|
| E1_B2_full | B2 | full FT, even chunks (4) | 148.8M | 95.9 |
| E2_B3_full | B3 (B2 context-extended to 1024) | full FT | 148.8M | 109.4 |
| E3_A_partial | A | partial FT, top 8 of 17 blocks | 60.0M | 51.0 |
| E4_C_adapters | C encoder | bottleneck adapters (64) on attention + FFN, 640-token chunks, lr 1e-4 | 2.18M | 39.9 |
| E5_D_bitfit | D (prefix 640) | BitFit, lr 1e-4 | 0.31M | 99.6 |
| E6_B2_lorma | B2 | LoRMA r8 / alpha 16, attention + FFN, lr 2e-4 | 2.0M | 101.2 |
| E7_B2_adalora | B2 | AdaLoRA r16 → target 4, lr 1e-4 | 4.1M | 141.9 |
| C1_B2_headtail | B2 | full FT, head + tail chunks | 148.8M | 103.5 |
| C3_B2_overlap | B2 | full FT, overlapping windows (stride 640) | 148.8M | 118.0 |
| W1_B2_full_weighted | B2 | full FT, class weights 0.847 / 1.220 | 148.8M | 98.3 |
| E8_KD_B2s12 | B2, first 12 of 18 blocks | KD student, full FT, T 2, alpha 0.5, teacher = best account-1 run (C3) | 107.4M | 64.5 |

### Results
**Single runs** (threshold 0.5):

| Run | dev_all | dev_excl_test | Test macro-F1 |
|---|---|---|---|
| C3_B2_overlap | 0.7489 | 0.7422 | 0.7574 |
| W1_B2_full_weighted | 0.7435 | 0.7496 | 0.7355 |
| E8_KD_B2s12 | 0.7427 | 0.7493 | 0.7341 |
| C1_B2_headtail | 0.7281 | 0.7273 | 0.7291 |
| E1_B2_full | 0.7357 | 0.7447 | 0.7238 |
| E7_B2_adalora | 0.5525 | 0.5620 | 0.5402 |
| E5_D_bitfit | 0.5120 | 0.5192 | 0.5027 |
| E3_A_partial | 0.4663 | 0.4642 | 0.4690 |
| E4_C_adapters | 0.3844 | 0.3773 | 0.3935 |
| E6_B2_lorma | 0.3749 | 0.3832 | 0.3641 |
| E2_B3_full | 0.3329 | 0.3333 | 0.3323 (predicts class 0 for every document) |

**Dev confusion matrices (dev_all, [[g0p0, g0p1], [g1p0, g1p1]]):**
- C3: [[1614, 135], [721, 1035]]
- W1: [[1494, 255], [634, 1122]]
- E8: [[1585, 164], [716, 1040]]
- E1: [[1567, 182], [723, 1033]]
- C1: [[1553, 196], [735, 1021]]
- E7: [[1491, 258], [1199, 557]]
- E5: [[1617, 132], [1368, 388]]
- E3: [[1680, 69], [1495, 261]]
- E4: [[1700, 49], [1658, 98]]
- E6: [[1732, 17], [1683, 73]]
- E2: [[1749, 0], [1756, 0]]

**Threshold tuning and ensembles:**

| Submission | Description | Test |
|---|---|---|
| C3 tuned | threshold 0.30 (tuned on dev_excl_test) | 0.7854 |
| ENS C3 0.75 / E8 0.25 | probability average, threshold 0.34 | 0.7864 |
| ENS_bestth C3 0.75 / E8 0.25 | each model centred at its own tuned threshold (0.295 / 0.3605) | 0.7887 |
| E8 tuned | threshold 0.3605 | 0.7640 |
| C3_long | C3 retrained for 5 epochs (best checkpoint epoch 2, step 7000) | 0.7633 |
| E8_long | KD student retrained for 8 epochs | 0.7577 (checkpoint chosen on test labels: 0.7626) |
| ENS C3_avg8 + E8L_avg8 + E8_1ep | re-chunked (8-chunk variants averaged), equal weights, threshold 0.361, selected on held-out dev_excl_test | 0.8013 |
| ENS C3_own + E8L_avg8 | threshold 0.409 | 0.8042 |
| best_robust_A_ENS07 | dual-split Track A (selected on held-out dev_excl_test) | 0.7831 |
| **best_track_B_overlap_ENS05** | C3 + E8_long_A, weights 0.587 / 0.413, threshold 0.392. **Tuned on dev_in_test, i.e. on the test documents' labels from dev.json** | **0.8082** (leaderboard) |

All Codabench scores equal the locally computed dev_in_test values to 4 decimals.

### Analysis
_To be written._

### Files
| File | Purpose |
|---|---|
| `analyze_cjpe.py` | Dataset statistics (lengths, duplicates, overlap) |
| `cjpe_data.py` | Loading, de-duplication, chunking |
| `backbones.py`, `phase1_models/`, `inject.py`, `heads.py`, `techniques/` | Backbones, head, techniques |
| `experiments.py` | Experiment matrix |
| `train_cjpe.py`, `run_many.py` | Training (1 run / queue over 2 GPUs) |
| `predict_cjpe.py` | Inference-only re-chunking of a finished run |
| `cjpe_select.py` | Threshold selection and probability-average ensembles |
| `dual_split_sweep.py` | Dual-split (Track A / Track B) ensemble optimisation that produced ENS05 |
| `capstone_phase4_cjpe.ipynb` | Kaggle notebook (`ACCOUNT` = 1 / 2 / "long" / "predict") |

### How to run
1. **Kaggle (2×T4, secret `HF`):** attach the Phase-1 outputs (and the B3 output for E2), set `ACCOUNT`, then Save & Run All.
2. **Locally:**
   - `python train_cjpe.py config.json`
   - `python predict_cjpe.py <run_dir> <out_dir> overlap:8:640`
   - `python dual_split_sweep.py` (reads the saved probabilities of the member runs)

Model weights are not included.

---

## Phase 5: Legal Statute Identification (LSI)

Folder: [`phase5/`](phase5/)

### Problem statement
Multi-label classification: predict every statute relevant to a legal document.
- Metric: macro-F1 over statute labels (secondary metrics: micro-F1, subset accuracy). Rare statutes count as much as frequent ones.
- Allowed techniques: full FT, partial FT, QLoRA, LoRMA (compulsory), AdaLoRA, adapters, prefix tuning, prompt tuning, and KD (mandatory).

### Data
- Train: 42,750 documents. Dev: 10,181. Test: 13,019.
- 100 statute labels (label space = all statute ids seen in train + dev). On average 3.7 labels per training document.
- Leakage guards: dev documents whose text also appears in test are excluded from model selection (**dev_excl_test**).

### Method (chunked multi-label document classifier, adapted from Phase 4)
1. **Chunking:** overlapping windows, up to 8 chunks per document, stride 640 (C: 640-token chunks, stride 448).
2. **Encoding:** each chunk is encoded by the Phase-1 backbone, then chunk mean pooling → attention pooling over chunks → MLP head (hidden 256, dropout 0.1).
3. **Output:** 100 independent sigmoid outputs.
4. **Loss:** BCE with `pos_weight` = √(inverse label frequency), capped at 5.
5. **KD:** each statute is treated as its own 2-class problem (logits [0, z] → softmax = sigmoid).
6. **Training:** 2 epochs, 8 documents per step (micro-batch 2).
   - AdamW: wd 0.01, warmup 6%, cosine schedule, clip 1.0, head lr 1e-4.
   - Best checkpoint saved on dev_excl_test (and dev_all).
   - The runs were split across two Kaggle accounts (`SPLIT` in `experiments.py`).
7. **Prediction:** a statute is predicted if P ≥ 0.5. `lsi_select.py` adds tuned global / per-statute thresholds and probability-average ensembles.

### Experiments (`experiments.py`)
| Run | Model | Technique | Settings |
|---|---|---|---|
| L1_B2_full | B2 | full FT | lr 2e-5 |
| L2_A_partial | A | partial FT | top 6 blocks, lr 3e-5 |
| L3_B3_qlora | B3 | QLoRA | r 16, alpha 32, attention + FFN, NF4 block 64, lr 1e-4 |
| L4_B2_lorma | B2 | LoRMA | r 16, alpha 32, attention + FFN, lr 5e-5 |
| L5_C_adapters | C encoder | bottleneck adapters | bottleneck 64, lr 5e-5 |
| L6_D_prefix | D (prefix 640) | prefix tuning | 32 prefix tokens, lr 5e-4 |
| L7_B2_kd | B2, first 12 blocks | KD student, full FT | teacher L1, T 2.0, alpha 0.5, lr 2e-5 |
| L8_B2_adalora | B2 | AdaLoRA | r 16 → target rank 8, alpha 32, t_init 100, t_final 1000, Δt 10, β 0.85/0.85, orth_reg 0.1, lr 5e-5 |

### Results
| Run | Dev macro-F1 (threshold 0.5) | Codabench test macro-F1 |
|---|---|---|
| **L1_B2_full** | 0.3896 | **0.3804** (leaderboard) |
| L3_B3_qlora | 0.3624 | 0.3571 |
| L4_B2_lorma | 0.3441 | 0.3394 |
| L7_B2_kd | 0.3120 | 0.3083 |
| L8_B2_adalora | 0.2604 | 0.2555 |
| L2_A_partial | 0.2275 | 0.2225 |
| L6_D_prefix | 0.2100 | 0.2095 |
| L5_C_adapters | 0.1294 | 0.1283 |

Dev = macro-F1 on dev_excl_test at threshold 0.5. L1, L4, L6 and L7 values are from account 1's `summary.csv` (best epoch 2).

**Best submission:** L1_B2_full (B2 full fine-tuning), test macro-F1 0.3804.

### Analysis
_To be written._

### Files
| File | Purpose |
|---|---|
| `lsi_data.py` | Loading, label space, leakage guards, chunking, metrics |
| `backbones.py`, `phase1_models/`, `inject.py`, `heads.py`, `techniques/` | Backbones, head, techniques |
| `experiments.py` | Experiment matrix, account split, KD teacher/student |
| `train_lsi.py`, `run_many.py` | Training (1 run / queue over 2 GPUs) |
| `predict_lsi.py` | Inference-only re-prediction of a finished run |
| `lsi_select.py` | Threshold selection (global / per-statute) and probability ensembles |
| `capstone_phase5_lsi.ipynb` | Kaggle notebook (`ACCOUNT` = 1 / 2) |

### How to run
1. **Kaggle (2×T4, secret `HF`):** attach the Phase-1 outputs (A, B2, B3, C, D), set `ACCOUNT`, then Save & Run All.
2. **Locally:**
   - `python train_lsi.py config.json`
   - `python lsi_select.py --name NAME --out ensembles --members runs/L1_B2_full [...]`

Model weights are not included.

---

## Phase 6: Abstractive Summarization of Legal Judgments

Folder: [`phase6/`](phase6/)

### Problem statement
Generate an abstractive summary of a legal judgment.
- Metric: BERTScore F1 (secondary metrics: ROUGE-1 F1, ROUGE-L F1).
- Requirements:
  - all four Phase-1 models;
  - at least 6 experiments;
  - at least 5 techniques from {Houlsby / Pfeiffer adapters, prefix tuning, prompt tuning, LoRA, QLoRA, AdaLoRA, LoRMA (compulsory)};
  - KD (mandatory).

### Data (`analyze_summ.py`, `analyze_align.py`)
- Train: 7,030 judgments with summaries. Test: 100 judgments.
  - No overlap between train and test.
  - 27 duplicate documents and 2 empty summaries.
- There is no dev split, so 500 training documents (seed 42) are held out for validation.
- Length in ByteBPE tokens:
  - documents: median 3,706 (p95 13.3k);
  - summaries: median 758 (p95 2.4k). Only 39% of summaries fit in 640 tokens, and 4% in 256.
- A summary is about 21% of the document's words (correlation 0.75).
- Summaries mostly follow document order: median Kendall τ between summary sentences and 600-word document chunks is 0.78.
- First and last chunks are over-represented in the summaries. The novel 4-gram rate is about 0.50.

### Method
1. **Source selection:** head + tail selection of `src_len` tokens (half head, half tail).
2. **Target:** the reference summary truncated to `tgt_len` tokens at a sentence boundary, followed by `</s>`.
3. **Model** (`gen_model.py`): the Phase-1 model keeps its tied LM head.
   - A / B2 / D read `[source] </s> [summary]`; D is bidirectional over the source.
   - C encodes the source and decodes the summary (decoder starts from the last source token).
4. **Loss:** token cross-entropy on summary tokens only. For KD, the loss is `kd_loss` (`techniques/kd.py`) over the vocabulary on the target tokens, with the teacher frozen.
5. **Training defaults:**
   - batch 8 examples (micro-batch 1), AdamW wd 0.01, warmup 6%, cosine schedule, clip 1.0;
   - evaluation every 250 steps, patience 3 evaluations;
   - checkpoint with the lowest validation loss.
6. **Decoding** (`summ_infer.py`):
   - length = 0.25 × document words, clamped to 20–180 words;
   - EOS allowed, temperature 0.3, no repeated 3-grams.
7. **Local scoring** (`score_summ.py`): ROUGE-1/L and BERTScore (roberta-large) on 100 validation documents.
8. **Stage 7 sweep:** every run was decoded again with only `max_words` changed (A/B2 210, C 180, D 225).

### Experiments (`experiments.py`)
| Run | Model | Technique | Settings | Epochs |
|---|---|---|---|---|
| R1_B2_LoRMA | B2 | LoRMA | r 8, alpha 16, q/k/v/out + FFN, lr 2e-4 | 10 |
| R2_A_LoRA | A | LoRA | r 8, alpha 16, q/k/v/out, lr 1e-4 | 10 |
| R3_B2_QLoRA | B2 | QLoRA | r 8, alpha 16, q/k/v/out, NF4 block 64, lr 1e-4 | 10 |
| R4_C_Houlsby | C | Houlsby adapters | bottleneck 64, source 384 / target 256, lr 1e-4 | 12 |
| R5_D_Prefix | D | prefix tuning | 16 prefix tokens, source 624, lr 5e-4 | 10 |
| R6_B2_FullFT | B2 | full FT | lr 2e-5 | 8 |
| R7_B2_KD | B2, first 12 blocks | KD student, full FT | teacher R1, T 2.0, alpha 0.5, lr 2e-5 | 10 |

### Results (Codabench test BERTScore F1)
| Run | Original decode (max 180 words) | Stage-7 sweep |
|---|---|---|
| R1 B2 LoRMA | 0.8344 | 0.8348 (max 210) |
| R2 A LoRA | 0.8189 | 0.8182 (max 210) |
| R3 B2 QLoRA | 0.8366 | 0.8349 (max 210) |
| R4 C Houlsby | 0.8251 | 0.8236 (max 180, identical settings) |
| R5 D Prefix | 0.8291 | 0.8279 (max 225) |
| **R6 B2 full FT** | 0.8374 | **0.8389** (max 210, leaderboard) |
| R7 B2 KD | 0.8294 | 0.8295 (max 210) |

R4's two submissions used identical decoding settings. Their scores differ by 0.0015, from sampling at temperature 0.3.

**Best submission:** `submission_R6_B2_FullFT__maxw210.zip`, BERTScore 0.8389.

### Analysis
_To be written._

### Files
| File | Purpose |
|---|---|
| `analyze_summ.py`, `analyze_align.py` | Dataset length statistics; summary ↔ document alignment |
| `summ_data.py` | Loading, cleaning, split, sentence-level encoding, head+tail selection, target building |
| `gen_model.py` | Phase-1 model + LM head for training and KV-cached generation (incl. prompt/prefix tuning) |
| `phase1_models/`, `inject.py`, `techniques/` | Model code, module surgery, techniques |
| `experiments.py` | Experiment matrix and account split |
| `train_summ.py`, `run_many.py` | Training (1 run / queue over 2 GPUs, KD waits for its teacher) |
| `summ_infer.py`, `score_summ.py` | Generation (val / test, sharding, zip); ROUGE + BERTScore |
| `capstone_phase6_summ.ipynb` | Kaggle notebook |

### How to run
1. **Kaggle (2×T4, secret `HF`):** attach the Phase-1 outputs, then Save & Run All.
2. **Locally:**
   - `python train_summ.py config.json`
   - `python summ_infer.py --run RUN_DIR --split test --mode single --decode decode.json --out out.json`

Model weights are not included.

---

## Phase 7: Prior Case Retrieval (PCR)

Folder: [`phase7/`](phase7/)

### Problem statement
Given a query case, retrieve the most relevant prior cases from a candidate pool.
- For every test query, submit exactly 7 unique candidate ids from that split's own pool, ordered by relevance.
- Metric: Micro-F1@7 (secondary metrics: Micro-Precision@7, Micro-Recall@7). The gold set is not truncated or padded to 7.
- Requirements:
  - at least 6 experiments;
  - at least 5 techniques from {full FT, partial FT, QLoRA, LoRMA (compulsory), prefix tuning, prompt tuning, IA³};
  - KD (mandatory).

### Data
| Split | Queries | Candidate pool | Relevant per query |
|---|---|---|---|
| Train | 827 | 4,320 | mean 6.4, median 4 (max 52) |
| Dev | 118 | 1,023 | mean 8.3, median 5 (max 92) |
| Test | 237 | 1,727 | hidden |

- Every query is also in its own pool, but it is never relevant to itself, so it is excluded from its own ranking.
- Case length: median about 4.6–5.9k ByteBPE tokens.
- 5,308 (query, relevant case) training pairs.
- **Pool structure:**
  - nearly every candidate is relevant to exactly one query (1.06 queries per relevant candidate on dev, 1.29 on train);
  - only 5% of dev candidates are shared between queries.
- **Upper bound:** the best possible F1@7 given the gold-set sizes is 0.618 on dev (0.639 on train).
- **BM25 reference** (`pcr_select.py --bm25-only`): dev F1@7 0.102, recall@100 0.32.

### Method (bi-encoder)
1. **Document encoder** (`train_pcr.py`, the Phase-4/5 chunked encoder): case text → ByteBPE → up to 6 evenly spaced chunks of 896 tokens. Each chunk is encoded by the Phase-1 backbone and mean-pooled. Attention pooling over chunks (or mean) follows, then a projection to 256 dimensions. The same encoder embeds queries and candidates.
2. **Score:** cosine similarity / temperature 0.05.
3. **Loss:** InfoNCE over each (query, relevant case) pair's candidate set.
   - Negatives: BM25 hard negatives (2 per positive, from the BM25 top-100) or random negatives.
   - In-batch negatives.
   - The query's other relevant cases and the query itself are masked out.
4. **Gradient caching:** a step first embeds all its documents without gradients, takes the gradient of the loss with respect to the embeddings, then re-encodes each micro-batch with gradients. This allows large candidate sets on T4 memory.
5. **KD:** the teacher's embeddings of every training document are cached. The student's and teacher's score rows over the same candidates go to `kd_loss`, which replaces InfoNCE.
6. **Training:** 2 epochs (capped at 250 minutes per run), 8 pairs per step (2 documents per micro-batch).
   - AdamW: head lr 3e-4, wd 0.01, warmup 5%, cosine schedule, clip 1.0, seed 42.
   - Dev evaluated before training and at each epoch end; the best checkpoint is kept on dev Micro-F1@7.
7. **Prediction:** rank the test pool for every test query (query excluded) and write the top 7 to `prediction.json`.

### Experiments (`experiments.py`; all on B2)
| Run | Technique | Settings | Change vs BASE |
|---|---|---|---|
| P1 | full FT | lr 1e-5 | — |
| P2 | partial FT | top 6 blocks, lr 2e-5 | — |
| P3 | QLoRA | r 16, alpha 32, attention + FFN, NF4 block 64, lr 1e-4 | — |
| P4 | LoRMA | r 16, alpha 32, attention + FFN, lr 5e-5 | — (KD teacher) |
| P5 | prefix tuning | 32 prefix tokens, lr 2e-4 | — |
| P6 | LoRMA | as P4 | mean pooling over chunks instead of attention |
| P7 | LoRMA | as P4 | random negatives instead of BM25 hard negatives |
| P8_kd | KD student: B2 truncated to 9 blocks, full FT | lr 2e-5, T 2.0, alpha 0.5, teacher P4 | 86.8M vs teacher 152.4M parameters |

Trainable parameters:
- P1: 148.8M
- P2: 41.5M
- P3: 4.1M of 28.7M
- LoRMA runs: 3.8M
- P5: 0.49M

Runs were split over two Kaggle accounts: account 1 ran P1, P3, P5 and P7; account 2 ran P2, P4, P6 and P8_kd.

### Results (Codabench test Micro-F1@7)
| Run | Test |
|---|---|
| **P7** (LoRMA, random negatives) | **0.3325** (leaderboard) |
| P6 (LoRMA, mean chunk pooling) | 0.3296 |
| P3 (QLoRA) | 0.3255 |
| P1 (full FT) | 0.3249 |
| P4 (LoRMA) | 0.3243 |
| P2 (partial FT) | 0.3231 |
| P8_kd (KD student) | 0.3201 |
| P5 (prefix tuning) | 0.3172 |

Dev Micro-F1@7 per run (untrained start vs best) is in each run's `summary.json` / `dev_scores.json` on Kaggle: _fill in_.

**Post-deadline local check (not submitted):** on BM25 dev scores, normalising each candidate's column of scores across all queries changes dev F1@7:
- per-query ranking only: 0.102;
- row z-score + column z-score: 0.331;
- column z-score only: 0.409.

### Analysis
_To be written._

### Files
| File | Purpose |
|---|---|
| `fetch_data.py` | Download the `Phase_7_PCR` files |
| `pcr_data.py` | Loading, ByteBPE encoding cache, chunking, numpy BM25, ranking, Micro-P/R/F1@k, prediction writer with format checks |
| `backbones.py`, `phase1_models/`, `inject.py`, `techniques/` | Backbones and techniques |
| `experiments.py` | Experiment matrix, account split, KD teacher/student |
| `train_pcr.py`, `run_many.py` | Bi-encoder training with gradient caching (1 run / queue over 2 GPUs) |
| `pcr_select.py` | Score fusion of finished runs (+ BM25), BM25 reference score |
| `capstone_phase7_pcr.ipynb` | Kaggle notebook (`ACCOUNT` = 1 / 2 / "all") |

### How to run
1. **Kaggle (2×T4, secret `HF`):** attach the Phase-1 outputs (B2 checkpoint + `legal_bpe.json`), set `ACCOUNT`, then Save & Run All.
2. **Locally:**
   - `python fetch_data.py`
   - `python train_pcr.py config.json`
   - `python pcr_select.py --bm25-only --data-dir data/Phase_7_PCR --out out`

Model weights are not included.
