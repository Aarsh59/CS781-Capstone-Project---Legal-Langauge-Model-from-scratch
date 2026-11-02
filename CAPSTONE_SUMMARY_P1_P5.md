# CS781 Capstone summary, Phases 1–6 (as of 2026-11-02)


- `phase3/codabench_phase3_scores.md`
- `phase4/codabench_phase4_scores.md`
- project notes

**Rules that apply to every phase:**
- Phases 2–7 start from our own Phase-1 models.
- Phases 2–4 must use all four Phase-1 models.
- At least 6 experiments and at least 6 Codabench submissions per phase.
- At least 5 different fine-tuning (FT) or PEFT techniques per phase. LoRMA is compulsory, and at least 1 knowledge-distillation (KD) run is mandatory (teacher larger than student).
- Changing only hyperparameters, heads or pooling does not count as a new technique.
- Rule 9: the LLM may not write architectures or techniques, tune hyperparameters, or write analysis.

---

## 1. Phase-1 models (pretraining from scratch)

**Task:** pretrain a legal language model from scratch.
- Size 50–300M parameters, trained only on the provided legal corpus.
- At least 4 substantially different architectures.
- Test: continue a 512-word prompt with 128 words.
- Metric: BERTScore F1 (secondary metrics: ROUGE-L, BLEU-4).
- Tokenizer: own byte-level BPE (ByteBPE) trained on legal text, 32k vocab. Pretraining data is about 70% Supreme Court (SC), plus Acts, High Court (HC) and Tribunal text.

| Model | Architecture | Params | Context | Notes |
|---|---|---|---|---|
| **A** | Decoder-only, standard multi-head attention (MHA), d768, 17 layers | 151.7M | 896 | Baseline, based on the A3 decoder |
| **B** | Decoder-only, grouped-query attention (GQA, 12 query heads → 4 KV heads), RoPE, d768, 18 layers | 148.6M | 896 | 346.8M training tokens |
| **B2** | B, warm-started and trained further | 148.6M | 896 | +971.7M tokens (about 1.32B total); val loss 2.547 → 2.239. **Main model for every later phase** |
| **C** | Encoder–decoder (10 + 10 layers, GQA, RoPE) | 202.7M | 640 source / 256 target | Phases 2–5 use the encoder only |
| **D** | Prefix-LM (bidirectional prefix, causal continuation), 20 layers | 162.3M | 896 (prefix 640) | |
| **B3** | B2 with the context extended 896 → 1024 | 148.6M | 1024 | +1.65B HC/Tribunal tokens. Loss on held-out text: HC 2.29→1.74, Tribunal 2.69→1.89, but SC 2.19→2.65 (worse) |

**Phase-1 Codabench test BERTScore:**

| Model + decoding | Score |
|---|---|
| A (mild sampling) | 0.8216 |
| C (mild sampling) | 0.8248 |
| D (mild sampling) | 0.8281 |
| B (mild sampling) | 0.8317 |
| B (greedy + 4-gram blocking) | 0.8338 |
| **B2, MBR (8 samples, t0.3, 3-gram blocking, chrF utility)** | **0.8422**|

- Decoding tricks tried: greedy, temperature/top-p sampling, n-gram blocking, beam search, minimum Bayes risk (MBR) decoding.


---

## 2. Phase 2: Legal NER (L-NER)

**Task:** predict character spans and labels in legal documents.
- Data: 70 training documents (8,393 spans), 35 test documents.
- Metric: strict span macro-F1.
- Technique menu: full FT, partial FT, prefix tuning, prompt tuning, LoRA, LoRMA, adapters, BitFit, KD.

**Setup:**
- Token classification over ByteBPE-aligned tokens, decoded back to character spans.
- Heads: linear, MLP or MLP+CRF.
- Class weights: inverse square root of class frequency.

**Techniques used:**
- full FT (B, C)
- LoRMA (B, C)
- LoRA (A)
- partial FT with k=2 plus a CRF head (B)
- BitFit (D)
- KD: teacher B (148.6M) → student = first 9 blocks of B (86.6M)

| Run | Test strict macro-F1 |
|---|---|
| B full FT (round 1) | 0.4494 |
| B full FT, lr 5e-5 | 0.4918 |
| B full FT, lr 5e-5, batch 8 (best single run) | 0.5033 |
| B LoRMA | 0.2066 |
| D BitFit | 0.1423 |
| A LoRA | 0.1350 |
| C full FT / C LoRMA | 0.0936 / 0.0556 |
| KD B → C (two attempts) | 0.0000 (predicted no entities, all O) |
| KD B → B with 9 blocks | 0.3745 |
| 5-run vote ensemble (at least 3 votes) + snapping spans to word boundaries | 0.5288 |
| Same ensemble + per-label vote thresholds (S6) | 0.5308 |
| **Final best** | **0.5313** |

---

## 3. Phase 3: Rhetorical Role prediction (RR)

**Task:** give each sentence one of 13 rhetorical-role labels.
- Data: 80 training documents (16,845 sentences); dev and test sets cover the CL and IT domains.
- Metric: macro-F1.
- Technique menu: LoRA, QLoRA, AdaLoRA, rsLoRA, LoRMA, prefix tuning, prompt tuning, IA³, KD. Full FT, partial FT, BitFit and adapters are **not** on this phase's list.

**Setup:**
- Each target sentence is encoded together with neighbouring context: 64 tokens left, 128 for the target, 64 right.
- Mean pooling, then an MLP head.
- Inverse-square-root class weights, 8 epochs.
- Three campaigns plus ensembles: 20 runs, then 7 more, then the 27 runs retrained without early stopping.

| Run | Test macro-F1 |
|---|---|
| B2 LoRMA r8, lr 2e-4 (best of campaign 1) | 0.4559 |
| B2 LoRA r8, lr 2e-4 | 0.4522 |
| B2 rsLoRA r16 | 0.4517 |
| KD: B2 LoRA teacher → 12-layer B2 LoRA student | 0.4501 |
| B2 QLoRA r8 (4-bit NF4 quantisation) | 0.4055 |
| B2 prompt tuning (16 tokens) / prefix tuning | 0.3816 / 0.3793 |
| D LoRA | 0.3801 |
| B2 AdaLoRA | 0.3636 |
| B2 IA³ | 0.3623 |
| A LoRA | 0.3502 |
| C (encoder) LoRA | 0.3108 |
| B2 LoRA, wider context (128 tokens each side), no early stopping | 0.4616 |
| **Anchored hard-vote ensemble (H_anchor)** | **0.4659** (final best) |

---

## 4. Phase 4: Court Judgment Prediction (CJPE)

**Task:** binary prediction (accepted = 1 or rejected = 0) from long Supreme Court case texts.
- Data: 37,387 training documents (31,653 after de-duplication); dev 3,505; test 1,517.
- Documents are long: median about 2.7k–3.5k tokens.
- Metric: macro-F1.
- Technique menu: full FT, partial FT, layer freezing, BitFit, LoRMA, AdaLoRA, adapters, KD.
- Caveat: all 1,517 test documents are also in the original dev set, with labels. So we score three dev subsets:
  - dev_all;
  - dev_excl_test (the honest selection set);
  - dev_in_test (equal to the test score).

**Setup (document classifier):**
- Split each document into up to k chunks, either evenly spaced, head + tail, or overlapping windows (stride 640).
- Encode each chunk with the backbone and mean-pool it.
- Combine chunks with attention pooling, then an MLP head.
- Loss: binary cross-entropy, 1 epoch.
- Then: threshold tuning, probability-averaging ensembles, long runs, and re-chunking at inference time only.

| Run | Test macro-F1 (threshold 0.5) |
|---|---|
| C3: B2 full FT, overlapping 896-token windows | 0.7574 |
| W1: B2 full FT, class-weighted | 0.7355 |
| E8: KD student (B2 with 12 of 18 blocks, 107M) | 0.7341 |
| C1: B2 full FT, head + tail | 0.7291 |
| E1: B2 full FT, evenly spaced chunks | 0.7238 |
| E7: B2 AdaLoRA | 0.5402 |
| E5: D BitFit | 0.5027 |
| E3: A partial FT | 0.4690 |
| E4: C adapters | 0.3935 |
| E6: B2 LoRMA | 0.3641 |
| E2: B3 full FT | 0.3323 (predicts one class only) |

| Improvements on top | Test macro-F1 |
|---|---|
| C3 with tuned threshold (0.30) | 0.7854 |
| Ensemble of C3 (0.75) and E8 (0.25), each model centred at its own tuned threshold | 0.7887 |
| Long runs: C3 for 5 epochs / E8 for 8 epochs | 0.7633 / 0.7577 |
| Re-chunked ensemble, selected on dev_excl_test | 0.8013 |
| **Track B ENS05** (tuned on the test documents, overlap-informed; selected on leaderboard) | **0.8082** |

---

## 5. Phase 5: Legal Statute Identification (LSI) (deadline 2026-10-26)

**Task:** multi-label prediction of one or more statute IDs per document.
- Data: 42,750 training documents, 10,181 dev, 13,019 test.
- 100 statute labels; on average 3.7 labels per training document.
- Metric: macro-F1 over the labels (secondary metrics: micro-F1 and subset accuracy).
- Rare statutes count as much as frequent ones.
- The spec asks to look at threshold strategies and rare labels.
- Technique menu: full FT, partial FT, QLoRA, LoRMA, AdaLoRA, adapters, prefix tuning, prompt tuning, KD.

**Setup:**
- Same chunked classifier as Phase 4, using overlapping windows (8 chunks, stride 640), mean and attention pooling, and an MLP head.
- Output: 100 independent sigmoid scores.
- Loss: BCE with `pos_weight` set to the square root of inverse label frequency, capped at 5.
- KD treats each statute as its own 2-class problem.
- Training: 2 epochs, with leakage guards on dev/test overlap.
- Checkpoint selection on dev_excl_test.
- The runs were split across two Kaggle accounts.

| Run | Model | Technique | Dev macro-F1 (threshold 0.5) |
|---|---|---|---|
| L1 | B2 | full FT, lr 2e-5 | 0.3816 |
| L2 | A | partial FT (top 6 blocks) | 0.2275 |
| L3 | B3 | QLoRA r16 | 0.3624 |
| L4 | B2 | LoRMA r16 | *account 1* |
| L5 | C (encoder) | adapters, bottleneck 64 | 0.1294 |
| L6 | D | prefix tuning, 32 prefix tokens | *account 1* |
| L7 | B2, 12 blocks | KD from L1 | *account 1* |
| L8 | B2 | AdaLoRA r16→8 | 0.2604 |


---

## 6. Phase 6: Abstractive summarization (deadline 2026-11-03)

**Task:** generate an abstractive summary of a legal judgment.
- Data: 7,030 training documents (summaries included), 100 test documents. No dev split, so 500 training documents (seed 42) are held out for validation.
- Metric: BERTScore F1 (secondary metrics: ROUGE-1, ROUGE-L).
- Documents are long: median 3,706 ByteBPE tokens. Summaries are also long: median 758 tokens, about 21% of the document's words.

**Setup:**
- Source: head+tail selection of 640 tokens (half head, half tail). Target: the reference truncated to 256 tokens at a sentence boundary, followed by `</s>`.
- Phase-1 model with its LM head, trained with token cross-entropy on target tokens only.
- Training: batch 8, cosine schedule, warmup 6%, eval every 250 steps, patience 3, checkpoint selected on validation loss.
- Decoding: length = 0.25 × document words, clamped to 20–180 words; EOS allowed; temperature 0.3; no-repeat 3-grams.
- Stage 7 sweep: each run was decoded again with only `max_words` changed (A/B2 210, C 180, D 225), giving two Codabench submissions per run.

| Run | Model | Technique | Epochs | Codabench (original, max 180 words) | Codabench (sweep) |
|---|---|---|---|---|---|
| R1 | B2 | LoRMA r8, attention + FFN | 10 | 0.8344 | 0.8348 (maxw210) |
| R2 | A | LoRA r8, attention | 10 | 0.8189 | 0.8182 (maxw210) |
| R3 | B2 | QLoRA | 10 | 0.8366 | 0.8349 (maxw210) |
| R4 | C | Houlsby adapters (src 384) | 12 | 0.8251 | 0.8236 (maxw180) |
| R5 | D | prefix tuning (src 624) | 10 | 0.8291 | 0.8279 (maxw225) |
| R6 | B2 | full FT | 8 | 0.8374 | **0.8389** (maxw210) |
| R7 | B2, 12 blocks | KD from R1 (T 2, α 0.5), full FT | 10 | 0.8294 | 0.8295 (maxw210) |

- Leaderboard entry: `submission_R6_B2_FullFT__maxw210.zip`, BERTScore 0.8389.
- R4's two submissions used identical decode settings (C's sweep cap = 180 = default); the scores differ by 0.0015 because of sampling at temperature 0.3.
