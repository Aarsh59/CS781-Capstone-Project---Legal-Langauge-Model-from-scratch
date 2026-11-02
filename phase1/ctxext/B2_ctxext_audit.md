# B2 context-extension audit (pre-decision, read-only)

Evidence labels: **[V]** verified from an artifact on this machine · **[E]** estimate (method stated) ·
**[U]** unknown with the artifacts here (what is needed is stated) · **[M]** from my notes of earlier sessions
(not re-verified; may refer to an earlier data build).

Nothing was trained, rebuilt or modified. The only compute: CPU forward passes of B2, tokenizer encoding of small text
samples, and reading checkpoints.

**Artifacts used**
- **Checkpoints and data:** `~/capstone/phase1/ab_inputs/{run_A,run_B,run_B2}/best.pt`, `ab_inputs/legal_bpe.json`, `ab_inputs/val.bin`.
- **Data recipe:** `phase1/notebooks/capstone_phase1_arch12.ipynb`, cells [4] (configuration), [17] (sampling helpers), [18] (tokenizer) and [19] (data build).
- **Training script:** `phase1/notebooks/capstone_phase1_refine.ipynb`, cell [10] (`train_ddp.py`).
- **Tokenizer source:** `phase1/genlocal/a1_tokenizer.py`.
- **Context-extension code:** `phase1/ctxext/train_ctxext.py`, `capstone_b2_ctxext.ipynb`.
- **Fertility samples:** small real-text samples (the first records of each corpus file).

**Not on this machine:**
- the Phase-1 data build's `manifest.json`;
- the B and B2 training logs (`log.jsonl`);
- the corpus files themselves. The Hugging Face cache holds only a reference.

See §10 for how to fetch them.

---
## 1. B2 architecture and checkpoint

| Item | Value | |
|---|---|---|
| Checkpoint | `~/capstone/phase1/ab_inputs/run_B2/best.pt` (594 MB) | [V] |
| Module | `arch3` (your attention3 GQA + rope + decoder3) | [V] |
| Parameters | 148,571,904 (LM head tied to the token embedding) | [V] |
| Layers / d_model / heads / KV heads / d_ff | 18 / 768 / 12 / 4 (GQA) / 3456 | [V] |
| Vocabulary | 32,000 (32 specials, ids 0–31; `</s>` = 3) | [V] |
| Normalisation | pre-norm (`norm_first: True`), LayerNorm | [V] |
| Positions | RoPE, split-half convention, base 10000; tables are **non-persistent buffers** (not in the checkpoint) | [V] |
| max_seq_len | 896; attention scale 0.125; dropout 0.1 | [V] |
| Checkpoint contents | `model, config, step, tokens, train_seconds, best_val, warmup_end_progress, world_size`. **No optimizer, scaler or scheduler state.** | [V] |
| Initialised from | `run_B/best.pt` (`config.init_from`), weights only: fresh optimiser, warm-up and schedule (refine `train_ddp.py` warm-start path) | [V] |
| Architecture vs predecessor B | identical (same `arch3` config) | [V] |
| Tokenizer | your `ByteBPE` (`a1_tokenizer.py`), `legal_bpe.json` (`tokenizer_path` in the config) | [V] |

## 2. Training history and token budget

| Stage | Run / checkpoint | Init | Data | Steps (at best.pt) | Tokens per step | Tokens processed (at best.pt) | Schedule | Best val loss |
|---|---|---|---|---|---|---|---|---|
| B | `run_B/best.pt` | scratch (init_std 0.02) | `pretrain_data_bpe/train.bin` (data notebook `aarsh59/notebook08c3659751`) | 43,006 | 9 seq × 896 = 8,064 (batch_tokens 8192) | **346,800,384** [V] | AdamW (0.9, 0.95), lr 3e-4 → min 3e-5, warm-up 1000, cosine, wd 0.1, clip 1.0, cap 350M tokens; 1×T4 | 2.5469 [V] |
| B2 | `run_B2/best.pt` | B best.pt (weights only) | same `train.bin` | 60,247 | 18 seq × 896 = 16,128 (batch_tokens 16384) | **971,663,616** [V] | same optimiser and values, wall-clock cosine to the session deadline (`max_tokens` None); 2×T4 DDP | 2.2394 [V] |

- **Cumulative exposure of B2's weights:** 346.8M + 971.7M = **1.318B tokens processed** [V]. That's about 8.9 tokens per parameter.
- **Tokens past best.pt:** B ran to about 350M tokens [M: 43,403 steps]. Steps after best.pt aren't in B2's lineage, because B2 started from B's best.pt.
- **Unique data vs exposure:**
  - `train.bin` is about 1.0B tokens in total [M: `TOTAL_TRAIN_TOKENS` = 1e9 in cell [4] [V]].
  - It already contains **repeated** Supreme Court documents (up to 4 epochs, capped at 70% share) and Acts (up to 2 epochs, 8%).
  - Training then drew **random windows** over that file, about 1.32 passes in total.
  - So unique Supreme Court tokens were seen about 2.8 × 1.32 ≈ **3.7 times** [E, using the earlier build's 2.78 SC epochs, M].
  - High Court / Tribunal documents in `train.bin` were seen about **1.3 times**, because each was used at most once in the file (cell [19]).
- **The two best-val numbers are not strictly comparable** [V, from the configs]: B used `eval_batches` 40, while B2 used `eval_windows` 360 fixed windows. Both are on the same `val.bin`.
- **Curves and overfitting checks need the training logs** [U]: `log.jsonl` of `run_B` and `run_B2`.

## 3. Corpus and source distribution

**Recipe** (verified from cells [4] and [19]): four sources, `train_{supreme_court,acts,high_courts,tribunals}.json` (HF `Exploration-Lab/CS-781-Capstone`).
- **Cleaning:** whitespace-normalised. Documents under 100 words are dropped. High Court / Tribunal / Acts documents with fertility above 2.0 tok/word are dropped.
- **Supreme Court:** every document. A 2% `md5(first 2000 chars)` hash split goes to **validation**; the rest is repeated up to 4 epochs, capped at 70% of 1B.
- **Acts:** up to 2 epochs, capped at 8%.
- **High Courts + Tribunals:**
  - **a random-seek sample** of 4× the remaining budget (seed 45) is encoded;
  - each document gets a DSIR-style score: mean log-probability under the SC unigram profile minus under its own source's profile;
  - **the most SC-like are kept until the remainder is filled**, each used once.
- **Assembly:** documents are shuffled into `train.bin` with `</s>` between them.

| Source | Docs raw / clean | Selected for train.bin | Tokens in train.bin | Corpus tokens | Unused |
|---|---|---|---|---|---|
| Supreme Court | [U] | all except 2% val | ~70% ≈ 0.70B with repeats [M] | ≈ 0.25B unique [E = 0.70B / 2.78] | ≈ 0 (2% held out for val) |
| Acts | [U] | all (≤ 2 epochs) | ~8% ≈ 0.08B [M] | ≥ 0.04B [E] | ≈ 0 |
| High Courts | [U] | top-scored subset of a sample | ~16.6% ≈ 0.17B [M] | [U] | **most of it** |
| Tribunals | [U] | top-scored subset of a sample | ~5.4% ≈ 0.05B [M] | [U] | **most of it** |
| **All** | — | — | 1.0B | **≈ 3.8B cleaned** [M, earlier HF-tokenizer build] | **≈ 2.5–3.3B, almost all HC/Tribunal** [E] |

- **The [M] shares and totals** come from my notes of the earlier HF-tokenizer data build (2026-09-28). The ByteBPE rebuild that B and B2 used followed the same recipe, but its exact figures are in its `manifest.json` [U]. That file records per source `docs_seen`, `docs_kept`, `dropped_short`, `dropped_fertility`, `unique_tokens`, `tokens_used` and `share`.
- **The unused-data estimate:** HC+Tribunal text is about 17 GB [V, cell [19] comment] ÷ ~4.8 chars/token (§5) ≈ **3.5B tokens**, minus about 0.22B used ≈ **3.3B unused** [E]. About 0.66B of those were scanned and rejected as less SC-like [E = 4× 0.22B − 0.22B].
- **Overlap:** `train.bin` stores **no document IDs**, and the scratch offsets were deleted after the build [V, cell [19]]. So overlap can't be read from artifacts.
  - **The reliable way to rebuild it:** replay the deterministic selection (fixed seeds 42–45, same sampler and scoring code) on the corpus to recover exactly which HC/Tribunal documents were picked.
  - **For near-duplicates:** compare `doc_hash` (md5 of the first 2000 normalised chars), plus MinHash or n-gram overlap against all SC documents. HC/Tribunal judgments often quote SC judgments.

## 4. train.bin / val.bin and genuinely new data

| File | Tokens | dtype | Structure | |
|---|---|---|---|---|
| train.bin | ≈ 1.0B (`manifest.train_tokens`) | uint16 | shuffled whole documents separated by `</s>` (id 3). No other specials, no padding. Repeats are physical copies (SC ≤ 4×, Acts ≤ 2×). | recipe [V], count [U] |
| val.bin | **5,230,665** | uint16 | **1,083** SC documents separated by `</s>`, max id 31999 | [V] |

- **No other token files were used** by B or B2 [V: `data_dir` and the configs].

| Category | Supreme Court | Acts | High Courts | Tribunals |
|---|---|---|---|---|
| 1. Already used in training | 98% of documents, about 3.7× exposure | all, about 2.6× | selected subset, about 1.3× | selected subset, about 1.3× |
| 2. Seen, could give more windows | yes (adds little: already repeated) | yes | yes (their selected subset) | yes |
| 3. Genuinely unseen | the 2% val slice only, which must stay held out | none | **most documents (≈ 3B tokens together with Tribunals) [E]** | **most** |
| 4. Must not enter training | `val.bin` / `val_prompts` (SC hash slice); Phase-1 `test.json` prompts; **all dev/test documents of Phases 2–7** (see the note below) | | | |

**Contamination risk for later phases (important):** Phases 2–7 use legal documents from the same court systems: NER, RR, judgment prediction, statutes, summaries, retrieval. Continued pretraining on unused High Court / Tribunal text could include documents that appear in a later phase's **test set**. Before training, deduplicate the new data against every phase's dev and test documents, by hash plus n-gram overlap.

## 5. Tokenizer and fertility

| Property | Value | |
|---|---|---|
| Implementation | your byte-level BPE with end-of-word marker (`ByteBPE`, `a1_tokenizer.py`); merge-rank encoding | [V] |
| Vocabulary | 32,000 = 32 specials + 256 bytes + merges | [V] |
| Pre-tokenisation | inside each whitespace word: `[^\W\d_]+ \| \d{1,3} \| [^\w\s]+ \| _+`, so letter runs, **digit groups of at most 3**, and punctuation runs are separate pieces | [V] |
| Normalisation | whitespace collapse (`" ".join(text.split())`); the data build normalises the same way | [V] |
| Specials | `<pad> <unk> <s> </s> <cls> <sep> <mask>` + `<extra_id_0..24>`; `</s>` (3) separates documents | [V] |
| Unknown tokens | none possible (byte-level); **0** `<unk>` observed | [V] |
| Training | cell [18]: 200M characters, mix SC 40 / Acts 30 / HC 15 / Tribunals 15 | [V] |
| New data compatible | yes: any text encodes with the same ids, so retraining isn't needed | [V] |

**Fertility** [V on these samples; **the samples are small and come from the start of each file**, so they're indicative, not representative]:

| Source | Docs | Characters | Words | Tokens | Tokens / word | Characters / token |
|---|---|---|---|---|---|---|
| Supreme Court (val.bin, first 2M tokens) | 447 | 9.51M | 1.62M | 2.00M | **1.235** | 4.76 |
| Supreme Court (sample) | 40 | 1.20M | 201.6k | 253.3k | 1.257 | 4.74 |
| High Courts (sample) | 14 | 0.45M | 74.9k | 93.1k | 1.242 | 4.78 |
| Tribunals (sample) | 9 | 0.40M | 67.6k | 81.9k | 1.211 | 4.87 |
| Acts (sample) | 10 | 0.24M | 40.0k | 52.0k | 1.298 | 4.56 |

- **Token-length distributions** are nearly identical across sources: about 18% 1-byte, 20% 2-byte and 20% 3-byte tokens.
- **No source has substantially higher fertility.** Acts are about 5% higher.
- **Implication:** raw text converts to tokens at about 4.6–4.9 characters per token for every source. So the HC/Tribunal volume estimate in §3 doesn't depend much on the mix.

**Fragmentation examples** [V]:
- **Whole tokens:** common legal words and phrases are mostly single tokens: `Section`, `Constitution`, `res judicata`, `locus standi`, `suo motu`, `mutatis mutandis`, `anticipatory bail`.
- **Fragmented:**
  - **4-digit years always split 3+1** (`197`+`8`, `201`+`5`), a direct result of the `\d{1,3}` rule.
  - **Dotted abbreviations split per letter:** `S.L.P. (C) No. 12345 of 2019` is 16 tokens; `W.P.(C) 4567/2018` is 12.
  - `Hon'ble` is 3 tokens and `vis-a-vis` is 5.
- **This matters at 1024 tokens** for citation-dense passages, but it is fixed: changing it would mean a new tokenizer.

## 6. Validation and model quality
- **Validation set:** `val.bin` = **Supreme Court only**: the 2% hash-split documents, 1,083 documents, 5.23M tokens [V].
- **Not available here:** per-source validation losses, curves and long-context results all **don't exist** in local artifacts. Curves need the logs [U].
- **Overlap:** the hash split happens before SC repetition, so val documents never enter `train.bin` [V, cell [19]].
- **Earlier measurements** [M]:
  - **Position-matched loss** (positions 640–896): B 2.436.
  - **Phase-1 test fertility:** HF tokenizer 1.2006 tok/word; ByteBPE value in `manifest.test_fertility` [U].
- **A quick probe on the real B2** [V; 2 windows of 1025 tokens from val.bin, so very noisy]:

| | Loss, positions < 896 | Loss, positions ≥ 896 |
|---|---|---|
| B2 alone, 896-token window | 2.101 | — |
| Extrapolate (1024 window) | 2.101 | 2.087 |
| Interpolate (1024 window) | 2.102 | 2.031 |
| NTK (1024 window) | 2.101 | 2.035 |

  Before any training, B2 isn't broken past 896. Two windows can't rank the three methods.
- **A reliable held-out set for continued pretraining** (suggested construction):
  - keep `val.bin` (SC);
  - add **hash-split held-out slices of the unused High Court and Tribunal documents**, carved off before any new training file is written;
  - **Acts can't give a clean held-out set,** because every Act was trained on. Either report Acts in-sample or skip them;
  - report each source separately, for positions 1–896 and 897–1024.
  - Keep Phase-1 `test.json` and every Phase 2–7 dataset out.

## 7. Context-extension code review (`train_ctxext.py`)

| Check | Status |
|---|---|
| Loads B2 into the model rebuilt at 1024 | **[V on the real B2]:** no missing or unexpected keys, no parameter changes shape. Only the non-persistent RoPE/mask buffers are rebuilt. |
| RoPE methods | extrapolate: positions 0..1023, base 10000. Interpolate: positions × 896/1024. NTK: base × (1024/896)^(64/62). [V code] |
| Extrapolate preserves B2 on 896 positions | **[V real B2]:** max logit difference 4.8e-05, identical loss (2.1009) |
| Phase-2/3 compatibility | [V tiny]: `best.pt` loads in `phase3/backbones.py` with `max_len` = the new length. **The Phase-3 loader rebuilds RoPE with arch3's default formula, i.e. extrapolate.** An interpolate/NTK-trained model would be loaded with the wrong positions unless `backbones.py` applies the same method. **A fix would be needed before using it downstream.** |
| Resume: model, optimiser, scaler, step, tokens, best value | saved and restored [V tiny] |
| Data-sampling state | stateless: windows come from RNG(seed, rank, step, micro), so resume continues identically [V tiny] |
| Learning-rate schedule across sessions | stateless function of step, with `total_steps` = `max_tokens` / tokens-per-step [V]. **Only consistent if the GPU count and `batch_tokens` stay the same in every session.** There's no guard yet; a 1-GPU session would change tokens-per-step and the schedule. |
| torch / CUDA RNG (dropout) | **not saved.** The bit-identical resume test used dropout 0. With B2's dropout of 0.1, resumed runs aren't bit-identical. That's statistically harmless, but the "bit-identical" claim only holds without dropout. |
| Checkpoint selection | `best.pt` uses `val_loss_new_len` only. Old-length loss is **logged but not part of the selection**, so retention of 896-token ability isn't protected automatically. |
| Micro-batch search | done before the optimiser and DDP buffers exist; it keeps peak memory under 80% of the GPU. Adam state (~1.2 GB) plus DDP buckets (~0.6 GB) come later. **Unverified on a real T4.** The probe cell measures it. |
| Only tested on tiny CPU models | DDP on 2×T4 (NCCL), fp16 + GradScaler at 1024, T4 memory and throughput, the real-data loader, and evaluation cost at 200 windows |

## 8. Compute budget
- **Hardware:** Kaggle 2×T4 (16 GB each), DDP, fp16 autocast + GradScaler, PyTorch SDPA attention (fastattn).
- **Measured throughput** at 896 tokens:

| Setup | Throughput | Source |
|---|---|---|
| B, 1×T4 | 346.8M / 27,227 s ≈ **12.7k tokens/s** | [V, checkpoint metadata] |
| B2, 2×T4 DDP | 971.7M / 40,651 s ≈ **23.9k tokens/s** | [V] |

  Whether `train_seconds` excludes evaluation time [U].
- **At 1024 tokens** [E]: about 5–10% slower (attention is a small part of FLOPs at this size), so about **21.5–23k tokens/s**. Tokens per step at batch 16384 = 16 sequences × 1024.
- **Memory:** about 9.2 GB peak at 896 on 1×T4 with micro-batch 9 [M, Phase-1 probe]. At 1024 a micro-batch of about 8 should fit [E]. The probe cell gives the real number.

| Additional tokens | Steps (16,384 tokens/step) | Hours at ~22k tokens/s, +5% evaluation/checkpoint | 11-hour sessions |
|---|---|---|---|
| 0.5B | 30.5k | ≈ 6.6 h | 1 |
| 1.0B | 61.0k | ≈ 13.3 h | 2 |
| 1.5B | 91.6k | ≈ 19.9 h | 2 |
| 2.0B | 122.1k | ≈ 26.5 h | 3 |

## 9. Executive summary

**Facts:**
1. **Cumulative exposure: 1.318B tokens** [V], 8.9 tokens per parameter.
2. **Unique data seen:** `train.bin` ≈ 1.0B tokens including repeats.
   - **Supreme Court:** about 0.25B unique tokens [E], seen about 3.7 times.
   - **High Court / Tribunal:** about 0.22B tokens, seen about 1.3 times.
   - **Acts:** all, about 2.6 times.
   - Exact document counts need `manifest.json` [U].
3. **Genuinely unseen data:** about **2.5–3.3B tokens, almost entirely High Court and Tribunal** [E]. SC and Acts have essentially nothing unseen beyond the SC validation slice.
4. **Fertility:** about 1.21–1.30 tokens per word, 4.6–4.9 characters per token, similar for all sources [V, small samples].
5. **Budget the remaining corpus can support with new data:** up to about 2.5–3B tokens of unseen HC/Tribunal text, after deduplication [E].
6. **Would more pretraining help?**
   - **Evidence for:** loss fell from 2.547 (B) to 2.239 (B2) with about 1B more tokens. That's suggestive but not strictly comparable, because the evaluation procedures differed.
   - **Still missing:** training curves, a check of whether B2's loss was still falling at the end, and per-source and long-context evaluation.

**Your decisions (rule 9). Constraints from the facts above, not values:**
7. **Mixture:**
   - genuinely new data is almost only HC/Tribunal;
   - the validation set is SC-only, so it won't show changes on HC/Tribunal text;
   - SC/Acts can only be revisited as repeats (SC is already at ~3.7×).
   - A per-source held-out set (§6) is needed to see the effect of whatever mixture you choose.
8. **Token target:** bounded by the unused data (≈ 3B) and by about 0.9B tokens per 11-hour session (§8).
9. **Hyperparameters to choose:** learning rate and minimum, warm-up (re-warm from a converged checkpoint), schedule, batch tokens, weight decay, clip, RoPE method.
   - B and B2 used AdamW (0.9, 0.95), lr 3e-4 → 3e-5, warm-up 1000, cosine, wd 0.1, clip 1.0, batch about 16k tokens.
   - **If you choose interpolate or NTK, `phase3/backbones.py` must apply the same RoPE change** (§7).
10. **Suggested criteria for "better than B2":**
   - per-source held-out loss at positions 1–896 should be **no worse** than B2's, to keep existing ability;
   - loss at positions 897–1024 should improve;
   - check SC `val.bin` loss at 896 against B2's 2.239, using the **same** evaluation procedure;
   - ideally, a small downstream check (e.g. Phase-3 dev macro-F1 with one fixed config).

## 10. Missing information and how to get it

| Missing | Where | How (you run it; token stays local) |
|---|---|---|
| `manifest.json`: per-source document and token counts, test fertility | data notebook `aarsh59/notebook08c3659751` output, `cs781_phase1/pretrain_data_bpe/` | command 1 below |
| B training curve | same notebook, `cs781_phase1/run_B/log.jsonl` | command 1 below |
| B2 training curve | your refine notebook's output, `run_B2/log.jsonl` | command 1 below, with that notebook's slug |
| Raw corpus sizes and counts, near-duplicate overlap, representative fertility | HF corpus (about 19 GB) | needs downloading the corpus, ideally inside a CPU-only Kaggle session (the planned data step) |

Command 1 (downloads only `manifest.json` and the `log.jsonl` files, not the token files or checkpoints):

```fish
~/.cache/kaggle_cli_venv314/bin/kaggle kernels output aarsh59/notebook08c3659751 -p ~/kaggle_outputs/p1data --file-pattern '(manifest\.json|log\.jsonl)$'
```
