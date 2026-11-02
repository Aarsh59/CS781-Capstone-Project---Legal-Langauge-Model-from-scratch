# Phase 3: advanced offline analysis (existing artifacts only)

Scripts: `common.py`, `diagnostics.py` (measurements, no tuned values), `optimize.py` (methods 1–6, leave-one-document-out).
Inputs: `../bundle`, `../experiment_inventory.csv`, read-only. No training, inference, downloads or submissions. No test labels exist.

**Status**
- Part A (measurements) and Part B (your decision-strategy grid, `run_user_grid.py`) are complete.
- The deliverable names `method_comparison.csv`, `loo_per_document.csv`, `loo_per_class.csv` and
  `selected_configurations_by_fold.json` are copies of the `ug_*` outputs of your grid.
- The generic runner `optimize.py` / `advanced_config.json` was superseded by your specification and was not used.

## Part A: measurements

### A1. Error complementarity vs the anchor (`candidate_complementarity*.csv`)
| Candidate | Fixes | Breaks | Net | CL net | IT net | Docs net-positive |
|---|---|---|---|---|---|---|
| G2 | — | — | **+45** | +26 | +19 | 5 |
| ENS_candidate | — | — | +34 | — | — | 5 |
| H | — | — | +19 | +15 | +4 | 4 |
| **B_lora_ctx_wide** (campaign-2, early-stopped twin of the anchor) | 264 | 250 | **+14** | 0 | +14 | 6 (largest single doc = 37% of the positive net) |
| C1 | — | — | +9 | −2 | +11 | 6 |
| best other single run (B_lorma_allep) | — | — | −27 | | | |

- The twin is the **only single run** that is net-positive against the anchor. Earlier anchored studies used only the 27
  `_allep` runs, so it never appeared in them. It is entirely an IT gain: CL net 0.
- G2 and H are net-positive on **CL sentences** (+26, +15), yet their CL macro-F1 is lower than the anchor's. The next
  item shows why: sentence counts and macro-F1 can move in opposite directions.

### A2. CCI Noida case study (`cci_noida_confusions.csv`, `cci_noida_per_class.csv`, `cl_documents_compared.csv`)
- 265 sentences. Gold labels present: Fact 27, ArgumentPetitioner 36, ArgumentRespondent 81, RatioOfTheDecision 111,
  RulingByPresentCourt 8, **None 2**. The anchor and H also predict PrecedentReliedUpon and RulingByLowerCourt, which
  have **no gold** sentences here. Under the local metric these enter the document's average with F1 = 0, so it averages 8 labels.
- **Most of the document-level drop comes from the 2 `None` sentences.**
  - None F1: anchor 0.667 → H 0.000. One label out of 8 contributes −0.667 / 8 = **−0.083** of H's −0.094 on this document.
  - The rest: Fact −0.094, ArgumentPetitioner −0.022, Ratio −0.019, ArgumentRespondent **+0.048** (all ÷ 8).
- **Most frequent confusion:** RatioOfTheDecision → Fact (anchor 22 sentences, H 33). Next: ArgumentRespondent → ArgumentPetitioner (16 / 10).
- **The other CL documents** (anchor → H macro-F1): `SC_Competition…` (848 sentences) 0.313 → 0.309; `COM_Shree_Cement` (32)
  0.344 → 0.421; `HC_State_of_Mizoram` (184) 0.466 → 0.509; `CCI_Western_Coalfield` (67) 0.693 → 0.687.
- **Takeaway (factual):** per-document and per-domain macro-F1 swings of this size can come from a handful of sentences in a rare label.

### A3. Transitions (`transitions.csv`, `transition_errors.csv`)
- Gold label changes between adjacent sentences across the 10 documents: **249**. Predicted changes: anchor **643**, H 713, G2 699, C1 720.
  Predictions switch roles about **2.6–2.9× as often as gold**.
- Error rate at gold role boundaries (401 sentences) vs inside segments:

| | At boundaries | Inside segments |
|---|---|---|
| Anchor | 0.506 | 0.448 |
| H | 0.491 | 0.441 |
| G2 | 0.461 | 0.433 |
| C1 | 0.466 | 0.452 |

- **This is the evidence item 6 asked for** before testing transition-aware decoding (method M6 in `optimize.py`). Whether
  smoothing helps macro-F1 is an empirical question for your run. Smoothing can remove rare-label sentences as easily as noise.

### A4. Absent-class sensitivity (`metric_sensitivity.csv`)
| Method | gold∪pred (local) | gold only | all 13 labels (absent = 0) |
|---|---|---|---|
| anchor | 0.5109 | 0.5109 | 0.4323 |
| H | 0.5251 | 0.5251 | 0.4443 |
| G2 | 0.5320 | 0.5320 | 0.4502 |
| C1 | 0.5177 | 0.5177 | 0.4380 |
| ENS_candidate | 0.5274 | 0.5274 | 0.4463 |

- On the pooled dev set, none of these methods predicts a label without gold, so the first two conventions agree. The all-13
  convention lowers every score by about 0.08 and **keeps the same ordering**. Codabench's convention remains **unknown**.

## Part B: your decision-strategy grid (`run_user_grid.py`, `user_hyperparameters.json`)

Your values are transcribed verbatim in `user_hyperparameters.json`.

**Interpretations, to confirm:**
- **Pool:** "its seed variant" read as `KD_allep`, `KD_seed123_allep`; "compatible runs from the audit" read as `B_lora_ctx_wide`.
- **S1 members** come from a fixed in-fold rule (pool ranked by net error correction on the training documents), so each
  fold stays inside your 30-configuration budget.
- **S2** used sequential narrowing: k × anchor weight at T = 1 (20 configurations), then T (3).
- **S5:** "ensemble only when the anchor is below the threshold" and "...and anchor and ensemble disagree" give identical
  labels, because a switch changes nothing where they agree. They are reported once.
- **S6:** local one-sentence correction (neighbours fixed at the anchor's predictions), plus the optional Viterbi variant.
- **Budget:** at most 20 configurations per strategy per fold (`ug_config_counts.csv`).

All held-out. Selection on each fold's 9 training documents (per-domain α: that domain's 4 training documents).
| Method | Pooled | vs anchor | vs H | CL (Δ anchor) | IT | Docs +/− | Largest doc share of gains | Same choice in folds |
|---|---|---|---|---|---|---|---|---|
| anchor | 0.5109 | — | −0.0142 | 0.4658 | 0.5333 | — | — | — |
| FIXED_H (not held-out) | 0.5251 | +0.0142 | — | 0.4459 (−0.020) | 0.5495 | 6/4 | 0.35 | — |
| FIXED_G2 (not held-out) | 0.5320 | +0.0211 | +0.0069 | 0.4509 (−0.015) | 0.5670 | 9/1 | 0.25 | — |
| S1 complementarity, hard | 0.5208 | +0.0099 | −0.0043 | 0.4470 (−0.019) | 0.5542 | 6/4 | 0.43 | 6 (anchor + 3, anchor weight 2 in 8 folds) |
| S2 soft, compatible group | 0.5236 | +0.0127 | −0.0015 | **0.3747 (−0.091)** | 0.5625 | 6/4 | 0.63 | 4 |
| S3 bias ArgumentPetitioner | 0.5123 | +0.0014 | −0.0128 | 0.4663 (+0.001) | 0.5365 | 5/4 | 0.68 | 10 (+0.6, the grid edge) |
| S3 bias ArgumentRespondent | 0.5090 | −0.0019 | | 0.4638 | 0.5291 | 3/3 | | 8 |
| S3 bias RulingByLowerCourt | 0.5089 | −0.0020 | | 0.4622 | 0.5347 | 1/4 | 1.00 | 9 |
| S3 bias RatioOfTheDecision | 0.5146 | +0.0037 | −0.0105 | 0.4723 (+0.007) | 0.5383 | 6/2 | 0.63 | 10 (**+0.15**, a boost, not a penalty) |
| S3 bias PrecedentReliedUpon | 0.5131 | +0.0022 | | 0.4655 | 0.5360 | 2/3 | 0.74 | 10 |
| S3 bias PrecedentNotReliedUpon / Statute | ≈0.510 | ≈ −0.001 | | ≈0.465 | | 0/2, 0/1 | | 9 |
| S3 combination (3 argument / lower-court biases) | 0.5102 | −0.0007 | | 0.4624 | 0.5363 | 4/5 | 0.65 | 7 |
| S4 always ensemble | 0.5236 | +0.0127 | −0.0015 | 0.3747 (−0.091) | 0.5625 | 6/4 | 0.63 | — |
| S4 α global | 0.5224 | +0.0115 | −0.0027 | 0.3747 (−0.091) | 0.5601 | 6/4 | 0.63 | 9 (α = 1.0) |
| **S4 α per domain** | **0.5300** | **+0.0191** | **+0.0049** | 0.4404 (−0.025) | 0.5625 | **8/2** | **0.64** | 5 (α_CL 0.25 or 0.5; α_IT 1.0 in all 5 IT folds) |
| S5 gate, max prob | 0.5230 | +0.0121 | −0.0021 | 0.3770 (−0.089) | 0.5652 | 6/4 | 0.61 | 9 |
| S5 gate, margin | 0.5187 | +0.0078 | −0.0064 | **0.4693 (+0.004)** | 0.5418 | 6/4 | 0.37 | 8 (0.20) |
| S5 gate, entropy | 0.5212 | +0.0103 | −0.0039 | 0.4477 (−0.018) | 0.5488 | 7/3 | 0.40 | 10 (0.4) |
| S6 transition, local | 0.5002 | −0.0107 | −0.0249 | 0.3966 (−0.069) | 0.5353 | 5/4 | 0.34 | 6 |
| S6 transition, Viterbi | 0.5230 | +0.0121 | −0.0021 | 0.4265 (−0.039) | **0.5749** | 7/3 | 0.32 | 8 (strength 0.5 = grid edge, smoothing 0.1) |

**Gate switches** (held-out, where the gate replaced a differing anchor label):

| Signal | Switches | Fixed | Broke |
|---|---|---|---|
| max prob | 296 | 120 | 69 |
| margin | 106 | 43 | 21 |
| entropy | 137 | 52 | 26 |

**Facts to note:**
- **Only one held-out method scores above H's pooled dev:** S4 α per domain, at **+0.0049**.
  - 64% of its document-level gains come from one document, `BHC_BomHC_2007_1278` (+0.136). The other 9 documents move between −0.007 and +0.025.
  - Its IT α is 1.0 in every IT fold. Your α_IT grid starts at 0.25, so IT can never fall back to the anchor alone.
  - Its CL α splits 3 : 2 between 0.25 and 0.5.
- **Cheap, CL-safe adjustments with perfectly stable choices** (Ratio +0.15 in 10/10 folds, ArgumentPetitioner +0.6 in 10/10)
  gain only **+0.001 to +0.004** over the anchor. Two choices sit at the edge of their grids (ArgumentPetitioner +0.6, Viterbi strength 0.5).
- **The margin gate** is the only ensemble-based method that **raises CL** (+0.004), with gains spread over documents (largest share 0.37).
  Its pooled gain is +0.008 over the anchor, which is below H.
- **Local one-sentence transition smoothing hurts** (−0.011). Viterbi helps IT (0.575, the highest IT) but costs CL (−0.039).

**Your selection rule, applied:**
1. **Improves pooled over the anchor:** many methods do.
2. **No severe CL regression:** rules out S2, S4-global, S5 max prob and S6.
3. **Helps multiple documents, not one outlier:** S4-per-domain fails this (64% from one document).
4. **Stable parameters:** S4-per-domain is only 5/10.

The methods that pass all four (S3 Ratio, S5 margin, S5 entropy, S1) all stay **below H's pooled dev score**.
That is **no robust improvement over the current best submission.**

## Uncertainty (applies to every Part B result)
With 10 documents, a document bootstrap of the anchor's dev score gives a 95% interval of about ±0.08. Seed-to-seed test
variation was about 0.01. Leave-one-document-out results show stability, not expected test scores.

## Recommendation
**Outcome: inconclusive → no robust improvement.**
- The only held-out strategy above H (S4 α per domain, +0.005 pooled dev) depends on one document for most of its gain and
  has unstable CL parameters. The gain is far below the ±0.08 uncertainty of a 10-document score.
- Every strategy that passes your stability and CL conditions scores below H.
- **H (0.4659) remains the most defensible submission.** Whether to spend a submission on S4 per domain anyway is your call.
  Its dev evidence is weaker than H's was.
