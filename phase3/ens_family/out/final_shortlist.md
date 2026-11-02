# Phase 3: family / anchored ensemble study: shortlist

All numbers are **development-set** results (10 documents: 5 CL, 5 IT, 2,142 sentences, 11 of 13 labels present).
**Codabench test performance of every new candidate is unknown** until it is submitted. The only known test score
is the anchor's: `B_lora_ctx_wide_allep` = **0.4616**.

Scripts: `ens_family/common.py`, `ens_family/step12_audit.py`, `ens_family/step3to9_ensembles.py` (rerun with
`python step12_audit.py && python step3to9_ensembles.py`). Inputs are read-only: `~/kaggle_outputs/allep/phase3/runs`.
Metric: `rr_metrics.rr_scores` (the pipeline's macro-F1). Label order: `label_vocab.json`.

## What was verified (Steps 1–2)
- All 27 runs: dev and test doc IDs, sentence counts and label order match the data. Probability files are aligned. `B_lora_crf_allep`
  probabilities are **pre-Viterbi** emissions, so they were excluded from all probability averaging.
- Reproduced exactly: anchor 0.5109 / CL 0.4658 / IT 0.5333; top-5 0.5192; top-7 0.5195; all-27 0.4952.
- Families (your definition, by technique type): F1 LoRA-type adapters on B (18), F2 prompt/prefix/IA3 (4), F3 KD
  students (2), F4 other backbones A/C/D (3). See `model_family_audit.csv`.
- Near-duplicates (≥0.99 agreement on dev **and** test): only `B_adalora_allep` ~ `B_adalora_r4_allep`
  (identical on all 2,142 dev sentences, 1 test sentence differs). Within-family majority voting counts them once per family.
- Anchor error analysis (`anchored_error_analysis.csv`): the anchor is wrong on 983 dev sentences (CL 675, IT 308).
  **Every one of the 26 other runs breaks more of the anchor's correct predictions than it fixes** (best net:
  `B_lorma_allep` −27, `KD_seed123_allep` −40). On its own, no run is a better "second opinion" than the anchor.

## Plain-language results (Steps 3–6; full table `ensemble_results.csv`, per-class `per_class_results.csv`)
| Method | Dev | CL | IT | vs anchor (dev / CL / IT) |
|---|---|---|---|---|
| Anchor alone | 0.5109 | 0.4658 | 0.5333 | — |
| All 27, uniform | 0.4952 | 0.3414 | 0.5446 | −0.016 / −0.124 / +0.011 |
| Top-7, uniform | 0.5195 | 0.4042 | 0.5586 | +0.009 / −0.062 / +0.025 |
| One rep per family (best dev), equal votes (C1) | 0.5177 | 0.4370 | 0.5505 | +0.007 / −0.029 / +0.017 |
| Within-family majority, equal family votes (D) | 0.4989 | 0.3439 | 0.5497 | −0.012 / −0.122 / +0.016 |
| Same, F1 family ×2 (E) | 0.4938 | 0.3441 | 0.5408 | −0.017 / −0.122 / +0.008 |
| Best 2 per family, uniform (F) | 0.5098 | 0.3578 | 0.5578 | −0.001 / −0.108 / +0.025 |
| Anchor + 1 model (any of 6) | 0.5109 | 0.4658 | 0.5333 | identical to the anchor |
| **Anchor + B_lorma_allep + KD_seed123_allep (G2)** | **0.5320** | 0.4509 | **0.5670** | **+0.021** / −0.015 / +0.034 |
| Anchor ×2 + top-4 net runs ×1 (H) | 0.5251 | 0.4459 | 0.5495 | +0.014 / −0.020 / +0.016 |
| Anchor + one majority vote per other family (I) | 0.5179 | 0.4379 | 0.5479 | +0.007 / −0.028 / +0.015 |
| Best probability average (anchor ×2 + 2 runs, J3) | 0.5198 | 0.4250 | 0.5560 | +0.009 / −0.041 / +0.023 |

What this means mechanically:
- **Equal family votes hurt (D, E, F).** Families F2 and F4 are weak (dev 0.36–0.44), so giving them a quarter of
  the vote each pulls CL down by about 0.11–0.12.
- **Anchor + 1 always equals the anchor.** In a 2-member vote every disagreement is a tie, and ties go to the anchor.
- **Anchor + 2** overrules the anchor only when **both** other models agree on the same different label. That is the
  smallest change that can still fix anchor errors, and it gives the best dev score here.
- **Anchor weight 1.0 and 1.5 give identical results** (H and I). Ties already go to the anchor, so 1.5 changes nothing.
  Weight 2.0 means the anchor is overruled only when at least 3 of the 4 others agree.
- **Probability averaging did not beat hard voting** (best 0.5198 < 0.5320). Calibration differs a lot between runs:
  ECE ranges from 0.058 to 0.272 (`summary.json`). Per your rule, no calibration work was pursued, and its zip was not kept.

## Shortlist (Step 8)
| # | Candidate | Members (weights) | Dev / CL / IT | Δ vs anchor | Docs better than anchor | LOO gain vs anchor (min…max) | Test sentences changed vs anchor | ZIP |
|---|---|---|---|---|---|---|---|---|
| 1 | Anchor (current best) | B_lora_ctx_wide_allep | 0.5109 / 0.4658 / 0.5333 | — | — | — | 0 | already submitted (0.4616) |
| 2 | Family-balanced C1 | ctx_wide_allep, KD_seed123_allep, D_run_allep, B_prompt_p16_allep (1 each, ties → anchor) | 0.5177 / 0.4370 / 0.5505 | +0.007 / −0.029 / +0.017 | 7/10 | −0.004…+0.013 (9/10 > 0) | 372 | `candidates/submission_C1_one_rep_per_family_bestdev.zip` |
| 3 | Anchor + 1 | — | identical to the anchor | 0 | 0/10 | 0 | 0 | none (would duplicate the anchor) |
| 4 | **Anchor + 2 (G2)** | ctx_wide_allep, B_lorma_allep, KD_seed123_allep (1 each, ties → anchor) | 0.5320 / 0.4509 / 0.5670 | +0.021 / −0.015 / +0.034 | 9/10 | +0.009…+0.028 (10/10 > 0) | 347 | `candidates/submission_G2_anchor+B_lorma_allep+KD_seed123_allep.zip` |
| 5 | Anchor-weighted H | anchor ×2; KD_seed123_allep, B_rslora_r16_allep, B_lorma_allep, B_rslora_r16_lr5e5_allep ×1 | 0.5251 / 0.4459 / 0.5495 | +0.014 / −0.020 / +0.016 | 6/10 | +0.002…+0.021 (10/10 > 0) | 342 | `candidates/submission_H_anchor_w2.0+top4net.zip` |

Per-class F1 change vs the anchor (dev):
| Label | Anchor | G2 | H | C1 |
|---|---|---|---|---|
| Fact | 0.637 | +0.022 | +0.005 | +0.025 |
| Issue | 0.690 | +0.096 | +0.096 | +0.069 |
| ArgumentPetitioner | 0.534 | −0.024 | −0.028 | −0.027 |
| ArgumentRespondent | 0.513 | +0.006 | −0.005 | −0.086 |
| Statute | 0.796 | −0.029 | −0.021 | −0.013 |
| PrecedentReliedUpon | 0.466 | +0.037 | +0.018 | +0.028 |
| PrecedentNotReliedUpon | 0.222 | +0.035 | +0.035 | +0.096 |
| RulingByLowerCourt | 0.420 | +0.012 | +0.009 | −0.066 |
| RatioOfTheDecision | 0.526 | +0.045 | +0.028 | +0.021 |
| RulingByPresentCourt | 0.449 | +0.036 | +0.016 | +0.036 |
| None | 0.367 | −0.003 | +0.004 | −0.010 |
(Issue has 15 dev sentences: +0.096 F1 corresponds to about one or two sentences.)

### Indicators for and against generalisation (facts, not predictions)
**Common to all new candidates:**
- (−) The members were chosen on the **same 10 dev documents** that score them. G2 is the best of 15
  predetermined pairs, so its +0.021 is optimistic. Leave-one-document-out results are a stability diagnostic only.
- (−) Dev has been optimistic before. Dev−test gaps ranged 0.02–0.08. The routing candidate gained +0.03 on dev and
  scored 0.4475 on test.
- (−) Every candidate lowers **domain-level CL macro-F1** (−0.015 to −0.029), even where most CL documents improve.
  The CL drop comes mainly from one document: `CCI_Noida_Software_Technology_Park…`, −0.093 to −0.116 for all three candidates.
- (+) None of them adds routing rules or label-specific hacks. They are plain votes with ties going to the anchor.

**G2:** (+) better than the anchor on 9 of 10 documents. LOO gain is positive whichever document is removed (+0.009…+0.028).
Gains are spread over 7 labels. Its partners are the two runs with the best net error correction (both still net-negative alone).
(−) It is the top of 15 tried pairs. It changes 347 test sentences (16%). ArgumentPetitioner and Statute go down.
**H:** (+) LOO gain is positive in 10/10. (−) Better on only 6/10 documents. Smaller gain than G2.
**C1:** (+) It is the only candidate with one model from every family. (−) ArgumentRespondent −0.086 and RulingByLowerCourt −0.066.
CL −0.029. One LOO split is negative.

## Candidate files (Step 9)
All are in `ens_family/out/candidates/`. Each was validated: `prediction.json` is at the zip root, test doc IDs and
sentence counts match, and every label is in `label_vocab`. None duplicates an existing submission: the nearest
existing prediction set differs on 330–336 test sentences. Nothing has been submitted.
- `submission_G2_anchor+B_lorma_allep+KD_seed123_allep.zip`: 3-model majority (anchor, B_lorma_allep, KD_seed123_allep). The anchor wins ties.
- `submission_H_anchor_w2.0+top4net.zip`: anchor weight 2 plus 4 runs weight 1. The anchor is overruled only if ≥3 of the 4 others agree.
- `submission_C1_one_rep_per_family_bestdev.zip`: best-dev run from each of the 4 families, equal votes. The anchor wins ties.
