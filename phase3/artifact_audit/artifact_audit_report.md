# Phase 3 artifact audit

Everything in this folder was produced by `audit_predictions.py`, which only reads existing artifacts. All
"verified" items were checked by code. Items marked **hypothesis** are not established by the data here.

## 1. What exists

| | Count | Notes |
|---|---|---|
| Distinct training experiments | **54** | campaign 3 `_allep` 27, campaign 2 7, campaign 1 20 |
| Ensembles recovered | **9** | campaign-2 ENS_top3/4/5/recall/weighted, ENS_candidate (routing), G2, H, C1 |
| Runs with valid dev predictions | 34 | campaigns 2 and 3 |
| Runs with valid test predictions | 54 | all, including the 20 campaign-1 test zips |
| Runs with usable probabilities (dev + test) | 34 | softmax, 13 columns in label_vocab order, rows sum to 1 |
| Runs with hard predictions only | 20 | campaign 1 (test only) |
| Integrity errors / warnings | **0 / 0** | 28 informational notes in `integrity_issues.csv` |
| Zips scanned | 101 | 36 are valid Phase-3 predictions. 65 are other phases (in `Phase3 Submissions/dump/` and `~/Downloads/`). See `zip_audit.csv`. |

**Verified integrity** (every run and ensemble):
- doc IDs match the official dev and test files, sentence counts match, and sentence order is preserved;
- every label is in label_vocab, and probability class order = label_vocab;
- probabilities are finite, non-negative and sum to 1, so they are not logits;
- for every non-CRF run, the argmax of `dev_probs` reproduces its `dev_predictions`;
- recomputed dev macro-F1 equals the saved `summary.json` value for all 34 runs.

**Ensemble reconstruction** (`ensembles.json`):
- **Test predictions of all 9 ensembles were rebuilt from their members and match the saved files exactly.**
- Dev: ENS_candidate rebuilt and matches.
- G2/H/C1 never had saved dev predictions, so their dev predictions were regenerated (not an independent check).
- The 5 campaign-2 ensembles' dev side cannot be rebuilt, because their members are campaign-1 runs with no local dev predictions.
- None of the ensembles uses min-votes, class-specific weights or document/domain routing, except ENS_candidate, which has 4 label→model routing rules.
- Ties always go to the earliest-listed member.

## 2. Duplicates and identical predictions (verified)
- `B_adalora_allep` and `B_adalora_r4_allep`: identical dev predictions, 1 differing test sentence.
- Identical **test** predictions between a campaign-1 run and its `_allep` re-run (best epoch inside the early-stopping window):
  B_adalora, B_adalora_r4, B_lora, B_lora_lr5e5, C_run, D_run.
- No two zip files are byte-identical in content.

## 3. Metric convention (verified for the local metric only)
- `rr_metrics.rr_scores` averages F1 over labels that appear in gold **or** predictions. Dev has no Dissent or
  PrecedentOverruled gold sentences. A run that predicts Dissent on dev therefore averages over 12 labels and gets
  an F1 of 0 for Dissent. A run that never predicts it averages over 11. Both cases occur among the 34 runs.
- **Not verified:** whether Codabench uses the same convention. The scorer is the Docker image `shub7ham/cs781-scoring:v2`,
  which was not inspected.

## 4. Best runs and ensembles (verified numbers)
| | Dev macro-F1 | Official test |
|---|---|---|
| H_anchor_w2.0+top4net (ensemble) | 0.5251 | **0.4659** (best) |
| G2 anchor + B_lorma_allep + KD_seed123_allep | **0.5320** (best dev) | 0.4652 |
| C1 one rep per family | 0.5177 | 0.4634 |
| B_lora_ctx_wide_allep (best single) | 0.5109 | 0.4616 |
| ENS_candidate (routing) | 0.5274 | 0.4475 |
| ENS_top5 (campaign 2) | 0.4999 | 0.4373 |
| B_lorma (campaign 1) | — (dev not local) | 0.4559 |

## 5. Error patterns (verified, individual runs on dev)
- **Per-label F1** (median / best across runs):
  - Lowest: RulingByLowerCourt 0.21 / 0.42; None 0.31 / 0.37; PrecedentNotReliedUpon 0.31 / 0.43;
    ArgumentRespondent 0.35 / 0.51; ArgumentPetitioner 0.45 / 0.56.
  - Highest: Statute 0.78 / 0.80; Fact 0.62 / 0.66.
  - Dissent is 0 wherever it is predicted, because there is no Dissent gold in dev.
- **Per-document median macro-F1** ranges from **0.29** (`SC_Competition Commission of India vs Bharti Airtel…`, CL)
  to **0.61** (`SC_2009_1357`, IT). Full list in `metrics/per_document.csv`.
- **Confusions:** see `metrics/confusion/<run>.csv`. In earlier analysis, the dominant pattern was that argument roles and
  RulingByLowerCourt get predicted as RatioOfTheDecision.

## 6. Complementary errors (verified; from `../ens_family/out/anchored_error_analysis.csv`)
- The best single run is wrong on 983 of 2,142 dev sentences. **Every other run, used alone as a second opinion, breaks more
  of its correct predictions than it fixes** (best net: B_lorma_allep −27, KD_seed123_allep −40).
- Gains appeared only when the anchor is overruled by agreement of several others: G2, which needs both of 2 others to agree, and
  H, which needs ≥3 of 4. That is consistent with their test results (+0.004 and +0.004 over the anchor).

## 7. Are the probabilities comparable for soft voting? (verified facts)
- **The training losses differ:** inv_sqrt class-weighted CE (most runs), unweighted CE (`B_lora_unweighted*`),
  KD (teacher soft targets + unweighted CE), and CRF (pre-decoding emissions only).
- **Calibration differs:** ECE ranges from 0.058 to 0.272 across the non-CRF runs (`../ens_family/out/summary.json`).
- **Soft voting lost to hard voting** in both earlier studies on dev (0.4565 vs 0.4976, and 0.5198 vs 0.5320).
- Campaign-1 runs have no probabilities at all.

## 8. Could the gains be dev overfitting? (facts; the conclusion is the reader's)
- The dev set is 10 documents. A document bootstrap of the best single run gives a 95% interval of **0.43–0.59**
  around 0.511, i.e. about ±0.08.
- All ensemble members and rules were chosen on that same dev set.
- Dev−test gaps observed: 0.02–0.08. The largest was ENS_candidate, the most heavily dev-selected rule set, at 0.080.
- Differences between the top 4 official scores (0.4616–0.4659) are below the measured seed-to-seed variation (~0.01).

## 9. Analyses possible without retraining (options, not recommendations)
- Use `hard_predictions_*.csv` for any hard-vote scheme over the 34 dev-scored runs.
- Use `probs/` for soft or weighted voting over the 34 probability runs. Keep CRF and different loss types apart, or calibrate them first.
- Use `metrics/per_document.csv` and `metrics/per_class.csv` for leave-one-document-out checks of any rule before spending a submission.
- Campaign-1 runs can take part in **test-side** hard votes only, so their dev effect can't be measured.

## Files
- `experiment_inventory.csv`: one row per experiment and ensemble (54 + 9), with original metrics, the Codabench score and ID, paths and status.
- `integrity_issues.csv`, `zip_audit.csv`, `ensembles.json`, `submissions.csv`, `audit_summary.json`.
- `metrics/`: recomputed metrics, kept separate from the originals:
  - `per_run_summary.csv` (has a `matches_original` column), `per_class.csv`, `per_document.csv`
  - `confusion/<run>.csv`, `pairwise_disagreement.csv` (dev and test, all pairs)
- `bundle/`: analysis-ready data. See `bundle/README.md` for schemas.

## Could not be recovered
- Campaign-1 dev predictions, probabilities, summaries, logs and configs: never downloaded from the first notebook version's
  Kaggle output. Their configs are rebuilt from the notebook builder.
- Checkpoints for all runs: Kaggle output only.
- Codabench's exact metric convention (scorer image not inspected).
- Codabench scores are transcribed from your screenshots, not read from Codabench.
