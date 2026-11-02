# Phase 3: second-stage offline analysis

Script: `second_stage.py`. It reads only `../bundle` and `../experiment_inventory.csv`. No training, inference, downloads
or submissions. Test labels do not exist and were not used. All scores below are **dev** scores. **No number here
predicts the official test score.**

**Metric (local):** `rr_metrics.rr_scores`, macro-F1 over labels present in gold **or** predictions. Dev has no Dissent
or PrecedentOverruled gold. A method that predicts Dissent anywhere therefore adds an F1 of 0 to its average. On a
single document, only that document's labels count, so per-document scores average over different label sets.
**Codabench's exact convention is unknown**: its scorer image was not inspected.

## 1. Complementarity with H (`complementarity_vs_H*.csv`)
- H is wrong on 964 of 2,142 dev sentences.
- **Against H, every individual run is net-negative.** No single run corrects more H errors than it breaks H-correct
  sentences. The best net values belong to other ensembles: G2 +26, ENS_candidate +15. Single runs start at −5
  (`B_lora_ctx_wide`, the early-stopped twin of H's anchor) and `B_lora_ctx_wide_allep` −19.
- Document and class breakdowns are in the `_by_document` and `_by_class` files.

## 2. Leave-one-document-out (`loo_summary.csv`, `loo_per_document.csv`, `loo_fold_choices.csv`)
Every `LOO_*` method chooses its members, k, weight or biases from the 9 training documents of each fold, then predicts
the held-out document. "Pooled" means the 10 held-out predictions are concatenated and scored once. The FIXED_* rows
are the submitted ensembles. **Their members were chosen on all 10 documents, so their rows are not held-out estimates.**

| Method | Pooled dev | CL | IT | Per-doc mean / median / sd | Docs better / worse than anchor |
|---|---|---|---|---|---|
| BASE_anchor (no optimisation) | 0.5109 | 0.4658 | 0.5333 | 0.478 / 0.461 / 0.126 | — |
| FIXED_H (submitted 0.4659) | 0.5251 | 0.4459 | 0.5495 | 0.496 / 0.477 / 0.126 | 6 / 4 |
| FIXED_G2 (submitted 0.4652) | 0.5320 | 0.4509 | 0.5670 | 0.520 / 0.489 / 0.151 | 9 / 1 |
| FIXED_C1 (submitted 0.4634) | 0.5177 | 0.4370 | 0.5505 | 0.473 / 0.429 / 0.123 | 7 / 3 |
| LOO_anchor_plus2 | 0.5191 | 0.4352 | 0.5511 | 0.513 / 0.460 / 0.153 | 8 / 2 |
| LOO_anchor_w_top4 | 0.5082 | 0.4303 | 0.5375 | 0.500 / 0.456 / 0.164 | 7 / 3 |
| LOO_topk_hard | 0.5045 | 0.4071 | 0.5370 | 0.473 / 0.438 / 0.129 | 4 / 6 |
| LOO_topk_soft (one calibration group) | **0.5282** | **0.3796** | 0.5659 | 0.486 / 0.440 / 0.128 | 7 / 3 |
| LOO_anchor_bias | 0.5092 | 0.4605 | 0.5348 | 0.473 / 0.430 / 0.127 | 4 / 4 |

What the held-out numbers show:
- **The G2 pair is a stable choice.** Selected inside each fold, the anchor+2 method picked B_lorma_allep + KD_seed123
  (either seed variant) in **8 of 10 folds**. Its held-out pooled gain over the anchor is **+0.008**, against +0.021
  in-sample. That is about the shrinkage you'd expect when selection is done honestly. On test, G2 beat the anchor by +0.0036.
- **Anchor weighting, top-k hard voting and class biases do not beat the anchor held-out** (−0.003, −0.006 and −0.002 pooled).
  The bias method never chose to lower RatioOfTheDecision (b_Ratio = 0 in all 10 folds), even though the confusion
  matrices showed Ratio absorbing other labels. It always picked a +0.3 or +0.6 boost for the argument roles / RulingByLowerCourt.
- **Soft top-k has the highest held-out pooled score (0.5282), but:**
  - its CL score is 0.3796, which is **−0.086** vs the anchor and **−0.066** vs H;
  - its gain comes mainly from **one document** (`BHC_BomHC_2007_1278`, +0.134);
  - by class, it comes from RatioOfTheDecision (+0.057), PrecedentNotReliedUpon (+0.084, 50 sentences) and
    PrecedentReliedUpon (+0.051). Statute (−0.021) and None (−0.022) go down;
  - it is +0.003 above H's (non-held-out) pooled score, which is far below the ±0.08 document-bootstrap uncertainty of a 10-document score.
- **One document hurts every combination:** `CCI_Noida_Software_Technology_Park…` (CL) drops by about 0.09–0.11 under
  every ensemble method. That one document accounts for much of the CL loss.

## 3. Hard voting vs probability averaging (`hard_vs_soft_in_sample.csv`, `calibration.csv`)
- **Probabilities used:** only verified ones. Softmax, label_vocab order, rows summing to 1, and argmax equal to the submitted labels.
  CRF runs (pre-Viterbi emissions) and KD / unweighted-loss runs were kept out of averaging.
  The averaging group is the 26 inv_sqrt-cross-entropy, non-CRF runs.
- **Calibration is not equivalent across runs:** ECE ranges from about 0.06 to 0.27 (`calibration.csv`), so the group restriction matters.
- **Inside that one group, soft voting is slightly above hard voting in-sample:** k = 3: 0.5354 vs 0.5311; k = 5: 0.5278 vs 0.5141;
  k = 7: 0.5323 vs 0.5170. The earlier studies found the opposite because they mixed loss types (KD, unweighted) in the average.

## 4. Per-class and per-domain
- `loo_per_class_pooled.csv` has precision, recall, F1 and support for every method's pooled held-out predictions.
- **The CL/IT split is the dominant pattern:** every ensemble raises IT (+0.004 to +0.034) and lowers CL (−0.005 to −0.086)
  relative to the anchor, which is the best CL model by a wide margin.

## 5. Recommendation
**The criterion, stated before looking:** a not-yet-submitted configuration is worth a submission only if its **held-out**
pooled gain over H is positive, it doesn't come from a single document, and it doesn't cost CL heavily.

- **Result against the criterion:** no configuration meets it.
  - The only held-out method above H is soft top-k, at +0.003. That's below any reasonable noise level.
    It depends on one document, and it costs 0.066 CL against H.
  - The anchor+2 method is held-out-stable, but it is essentially G2, which is already submitted (0.4652).
- **On this evidence, another submission is not justified.** H (0.4659), G2 (0.4652) and C1 (0.4634) already cover the
  configurations that hold up under leave-one-document-out. The final decision is yours.

## Files
- `second_stage.py`: the exact script.
- Complementarity: `complementarity_vs_H.csv`, `complementarity_vs_H_by_document.csv`, `complementarity_vs_H_by_class.csv`.
- Leave-one-document-out: `loo_summary.csv`, `loo_per_document.csv`, `loo_per_class_pooled.csv`, `loo_fold_choices.csv`.
- Voting and calibration: `hard_vs_soft_in_sample.csv`, `calibration.csv`.
