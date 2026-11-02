# Phase 3 analysis bundle

Built by `../audit_predictions.py` from saved artifacts only. No training, no inference, no test labels (none exist locally).

| File | Rows / shape | Keys and columns |
|---|---|---|
| `label_vocab.json` | 13 labels | `labels` = canonical class order (from `data/Phase_3_RR/Train/label_vocab.json`), `index` = label -> column position |
| `dev_gold.csv` | 2,142 sentences, 10 docs (CL 5 / IT 5) | `split, doc_id, domain, sentence_index, gold_label, sentence_text` |
| `test_meta.csv` | 2,197 sentences, 10 docs (CL 5 / IT 5) | `split, doc_id, domain, sentence_index, sentence_text` (no labels) |
| `hard_predictions_dev.csv` | 2,142 rows | `split, doc_id, sentence_index`, then one column per experiment/ensemble with dev predictions (34 runs + 9 ensembles minus those without dev) |
| `hard_predictions_test.csv` | 2,197 rows | same layout, every experiment/ensemble with test predictions (54 runs + 9 ensembles) |
| `probs/<run>_<split>.csv` | 2,142 (dev) / 2,197 (test) rows | `split, doc_id, sentence_index, decode, p_<label>` for each of the 13 labels, columns named by label |
| `evaluation/rr_metrics.py` | — | the metric used everywhere (macro over labels present in gold OR predictions) |
| `evaluation/rr_ensemble.py` | — | `vote(preds, weights)`: weighted hard vote, ties -> earliest listed member |

Sentence order = the order of `text` in the official JSON files. `sentence_index` starts at 0 for each document.

`decode` column in the probability files:
- `argmax`: the submitted label is the argmax of these probabilities (verified for every such run).
- `head.decode`: CRF run (`B_lora_crf`, `B_lora_crf_allep`). The probabilities are the softmax of the **pre-Viterbi emission
  scores**. The submitted labels come from Viterbi decoding over the document, so they are not the argmax of these probabilities.

## What is missing and why
- Campaign-1 runs (20, no suffix): only their test submission zips are local. Their dev predictions, probabilities,
  summaries, logs and checkpoints stayed in the Kaggle output of the first notebook version and were never downloaded.
  Their configs in `../experiment_inventory.csv` are rebuilt from `build_notebook_rr.py` (campaign-1 BASE, patience 2).
- `best_model.pt` checkpoints: not downloaded for any campaign (Kaggle output only).
- Campaign-2 ensembles (ENS_top3/4/5/recall/weighted): their dev predictions are saved, but their members are
  campaign-1 runs, so the dev side cannot be rebuilt from members locally. The test side was rebuilt and matches exactly.
- Test labels: not available. Do not infer them from leaderboard scores.
