# advanced_optimization

Offline only: reads `../bundle` and `../experiment_inventory.csv`, never modifies them.

1. `python diagnostics.py`: measurements, no tuned values. Writes:
   - `candidate_complementarity*.csv`
   - `transitions.csv`, `transition_errors.csv`
   - `cci_noida_*.csv`, `cl_documents_compared.csv`
   - `metric_sensitivity.csv`
2. Fill every `null` in `advanced_config.json`. These are your candidate grids and decision thresholds (course rule 9).
3. `python optimize.py`: leave-one-document-out evaluation of methods 1–6, with all fitting on each fold's 9 training documents. Writes:
   - `method_comparison.csv`, `loo_per_document.csv`, `loo_per_class.csv`
   - `selected_configurations_by_fold.json`

`common.py` holds the loaders and the metric wrapper. `score(P, ds, convention)` supports `gold_or_pred` (the local scorer),
`gold_only` and `all13`.

Use the venv that has numpy: `~/.cache/cs781_smoke_gpu_venv/bin/python`.
