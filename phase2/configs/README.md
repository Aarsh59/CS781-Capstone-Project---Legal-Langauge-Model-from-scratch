# Run config fields (train_ner.py)

`null` = your choice (course rule 9). Required ones fail loudly if left null.

| key | values | notes |
|---|---|---|
| phase1_ckpt | path to a Phase-1 `best.pt` | A=run_A, B=run_B2 (or run_B), C=run_C, D=run_D |
| backbone.prefix | D only: `"full"` / int / `0` | full = bidirectional window; int p = bidirectional first p tokens; 0 = causal |
| backbone.encdec_mode | C only: `"encoder"` / `"decoder"` | encoder states, or decoder states with cross-attention to the encoder |
| backbone.truncate_layers | int or null | keep first k blocks (KD student) |
| backbone.dropout | float or null | null = checkpoint value (0.1) |
| preprocess.split_punct | true / false | split punctuation runs; alignment ceiling 99.30% vs 98.19% on train |
| preprocess.tag_scheme | `"BIO"` / `"BIOES"` | |
| split | `{"mode":"random","val_frac":f,"seed":s}` / `{"mode":"fold","val_fold":"fold2"}` / `{"mode":"none"}` | none = train on all 70 docs, no validation (saves last epoch) |
| window.train_len / infer_len | ≤ backbone max_seq_len (A/B/D 896, C encoder 640, C decoder mode 256) | |
| window.train_stride / infer_stride | ≤ len | |
| window.merge | `"center"` / `"left_context"` / `"mean"` | how overlapping window logits are combined |
| head | `{"name": ..., ...}` | passed to `heads.build_head` |
| technique | `{"name": "full_ft"/"lora"/"lorma"/"adapters"/"bitfit"/"partial_ft", ...}` | passed to `techniques/<name>.apply` |
| kd | null / `{"teacher_run": "runs/<teacher name>", ...}` | extra keys passed to `techniques/kd.kd_loss` |
| train.batch_size / micro_batch | windows per optimizer step / per forward | batch_size % micro_batch == 0 |
| train.schedule | `"linear"` / `"cosine"` / `"constant"` | after `warmup_frac` linear warm-up |
| train.class_weights | null / `"inv_freq"` / `"inv_sqrt"` | CE weights from train tag counts (mean 1) |
| train.o_weight | null / float | overrides the O-tag weight |
| train.patience | null / int | stop after this many epochs without val macro-F1 gain |
| train.max_minutes | null / float | hard stop for training (then evaluates + predicts) |
| decode.repair | `"lenient"` / `"strict"` | dangling I-/E- tag starts an entity / is dropped |
