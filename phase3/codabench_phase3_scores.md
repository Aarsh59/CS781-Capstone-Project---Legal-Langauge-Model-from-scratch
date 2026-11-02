# Phase 3 (RR) — Codabench test scores (transcribed from the submissions page)

Source: Codabench "My submissions" table, submissions made 2026-10-16 01:22–01:36 (local time).
Score = test Macro-F1 (primary metric). Transcription only; no interpretation (course rule 9).

Kaggle campaign: 20 training runs (one session, 2×T4), config in capstone_phase3_rr.ipynb (Stage 3).
Shared settings (BASE): B2 unless noted, mlp head (hidden 256, dropout 0.1), mean pooling, context L/T/R 64/128/64,
max_targets 4, 8 epochs, patience 2, batch 16, cosine, class_weights inv_sqrt, r=8/alpha=16 for LoRA-family.

| Run | Model | Technique / change vs BASE | Test Macro-F1 | Codabench ID |
|---|---|---|---|---|
| B_lorma | B2 | LoRMA r8, lr 2e-4 | 0.4559 | 991111 (on leaderboard) |
| B_lora_lr2e4 | B2 | LoRA r8, lr 2e-4 | 0.4522 | 991107 |
| B_rslora_r16 | B2 | rsLoRA r16/alpha 32 | 0.4517 | (reported by user) |
| KD | B2, 12 of 18 layers | LoRA r8 student, teacher B_lora, T=2, alpha=0.5 | 0.4501 | 991125 |
| B_lora_r16 | B2 | LoRA r16/alpha 32 | 0.4461 | 991110 |
| B_rslora | B2 | rsLoRA r8 | 0.4460 | 991130 |
| B_qlora | B2 | QLoRA r8, NF4 block 64 | 0.4055 | 991117 |
| B_lora | B2 | LoRA r8, lr 1e-4 | 0.4015 | 991104 |
| B_lora_lr5e5 | B2 | LoRA r8, lr 5e-5 | 0.3937 | 991108 |
| B_lora_ctx_local | B2 | LoRA r8, context L/R 0/0 | 0.3927 | 991105 |
| B_lora_inv_freq | B2 | LoRA r8, class_weights inv_freq | 0.3868 | 991106 (also 991088, same score) |
| B_prompt_p16 | B2 | prompt tuning, 16 tokens | 0.3816 | 991116 |
| D_run | D | LoRA r8, prefix 640 | 0.3801 | 991123 |
| B_prefix_tuning | B2 | prefix tuning, 8 prefixes | 0.3793 | 991112 |
| B_adalora_r4 | B2 | AdaLoRA r8 -> target_rank 4 | 0.3636 | 991073 (also 991069, same score) |
| B_adalora | B2 | AdaLoRA r8 -> target_rank 2 | 0.3632 | 991066 |
| B_ia3 | B2 | IA3 | 0.3623 | 991081 |
| B_prompt_tuning | B2 | prompt tuning, 8 tokens | 0.3552 | 991114 |
| A_run | A | LoRA r8 | 0.3502 | 991065 |
| C_run | C (encoder mode) | LoRA r8, window 512 | 0.3108 | 991121 |

Submission bookkeeping (from the same page):
- All 20 runs have a test score (B_rslora_r16 = 0.4517, reported by the user after the screenshot).
- Duplicates with identical scores: B_lora_inv_freq (0.3868 twice), B_adalora_r4 (0.3636 twice).
- Cancelled entries: 991118 (B_rslora), 991113 (B_prompt_p16), 991084 (B_lora_ctx_local), 991082 (B_lora).
- 25 entries in total on this page (21 finished + 4 cancelled). Check how the daily limit is counted (UTC day, and
  whether cancelled entries count) before submitting ensembles before the 2026-10-16 23:59 deadline.
- Dev macro-F1 per run: summary.csv in /kaggle/working/submissions (not transcribed here yet).

## Campaign 2 (submitted 2026-10-16 03:26–03:30, local time) — transcribed, no interpretation

Same BASE as campaign 1 (backbone dropout 0.0). "vs" = the campaign-1 run it differs from, and the only difference.

| Run | vs | Only difference | Test Macro-F1 | Counterpart score | Codabench ID |
|---|---|---|---|---|---|
| B_rslora_r16_lr2e4 | B_rslora_r16 | lr 1e-4 -> 2e-4 | 0.4498 | 0.4517 (991135) | 991343 |
| B_rslora_r16_lr5e5 | B_rslora_r16 | lr 1e-4 -> 5e-5 | 0.4445 | 0.4517 | 991344 |
| KD_seed123 | KD | seed 42 -> 123 | 0.4407 | 0.4501 | 991351 |
| B_lora_ctx_wide | B_lora | left/right context 64 -> 128 | 0.4269 | 0.4015 | 991339 |
| B_lora_seed123 | B_lora | seed 42 -> 123 | 0.4147 | 0.4015 | 991340 |
| B_lora_unweighted | B_lora | class_weights inv_sqrt -> None | 0.4009 | 0.4015 | 991342 |
| B_lora_crf | B_lora | head mlp -> mlp_crf (CRF loss ignores class weights) | 0.3968 | 0.4015 | 991337 |

Ensembles (weighted hard majority vote per sentence, ties -> earliest member; members from campaign 1):

| Ensemble | Members (weights) | Test Macro-F1 | Codabench ID |
|---|---|---|---|
| ENS_top3 | KD, B_rslora_r16, B_lora | 0.4458 | 991346 |
| ENS_weighted | KD, B_rslora_r16, B_lora, B_qlora, B_lorma (3,3,2,1,1) | 0.4424 | 991353 (991350 cancelled) |
| ENS_top5 | KD, B_rslora_r16, B_lora, B_qlora, B_lorma | 0.4373 | 991349 |
| ENS_top4 | KD, B_rslora_r16, B_lora, B_qlora | 0.4360 | 991347 |
| ENS_recall | KD, B_rslora_r16, B_lora, B_lora_inv_freq | 0.4309 | 991345 |

Best test score overall is still B_lorma 0.4559 (991111).
Seed replicas (same config, seed 42 vs 123): B_lora 0.4015 vs 0.4147 (difference 0.0132); KD 0.4501 vs 0.4407 (0.0094).

## Post-hoc ensemble experiment (ens_experiments.py, outputs in ens_exp_out/)

| Submission | Rule | Dev all / CL / IT | Test Macro-F1 |
|---|---|---|---|
| ENS_candidate | 7-run hard vote + routing: ArgumentPetitioner<-B_lora_ctx_wide; ArgumentRespondent<-B_rslora_r16_lr2e4; RulingByLowerCourt<-B_rslora_r16_lr5e5; PrecedentNotReliedUpon<-B_lora_ctx_wide | 0.5274 / 0.3769 / 0.5576 | 0.4475 |

Dev - test gap for ENS_candidate: 0.0799. The routing candidates were chosen after looking at full-dev per-label F1.
The 7-run hard vote without routing was not submitted (no test score to compare against).

## Campaign 3: the 27 runs re-trained with patience None (suffix _allep), submitted 2026-10-16 10:33–10:34

Only runs whose test predictions changed noticeably vs their early-stopped twin were submitted.

| Run | best epoch / ran | Dev all / CL / IT | Test Macro-F1 | Twin (early-stopped) test | Difference | Codabench ID |
|---|---|---|---|---|---|---|
| B_lora_ctx_wide_allep | 4 / 8 | 0.5109 / 0.4658 / 0.5333 | **0.4616** (new best, on leaderboard) | 0.4269 | +0.0347 | 991973 |
| KD_allep | 8 / 8 | 0.4858 / 0.3783 / 0.5292 | 0.4496 | 0.4501 | -0.0005 | 991970 |
| KD_seed123_allep | 5 / 8 | 0.4916 / 0.4016 / 0.5343 | 0.4461 | 0.4407 | +0.0054 | 991969 |
| B_lorma_allep | 2 / 8 | 0.4781 / 0.3443 / 0.5266 | 0.4386 | 0.4559 | -0.0173 | 991972 |

Not submitted: 19 of the 27 _allep runs changed 0–1.8% of test predictions vs their twins (best epoch 1–3,
i.e. the same window early stopping already covered).

## Family / anchored ensembles (ens_family/, submitted 2026-10-16 11:09–11:10)

| Submission | Members (weights; ties -> anchor) | Dev all / CL / IT | Test Macro-F1 | Codabench ID |
|---|---|---|---|---|
| H_anchor_w2.0+top4net | B_lora_ctx_wide_allep x2; KD_seed123_allep, B_rslora_r16_allep, B_lorma_allep, B_rslora_r16_lr5e5_allep x1 | 0.5251 / 0.4459 / 0.5495 | **0.4659** (new best, on leaderboard) | 992031 |
| G2_anchor+B_lorma_allep+KD_seed123_allep | B_lora_ctx_wide_allep, B_lorma_allep, KD_seed123_allep x1 | 0.5320 / 0.4509 / 0.5670 | 0.4652 | 992028 |
| C1_one_rep_per_family_bestdev | B_lora_ctx_wide_allep, KD_seed123_allep, D_run_allep, B_prompt_p16_allep x1 | 0.5177 / 0.4370 / 0.5505 | 0.4634 | 992032 |

Anchor (B_lora_ctx_wide_allep) test: 0.4616. Codabench counter after these: 7 of 40 today, 47 of 160 total.
