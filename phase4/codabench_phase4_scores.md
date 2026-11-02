# Phase 4 (CJPE) — Codabench test scores and dev scores (transcribed; no interpretation, course rule 9)

Source: Codabench "My submissions" (Phase 4), submissions 2026-10-19 18:33–18:40 IST; counter after them: 7 of 40 today,
7 of 160 total (5 entries listed); account-1 runs submitted 2026-10-19 19:04–19:06 (11 entries listed in total). Test score = Codabench test Macro-F1 (primary metric).
Dev = this pipeline's dev Macro-F1 at threshold 0.5: dev_all = released dev (3,505 rows, the 994 duplicate rows counted
as released), dev_excl_test = dev rows whose text is not in test.json (1,988).
All runs: 1 epoch on the 31,653 de-duplicated train documents, max 4 chunks per document, attention pooling, MLP head.

| Run | Model | Technique / chunking | dev_all | dev_excl_test | Test Macro-F1 | Codabench ID |
|---|---|---|---|---|---|---|
| C3_B2_overlap | B2 | full FT, overlapping 896 windows (stride 640) | 0.7489 | 0.7422 | 0.7574 | 1002980 (on leaderboard) |
| W1_B2_full_weighted | B2 | full FT, class weights 0.847/1.220, even | 0.7435 | 0.7496 | 0.7355 | 1002987 |
| C1_B2_headtail | B2 | full FT, head + tail | 0.7281 | 0.7273 | 0.7291 | 1002991 (re-submitted; 1002979 no longer listed) |
| E7_B2_adalora | B2 | AdaLoRA r16 -> 4, even | 0.5525 | 0.5620 | 0.5402 | 1002986 |
| E6_B2_lorma | B2 | LoRMA r8, even | 0.3749 | 0.3832 | 0.3641 | 1002992 (re-submitted; 1002984 no longer listed) |
| E1_B2_full | B2 | full FT, even | 0.7357 | 0.7447 | 0.7238 | 1003024 |
| E8_KD_B2s12 | B2, 12 of 18 blocks | KD student (full FT, 107.4M), T 2, alpha 0.5, teacher = best account-1 run | 0.7427 | 0.7493 | 0.7341 | 1003032 |
| E5_D_bitfit | D | BitFit, prefix 640, even | 0.5120 | 0.5192 | 0.5027 | 1003031 |
| E3_A_partial | A | partial FT (top 8 of 17 blocks), even | 0.4663 | 0.4642 | 0.4690 | 1003028 |
| E4_C_adapters | C (encoder) | adapters (bottleneck 64, attn + ffn), 640-token chunks | 0.3844 | 0.3773 | 0.3935 | 1003029 |
| E2_B3_full | B3 | full FT, even, 1024-token chunks (dev: every row predicted 0) | 0.3329 | 0.3333 | 0.3323 | 1003025 |

## Dev per-class precision / recall (dev_all; class 1 = accepted), from the two accounts' summary.csv

| Run | P0 | R0 | P1 | R1 | Confusion [[g0p0, g0p1], [g1p0, g1p1]] | Trainable | Train min |
|---|---|---|---|---|---|---|---|
| C3_B2_overlap | 0.6912 | 0.9228 | 0.8846 | 0.5894 | [[1614, 135], [721, 1035]] | 148.8M | 118.0 |
| W1_B2_full_weighted | 0.7021 | 0.8542 | 0.8148 | 0.6390 | [[1494, 255], [634, 1122]] | 148.8M | 98.3 |
| E8_KD_B2s12 | 0.6888 | 0.9062 | 0.8638 | 0.5923 | [[1585, 164], [716, 1040]] | 107.4M | 64.5 |
| E1_B2_full | 0.6843 | 0.8959 | 0.8502 | 0.5883 | [[1567, 182], [723, 1033]] | 148.8M | 95.9 |
| C1_B2_headtail | 0.6788 | 0.8879 | 0.8389 | 0.5814 | [[1553, 196], [735, 1021]] | 148.8M | 103.5 |
| E7_B2_adalora | 0.5543 | 0.8525 | 0.6834 | 0.3172 | [[1491, 258], [1199, 557]] | 4.1M | 141.9 |
| E5_D_bitfit | 0.5417 | 0.9245 | 0.7462 | 0.2210 | [[1617, 132], [1368, 388]] | 0.31M | 99.6 |
| E3_A_partial | 0.5291 | 0.9605 | 0.7909 | 0.1486 | [[1680, 69], [1495, 261]] | 60.0M | 51.0 |
| E4_C_adapters | 0.5063 | 0.9720 | 0.6667 | 0.0558 | [[1700, 49], [1658, 98]] | 2.18M | 39.9 |
| E6_B2_lorma | 0.5072 | 0.9903 | 0.8111 | 0.0416 | [[1732, 17], [1683, 73]] | 2.0M | 101.2 |
| E2_B3_full | 0.4990 | 1.0000 | 0.0000 | 0.0000 | [[1749, 0], [1756, 0]] | 148.8M | 109.4 |

## Threshold / ensemble sweep (sweep_c3.py; selected on dev_excl_test), submitted 2026-10-19 19:31–19:38 IST

| Submission | Weights C3/E8/W1 | Threshold | dev_excl_test F1 (in-sample / held-out) | Test Macro-F1 | Codabench ID |
|---|---|---|---|---|---|
| ENS_C30.75_E80.25_W10.00 | 0.75/0.25/0 | 0.34 | 0.7998 / 0.7959 | 0.7864 | 1003106 (on leaderboard) |
| ENS_C30.60_E80.20_W10.20 | 0.60/0.20/0.20 | 0.36 | 0.8018 / 0.7978 | 0.7857 | 1003114 |
| C3_B2_overlap_tuned | 1/0/0 | 0.30 | 0.7846 / 0.7799 | 0.7854 | 1003096 |
| ENS_C30.75_E80.00_W10.25 | 0.75/0/0.25 | 0.37 | 0.7906 / 0.7859 | 0.7825 | 1003104 |
| ENS_C30.70_E80.15_W10.15 | 0.70/0.15/0.15 | 0.37 | 0.7986 / 0.7948 | 0.7825 | 1003103 |
| ENS_C30.50_E80.25_W10.25 | 0.50/0.25/0.25 | 0.39 | 0.8054 / 0.8025 | 0.7818 | 1003088 |
| ENS_C30.50_E80.00_W10.50 | 0.50/0/0.50 | 0.41 | 0.7981 / 0.7943 | 0.7810 | 1003113 |
| C3_B2_overlap_0.5 | 1/0/0 | 0.50 | 0.7422 / 0.7418 | 0.7574 | 1003092 |
| W1_B2_full_weighted_tune | 0/0/1 | 0.38 | 0.7666 / 0.7614 | 0.7467 | 1003112 |
| W1_B2_full_weighted_0.5 | 0/0/1 | 0.50 | 0.7496 / 0.7494 | 0.7355 | 1003107 |
| E8_KD_B2s12_0.5 | 0/1/0 | 0.50 | 0.7493 / 0.7496 | 0.7341 | 1003098 |

Codabench counter not visible in the screenshot; 22 entries listed in total.

## Later submissions (transcribed)

| Submission | What | Test Macro-F1 | Codabench ID |
|---|---|---|---|
| ENS_bestth_C30.75_E80.25 | C3/E8 0.75/0.25, each centred at its own tuned threshold (0.295 / 0.3605) | 0.7887 (on leaderboard) | 1003142 |
| E8_KD_B2s12_th0.3605 | E8 alone, threshold 0.3605 | 0.7640 | 1003137 |
| C3_long_best_dev_all / _best_dev_excl_te / _best_dev_in_test | C3 fresh 5 epochs, all three bests = e2_step7000, threshold 0.5 | 0.7633 (all three) | 1004322 / 1004323 / 1004324 |
| E8_long_best_dev_all / _best_dev_excl_te | E8 fresh 8 epochs (KD), same checkpoint for both keys, threshold 0.5 | 0.7577 | 1004326 / 1004327 |
| E8_long_best_dev_in_test | E8 fresh 8 epochs, checkpoint e3_step8000 (selected with test labels) | 0.7626 | 1004328 |

Counter on 2026-10-20 09:17: 6 of 40 today, 35 of 160 total.

## 2026-10-20 submissions (transcribed)

| Submission | What | Selected on | Test Macro-F1 | Codabench ID |
|---|---|---|---|---|
| ENS_C3_avg8+E8L_avg8+E8_1ep | re-chunked C3 (avg of 8-chunk variants) + E8_long (avg8) + E8 1ep, equal, thr 0.361 | held-out dev_excl_test (rank 1) | 0.8013 | 1005218 |
| ENS_C3_avg8+E8L_ov8+E8_1ep | as above with E8_long overlap:8, thr 0.379 | held-out (rank 2) | 0.8009 | 1005216 |
| ENS_C3_own+E8L_avg8 | thr 0.409 | held-out (rank 3) | 0.8042 | 1005215 |
| ENS_C3_own+E8L_ov8+E8_1ep | thr 0.410 | held-out (rank 4) | 0.7982 | 1005214 |
| ENS_E8L_avg8+C3L_ov8+E8_1ep | thr 0.363 | held-out (rank 5) | 0.7932 | 1005213 |
| best_robust_A_ENS07 | dual-split Track A | held-out Track A | 0.7831 | 1004370 |
| best_track_B_overlap_ENS05 | dual-split Track B (OVERLAP-INFORMED: tuned on the test documents) | dev_in_test | 0.8082 (selected on leaderboard) | 1004367 |
| ENS_C3_1ep+E8_long | equal weights, thr 0.40 (ens_search #4) | held-out + dev_all tie-break | 0.8016 | 1004362 |

All Codabench scores above equal the locally computed dev_in_test values (labels from the original dev.json copies of the
test documents) to 4 decimals -> the scored test labels are unchanged.
