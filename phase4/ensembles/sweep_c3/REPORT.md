# Phase 4 — threshold + ensemble sweep (validation only; selection set dev_excl_test, n = 1,988)

Validation metrics only. Official test score so far: C3_B2_overlap @0.5 = 0.7574 (Codabench 1002980).
Held-out = threshold chosen on a random half of dev_excl_test, scored on the other half (500 splits).
Criterion from your spec: keep an ensemble only if it beats the TUNED C3 baseline meaningfully and repeatably.

| Candidate | Weights C3/E8/W1 | Threshold | In-sample F1 | Held-out F1 | vs tuned C3 (held-out) | P1 | R1 | Train min | Test share pred 1 | Zip |
|---|---|---|---|---|---|---|---|---|---|---|
| ENS_C30.50_E80.25_W10.25 | 0.50/0.25/0.25 | 0.39 | 0.8054 | 0.8025 ± 0.0087 | +0.0226 | 0.836 | 0.761 | 280.8 | 0.4885 | ENS_C30.50_E80.25_W10.25/submission_ENS_C30.50_E80.25_W10.25.zip |
| ENS_C30.60_E80.20_W10.20 | 0.60/0.20/0.20 | 0.36 | 0.8018 | 0.7978 ± 0.0090 | +0.0179 | 0.810 | 0.789 | 280.8 | 0.5069 | ENS_C30.60_E80.20_W10.20/submission_ENS_C30.60_E80.20_W10.20.zip |
| ENS_C30.75_E80.25_W10.00 | 0.75/0.25/0.00 | 0.34 | 0.7998 | 0.7959 ± 0.0101 | +0.0160 | 0.803 | 0.795 | 182.5 | 0.5115 | ENS_C30.75_E80.25_W10.00/submission_ENS_C30.75_E80.25_W10.00.zip |
| ENS_C30.70_E80.15_W10.15 | 0.70/0.15/0.15 | 0.37 | 0.7986 | 0.7948 ± 0.0096 | +0.0149 | 0.818 | 0.769 | 280.8 | 0.4931 | ENS_C30.70_E80.15_W10.15/submission_ENS_C30.70_E80.15_W10.15.zip |
| ENS_C30.50_E80.00_W10.50 | 0.50/0.00/0.50 | 0.41 | 0.7981 | 0.7943 ± 0.0102 | +0.0144 | 0.838 | 0.740 | 216.3 | 0.468 | ENS_C30.50_E80.00_W10.50/submission_ENS_C30.50_E80.00_W10.50.zip |
| E8_KD_B2s12@tuned | 0.00/1.00/0.00 | 0.36 | 0.7928 | 0.7898 ± 0.0102 | +0.0099 | 0.790 | 0.797 | 64.5 | 0.5142 | E8_KD_B2s12_tuned/submission_E8_KD_B2s12_tuned.zip |
| ENS_C30.75_E80.00_W10.25 | 0.75/0.00/0.25 | 0.37 | 0.7906 | 0.7859 ± 0.0094 | +0.0060 | 0.807 | 0.765 | 216.3 | 0.4957 | ENS_C30.75_E80.00_W10.25/submission_ENS_C30.75_E80.00_W10.25.zip |
| C3_B2_overlap@tuned | 1.00/0.00/0.00 | 0.30 | 0.7846 | 0.7799 ± 0.0109 | +0.0000 | 0.775 | 0.803 | 118.0 | 0.5399 | C3_B2_overlap_tuned/submission_C3_B2_overlap_tuned.zip |
| W1_B2_full_weighted@tuned | 0.00/0.00/1.00 | 0.38 | 0.7666 | 0.7614 ± 0.0104 | -0.0185 | 0.760 | 0.779 | 98.3 | 0.5221 | W1_B2_full_weighted_tuned/submission_W1_B2_full_weighted_tuned.zip |
| E8_KD_B2s12@0.5 | 0.00/1.00/0.00 | 0.50 | 0.7493 | 0.7496 ± 0.0101 | -0.0303 | 0.874 | 0.598 | 64.5 | 0.3454 | E8_KD_B2s12_0.5/submission_E8_KD_B2s12_0.5.zip |
| W1_B2_full_weighted@0.5 | 0.00/0.00/1.00 | 0.50 | 0.7496 | 0.7494 ± 0.0096 | -0.0305 | 0.823 | 0.644 | 98.3 | 0.3949 | W1_B2_full_weighted_0.5/submission_W1_B2_full_weighted_0.5.zip |
| C3_B2_overlap@0.5 | 1.00/0.00/0.00 | 0.50 | 0.7422 | 0.7418 ± 0.0097 | -0.0381 | 0.890 | 0.571 | 118.0 | 0.3507 | C3_B2_overlap_0.5/submission_C3_B2_overlap_0.5.zip |

Member agreement on dev_excl_test at 0.5: {'C3_B2_overlap~E8_KD_B2s12': 0.836, 'C3_B2_overlap~W1_B2_full_weighted': 0.8249, 'E8_KD_B2s12~W1_B2_full_weighted': 0.8561}
Longer C3 training (part 3): not run.
