# Phase 4 — dual-split ensemble optimisation

Validation numbers only. dev_in_test = the 1,517 test documents (labels from dev.json): Track B and every dev_in_test column are OVERLAP-INFORMED, i.e. equal to / optimised on the test score; not an independent estimate.
dev_excl_test = 994 distinct documents (dev.json lists each twice; earlier sweeps' held-out checks split rows and were therefore optimistic).
Search: seed 0, 400 Dirichlet weight vectors + initial + one-hot + coordinate refinement per (config, track); thresholds 0.20-0.60 step 0.01, refined +/-0.01 step 0.001.

| ID | Members | Track | Weights | Threshold | dev_excl_test | dev_in_test (= test) | dev_all | held-out Track A |
|---|---|---|---|---|---|---|---|---|
| ENS07 | C3+E8+E8_long_A | A | {"C3": 0.31136, "E8": 0.524425, "E8_long_A": 0.164215} | 0.367 | 0.8179 | 0.7831 | 0.8029 | 0.8045 ± 0.0218 |
| ENS07 | C3+E8+E8_long_A | B | {"C3": 0.513853, "E8": 0.08433, "E8_long_A": 0.401816} | 0.378 | 0.7978 | 0.8075 | 0.8020 | 0.8045 ± 0.0218 |
| ENS02 | C3+E8 | A | {"C3": 0.390247, "E8": 0.609753} | 0.354 | 0.8129 | 0.7785 | 0.7980 | 0.7990 ± 0.0260 |
| ENS02 | C3+E8 | B | {"C3": 0.617933, "E8": 0.382067} | 0.297 | 0.7893 | 0.7927 | 0.7908 | 0.7990 ± 0.0260 |
| ENS06 | C3+E8+C3_long | A | {"C3": 0.366348, "E8": 0.536149, "C3_long": 0.097503} | 0.354 | 0.8149 | 0.7868 | 0.8028 | 0.7968 ± 0.0206 |
| ENS06 | C3+E8+C3_long | B | {"C3": 0.236408, "E8": 0.471402, "C3_long": 0.29219} | 0.4 | 0.8047 | 0.7983 | 0.8020 | 0.7968 ± 0.0206 |
| ENS10 | C3+E8+W1+E8_long_A | A | {"C3": 0.241513, "E8": 0.428857, "W1": 0.039674, "E8_long_A": 0.289955} | 0.378 | 0.8179 | 0.7791 | 0.8011 | 0.7912 ± 0.0278 |
| ENS10 | C3+E8+W1+E8_long_A | B | {"C3": 0.452688, "E8": 0.180943, "W1": 0.013596, "E8_long_A": 0.352773} | 0.405 | 0.8024 | 0.8074 | 0.8046 | 0.7912 ± 0.0278 |
| SINGLE_E8 | E8 | A | {"E8": 1.0} | 0.357 | 0.7937 | 0.7632 | 0.7805 | 0.7910 ± 0.0233 |
| SINGLE_E8 | E8 | B | {"E8": 1.0} | 0.412 | 0.7815 | 0.7660 | 0.7745 | 0.7910 ± 0.0233 |
| ENS09 | C3+E8+W1+C3_long | A | {"C3": 0.203025, "E8": 0.534194, "W1": 0.093508, "C3_long": 0.169273} | 0.403 | 0.8157 | 0.7864 | 0.8030 | 0.7910 ± 0.0235 |
| ENS09 | C3+E8+W1+C3_long | B | {"C3": 0.23471, "E8": 0.444537, "W1": 0.036559, "C3_long": 0.284194} | 0.398 | 0.8048 | 0.8009 | 0.8031 | 0.7910 ± 0.0235 |
| ENS01 | C3+E8+W1 | A | {"C3": 0.382515, "E8": 0.613777, "W1": 0.003708} | 0.352 | 0.8119 | 0.7791 | 0.7977 | 0.7906 ± 0.0282 |
| ENS01 | C3+E8+W1 | B | {"C3": 0.573629, "E8": 0.263036, "W1": 0.163335} | 0.341 | 0.7998 | 0.7934 | 0.7971 | 0.7906 ± 0.0282 |
| ENS05 | C3+E8_long_A | A | {"C3": 0.569153, "E8_long_A": 0.430847} | 0.401 | 0.8085 | 0.8068 | 0.8078 | 0.7906 ± 0.0227 |
| ENS05 | C3+E8_long_A | B | {"C3": 0.586992, "E8_long_A": 0.413008} | 0.392 | 0.8037 | 0.8082 | 0.8056 | 0.7906 ± 0.0227 |
| ENS03 | C3+W1 | A | {"C3": 0.33699, "W1": 0.66301} | 0.373 | 0.8008 | 0.7770 | 0.7905 | 0.7849 ± 0.0204 |
| ENS03 | C3+W1 | B | {"C3": 0.569717, "W1": 0.430283} | 0.362 | 0.7907 | 0.7902 | 0.7906 | 0.7849 ± 0.0204 |
| ENS08 | C3+W1+C3_long | A | {"C3": 0.359625, "W1": 0.448347, "C3_long": 0.192028} | 0.414 | 0.8025 | 0.7910 | 0.7976 | 0.7844 ± 0.0207 |
| ENS08 | C3+W1+C3_long | B | {"C3": 0.416855, "W1": 0.286916, "C3_long": 0.296229} | 0.401 | 0.7897 | 0.7950 | 0.7920 | 0.7844 ± 0.0207 |
| SINGLE_C3 | C3 | A | {"C3": 1.0} | 0.295 | 0.7856 | 0.7833 | 0.7846 | 0.7769 ± 0.0268 |
| SINGLE_C3 | C3 | B | {"C3": 1.0} | 0.299 | 0.7846 | 0.7860 | 0.7853 | 0.7769 ± 0.0268 |
| ENS04 | C3+C3_long | A | {"C3": 0.756157, "C3_long": 0.243843} | 0.387 | 0.7946 | 0.7811 | 0.7888 | 0.7755 ± 0.0190 |
| ENS04 | C3+C3_long | B | {"C3": 0.671885, "C3_long": 0.328115} | 0.363 | 0.7827 | 0.7922 | 0.7869 | 0.7755 ± 0.0190 |
| SINGLE_C3_long | C3_long | A | {"C3_long": 1.0} | 0.52 | 0.7774 | 0.7552 | 0.7679 | 0.7711 ± 0.0234 |
| SINGLE_C3_long | C3_long | B | {"C3_long": 1.0} | 0.479 | 0.7704 | 0.7686 | 0.7697 | 0.7711 ± 0.0234 |
| SINGLE_E8_long_A | E8_long_A | A | {"E8_long_A": 1.0} | 0.469 | 0.7828 | 0.7580 | 0.7721 | 0.7710 ± 0.0247 |
| SINGLE_E8_long_A | E8_long_A | B | {"E8_long_A": 1.0} | 0.367 | 0.7730 | 0.7679 | 0.7708 | 0.7710 ± 0.0247 |
| SINGLE_W1 | W1 | A | {"W1": 1.0} | 0.38 | 0.7666 | 0.7467 | 0.7580 | 0.7576 ± 0.0250 |
| SINGLE_W1 | W1 | B | {"W1": 1.0} | 0.357 | 0.7479 | 0.7535 | 0.7503 | 0.7576 ± 0.0250 |

## Shortlist (by the spec's criteria)
- Best single model (held-out Track A): SINGLE_E8 — held-out 0.7910
- Best Track A in-split: ENS10 — dev_excl_test 0.8179, held-out 0.7912
- Best robust (held-out Track A): ENS07 — 0.8045 ± 0.0218
- Best Track B (OVERLAP-INFORMED): ENS05 — dev_in_test 0.8082

Submissions: {
 "best_single_A": {
  "id": "SINGLE_E8",
  "track": "A",
  "zip": "/home/aarsh_kaushik/capstone/phase4/ensembles/dual_split_sweep/best_single_A_SINGLE_E8/submission_best_single_A_SINGLE_E8.zip"
 },
 "best_track_A": {
  "id": "ENS10",
  "track": "A",
  "zip": "/home/aarsh_kaushik/capstone/phase4/ensembles/dual_split_sweep/best_track_A_ENS10/submission_best_track_A_ENS10.zip"
 },
 "best_robust_A": {
  "id": "ENS07",
  "track": "A",
  "zip": "/home/aarsh_kaushik/capstone/phase4/ensembles/dual_split_sweep/best_robust_A_ENS07/submission_best_robust_A_ENS07.zip"
 },
 "best_track_B_overlap": {
  "id": "ENS05",
  "track": "B",
  "zip": "/home/aarsh_kaushik/capstone/phase4/ensembles/dual_split_sweep/best_track_B_overlap_ENS05/submission_best_track_B_overlap_ENS05.zip"
 }
}

Interpretation / final choice: yours (course rule 9).
