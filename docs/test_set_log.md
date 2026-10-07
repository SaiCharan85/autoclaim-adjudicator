# Locked test-set log

Every evaluation on a locked test set is appended here automatically (`scripts/train_fraud.py --final`). Model and protocol decisions use validation data only.

| when (UTC) | dataset | code | headline |
|---|---|---|---|
| 2026-10-04 07:53 | sim_us | 48695c1+dirty | catboost test PR-AUC 0.4022, recall@5% 0.4000 |
| 2026-10-04 07:56 | legacy_1990s | 48695c1+dirty | catboost test PR-AUC 0.1122, recall@5% 0.1080 |
| 2026-10-04 08:18 | sim_us | 48695c1+dirty | catboost test PR-AUC 0.4092, recall@5% 0.3981 |
| 2026-10-04 08:21 | legacy_1990s | 48695c1+dirty | catboost test PR-AUC 0.1122, recall@5% 0.1080 |
| 2026-10-04 22:02 | sim_us (two-stage triage) | 5ba15fc+dirty | true-fraud recall 0.7850 at review rate 0.0870 (stage-2 threshold 0.3247) |
| 2026-10-05 06:03 | sim_us (two-stage triage) | 5ba15fc+dirty | true-fraud recall 0.7830 at review rate 0.1000 (stage-2 threshold 0.1745) |
| 2026-10-05 17:03 | flood (FEMA NFIP, losses >= 2024-07-01) | 7cdc800+dirty | auto_rate 0.414, auto_agreement 0.986, n=3000 |
| 2026-10-06 16:43 | sim_us | 60f90e1+dirty | fraud LLM benchmark n=200: ml ROC-AUC 0.855, llm ROC-AUC 0.631, hybrid ROC-AUC 0.853 |
| 2026-10-07 03:18 | sim_us | 60f90e1+dirty | fraud LLM benchmark n=200: ml ROC-AUC 0.855, llm ROC-AUC 0.631, hybrid ROC-AUC 0.853 |
| 2026-10-07 03:25 | note | - | The 03:18 fraud LLM benchmark row is an accidental re-run of the 16:43 run (a resumed pipeline still held its --final step). All calls came from the exact-match cache with the same seed: identical numbers, no new information, no decision changed. One real look at this test. |
