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
