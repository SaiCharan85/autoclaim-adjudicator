# Two-stage fraud triage: validation (H1 2024): threshold fitted here

Models: stage1 catboost-ee3e3e0c1f33, stage2 catboost-c2069417eb94. Stage 1 refers the top 5% at first notice; stage 2 refers claims scoring >= 0.1745 after the appraisal. Models train on confirmed fraud; recall is measured on TRUE fraud.

## Policy (interception)
| measured on | stage1_review_rate | stage2_review_rate | review_rate | recall | recall_stage1 | precision | dollar_recall |
|---|---|---|---|---|---|---|---|
| true fraud | 0.05 | 0.045 | 0.095 | 0.796 | 0.462 | 0.643 | 0.881 |
| confirmed fraud | 0.05 | 0.045 | 0.095 | 0.802 | 0.454 | 0.476 | 0.894 |

| bootstrap | stage1_review_rate | stage2_review_rate | review_rate | recall | recall_stage1 | precision | dollar_recall |
|---|---|---|---|---|---|---|---|
| true fraud 95% CI | 0.050 - 0.050 | 0.040 - 0.050 | 0.091 - 0.100 | 0.767 - 0.826 | 0.431 - 0.493 | 0.607 - 0.678 | 0.836 - 0.914 |

## By fraud type
| fraud type | n | stage1_recall | two_stage_recall |
|---|---|---|---|
| inflated_damage | 509.0 | 0.452 | 0.806 |
| prior_damage | 95.0 | 0.442 | 0.737 |
| staged_collision | 19.0 | 0.632 | 0.789 |
| owner_give_up | 15.0 | 0.733 | 0.867 |

## Each stage alone
| model | roc_auc_true | roc_auc_confirmed | review_share_for_80pct_true |
|---|---|---|---|
| stage 1 (first notice) | 0.877 | 0.866 | 0.265 |
| stage 2 (after appraisal) | 0.962 | 0.953 | 0.093 |

### Stage 1 recall curve (true fraud)
| budget | recall | precision | dollar_recall |
|---|---|---|---|
| 0.05 | 0.462 | 0.711 | 0.677 |
| 0.1 | 0.613 | 0.472 | 0.792 |
| 0.15 | 0.699 | 0.359 | 0.844 |
| 0.2 | 0.759 | 0.292 | 0.881 |
| 0.3 | 0.828 | 0.212 | 0.912 |

### Stage 2 recall curve (true fraud)
| budget | recall | precision | dollar_recall |
|---|---|---|---|
| 0.05 | 0.58 | 0.892 | 0.723 |
| 0.1 | 0.81 | 0.624 | 0.88 |
| 0.15 | 0.886 | 0.455 | 0.947 |
| 0.2 | 0.926 | 0.357 | 0.964 |
| 0.3 | 0.964 | 0.247 | 0.987 |
