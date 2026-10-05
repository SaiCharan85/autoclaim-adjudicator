# Two-stage fraud triage: LOCKED TEST (H2 2024)

Models: stage1 catboost-5e5590cf3e5e, stage2 catboost-28259c61eaba. Stage 1 refers the top 5% at first notice; stage 2 refers claims scoring >= 0.1745 after the appraisal. Models train on confirmed fraud; recall is measured on TRUE fraud.

## Policy (interception)
| measured on | stage1_review_rate | stage2_review_rate | review_rate | recall | recall_stage1 | precision | dollar_recall |
|---|---|---|---|---|---|---|---|
| true fraud | 0.05 | 0.05 | 0.1 | 0.783 | 0.442 | 0.615 | 0.897 |
| confirmed fraud | 0.05 | 0.05 | 0.1 | 0.777 | 0.436 | 0.475 | 0.88 |

| bootstrap | stage1_review_rate | stage2_review_rate | review_rate | recall | recall_stage1 | precision | dollar_recall |
|---|---|---|---|---|---|---|---|
| true fraud 95% CI | 0.050 - 0.050 | 0.045 - 0.055 | 0.095 - 0.105 | 0.751 - 0.815 | 0.413 - 0.470 | 0.577 - 0.649 | 0.858 - 0.925 |

## By fraud type
| fraud type | n | stage1_recall | two_stage_recall |
|---|---|---|---|
| inflated_damage | 525.0 | 0.392 | 0.783 |
| prior_damage | 98.0 | 0.571 | 0.755 |
| staged_collision | 23.0 | 0.652 | 0.739 |
| owner_give_up | 17.0 | 0.941 | 1.0 |

## Each stage alone
| model | roc_auc_true | roc_auc_confirmed | review_share_for_80pct_true |
|---|---|---|---|
| stage 1 (first notice) | 0.86 | 0.852 | 0.321 |
| stage 2 (after appraisal) | 0.955 | 0.949 | 0.105 |

### Stage 1 recall curve (true fraud)
| budget | recall | precision | dollar_recall |
|---|---|---|---|
| 0.05 | 0.442 | 0.693 | 0.697 |
| 0.1 | 0.561 | 0.44 | 0.755 |
| 0.15 | 0.646 | 0.338 | 0.792 |
| 0.2 | 0.713 | 0.28 | 0.842 |
| 0.3 | 0.787 | 0.206 | 0.889 |

### Stage 2 recall curve (true fraud)
| budget | recall | precision | dollar_recall |
|---|---|---|---|
| 0.05 | 0.575 | 0.901 | 0.758 |
| 0.1 | 0.79 | 0.62 | 0.897 |
| 0.15 | 0.854 | 0.447 | 0.933 |
| 0.2 | 0.9 | 0.353 | 0.957 |
| 0.3 | 0.946 | 0.247 | 0.975 |
