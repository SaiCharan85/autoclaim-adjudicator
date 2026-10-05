# Two-stage triage: rolling-origin CV (quarterly, before the locked test)

Each fold fits both stages on the past, fits the stage-2 threshold on the previous quarter with CONFIRMED labels, and scores the next quarter against TRUE fraud.

| half_life_days | n_test_mean | n_test_std | true_recall_mean | true_recall_std | confirmed_recall_mean | confirmed_recall_std | review_rate_mean | review_rate_std | precision_true_mean | precision_true_std | auc_stage1_true_mean | auc_stage1_true_std | auc_stage2_true_mean | auc_stage2_true_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1460 | 12000.0 | 0.0 | 0.801 | 0.025 | 0.8 | 0.026 | 0.104 | 0.002 | 0.622 | 0.022 | 0.878 | 0.008 | 0.962 | 0.008 |
| none | 12000.0 | 0.0 | 0.804 | 0.019 | 0.802 | 0.022 | 0.101 | 0.001 | 0.639 | 0.02 | 0.883 | 0.006 | 0.963 | 0.007 |

| test_quarter | half_life_days | n_test | true_recall | confirmed_recall | review_rate | precision_true | auc_stage1_true | auc_stage2_true |
|---|---|---|---|---|---|---|---|---|
| 2021-01-01 | none | 12000 | 0.802 | 0.795 | 0.1 | 0.657 | 0.878 | 0.968 |
| 2022-01-01 | none | 12000 | 0.786 | 0.784 | 0.102 | 0.617 | 0.88 | 0.955 |
| 2023-01-01 | none | 12000 | 0.824 | 0.827 | 0.102 | 0.644 | 0.89 | 0.967 |
| 2021-01-01 | 1460 | 12000 | 0.804 | 0.797 | 0.102 | 0.644 | 0.874 | 0.967 |
| 2022-01-01 | 1460 | 12000 | 0.775 | 0.775 | 0.103 | 0.6 | 0.873 | 0.952 |
| 2023-01-01 | 1460 | 12000 | 0.824 | 0.827 | 0.106 | 0.622 | 0.887 | 0.965 |
