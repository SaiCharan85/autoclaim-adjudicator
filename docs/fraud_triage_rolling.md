# Two-stage triage: rolling-origin CV (quarterly, before the locked test)

Each fold fits both stages on the past, fits the stage-2 threshold on the previous quarter with CONFIRMED labels, and scores the next quarter against TRUE fraud.

| half_life_days | n_test_mean | n_test_std | true_recall_mean | true_recall_std | confirmed_recall_mean | confirmed_recall_std | review_rate_mean | review_rate_std | precision_true_mean | precision_true_std | auc_stage1_true_mean | auc_stage1_true_std | auc_stage2_true_mean | auc_stage2_true_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 365 | 4177.5 | 129.941 | 0.804 | 0.046 | 0.804 | 0.056 | 0.109 | 0.024 | 0.602 | 0.057 | 0.868 | 0.013 | 0.958 | 0.004 |
| none | 4177.5 | 129.941 | 0.799 | 0.041 | 0.803 | 0.052 | 0.106 | 0.022 | 0.612 | 0.055 | 0.869 | 0.013 | 0.959 | 0.003 |

| test_quarter | half_life_days | n_test | true_recall | confirmed_recall | review_rate | precision_true | auc_stage1_true | auc_stage2_true |
|---|---|---|---|---|---|---|---|---|
| 2023-01-01 | none | 4278 | 0.869 | 0.886 | 0.15 | 0.507 | 0.866 | 0.958 |
| 2023-04-01 | none | 4059 | 0.767 | 0.765 | 0.097 | 0.628 | 0.855 | 0.954 |
| 2023-07-01 | none | 4058 | 0.806 | 0.82 | 0.102 | 0.631 | 0.866 | 0.96 |
| 2023-10-01 | none | 4386 | 0.77 | 0.76 | 0.093 | 0.66 | 0.874 | 0.956 |
| 2024-01-01 | none | 4144 | 0.817 | 0.828 | 0.104 | 0.601 | 0.893 | 0.963 |
| 2024-04-01 | none | 4140 | 0.763 | 0.757 | 0.092 | 0.641 | 0.861 | 0.96 |
| 2023-01-01 | 365 | 4278 | 0.882 | 0.896 | 0.157 | 0.491 | 0.868 | 0.958 |
| 2023-04-01 | 365 | 4059 | 0.77 | 0.761 | 0.098 | 0.623 | 0.851 | 0.953 |
| 2023-07-01 | 365 | 4058 | 0.806 | 0.816 | 0.104 | 0.619 | 0.863 | 0.96 |
| 2023-10-01 | 365 | 4386 | 0.773 | 0.76 | 0.095 | 0.644 | 0.872 | 0.956 |
| 2024-01-01 | 365 | 4144 | 0.826 | 0.837 | 0.106 | 0.595 | 0.889 | 0.963 |
| 2024-04-01 | 365 | 4140 | 0.763 | 0.757 | 0.093 | 0.64 | 0.862 | 0.96 |
