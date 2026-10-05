# Drift study: train on one era, test on later eras

Multi-decade world (real GES 2002-2015 / CRSS 2016-2024 crashes, BLS-priced, simulated policy and fraud). Data before the locked test window only. Metrics vs TRUE fraud.

| stage | train | test | years_gap | roc_auc_true | recall_at_5pct_true |
|---|---|---|---|---|---|
| stage 1 (first notice) | 2002-2009 | 2010-2015 | 1 | 0.861 | 0.423 |
| stage 1 (first notice) | 2002-2009 | 2016-2019 | 7 | 0.858 | 0.424 |
| stage 1 (first notice) | 2002-2009 | 2020-2023 | 11 | 0.859 | 0.415 |
| stage 1 (first notice) | 2010-2015 | 2016-2019 | 1 | 0.865 | 0.435 |
| stage 1 (first notice) | 2010-2015 | 2020-2023 | 5 | 0.865 | 0.428 |
| stage 1 (first notice) | 2016-2019 | 2020-2023 | 1 | 0.867 | 0.431 |
| stage 1 (first notice) | 2002-2019 (all past) | 2020-2023 | 1 | 0.877 | 0.441 |
| stage 2 (appraisal) | 2002-2009 | 2010-2015 | 1 | 0.954 | 0.562 |
| stage 2 (appraisal) | 2002-2009 | 2016-2019 | 7 | 0.956 | 0.569 |
| stage 2 (appraisal) | 2002-2009 | 2020-2023 | 11 | 0.957 | 0.559 |
| stage 2 (appraisal) | 2010-2015 | 2016-2019 | 1 | 0.957 | 0.569 |
| stage 2 (appraisal) | 2010-2015 | 2020-2023 | 5 | 0.958 | 0.558 |
| stage 2 (appraisal) | 2016-2019 | 2020-2023 | 1 | 0.96 | 0.569 |
| stage 2 (appraisal) | 2002-2019 (all past) | 2020-2023 | 1 | 0.962 | 0.572 |
