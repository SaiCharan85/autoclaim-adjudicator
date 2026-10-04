# Evaluation protocol

How this project avoids data leakage and overfitting, and how it would be A/B-tested and tested on
real claims. Rules marked **[enforced]** are checked by code or tests; the rest are procedure.

## 1. Data leakage

| Leak | Guard |
|---|---|
| **Target leakage**: a feature that encodes the answer | Every column is tagged by when it becomes known (`lines/auto/simulator/columns.py`). Only first-notice-of-loss (FNOL) columns can be features. **[enforced]** |
| | The single-feature canary flags any feature that alone predicts the label with ROC-AUC > 0.9. **[enforced, every run]** |
| **Pipeline leakage**: information flowing through preprocessing | Shuffled-label canary: train on permuted labels; validation ROC-AUC must sit within chance (0.5 ± 4 SE). **[enforced, every run]** |
| | Encoders, vocabularies, scalers and the anomaly reference are fit on training data only. **[tested]** |
| **Time leakage**: training on the future | Splits are strictly time-ordered (train < validation < test). `assert_time_ordered` runs inside every benchmark and training call. **[enforced]** |
| **Record leakage**: one event in two splits | Each real crash record is used at most once; holdouts exclude every record the main set used. **[tested]** |
| **Ground-truth leakage**: the harness seeing the answer | The coverage oracle and `gt_*` columns are evaluation-only. The harness must not import the oracle (import test added in Step 5). |
| **Test-set reuse**: tuning on the test set | 3-way split. All decisions use validation. The locked test set is scored only with `--final`, and each use is appended with the code version to `docs/test_set_log.md`. **[enforced + logged]** |

Harness-level rules (Steps 4–8):
- Evaluation claims come from the test window only.
- Few-shot examples, feedback memory and fine-tuning data come from train and validation only.
- Prompts are iterated on a dev slice, never the test slice.

## 2. Overfitting
- **Early stopping** on a slice of the training period (never validation or test), using ROC-AUC.
- **Train-vs-validation gap**: every benchmark row reports `train PR-AUC − validation PR-AUC` and
  flags a gap above 0.15.
- **Seeds**: every metric is mean ± std over 3 seeds; confidence intervals use the seed-averaged
  predictions.
- **Simulator overfitting**: the *shift* holdout comes from a perturbed world. A model that learned
  simulator quirks instead of fraud degrades there. The real 1994–96 labels are the final reality
  check.
- **Pre-registered decisions**: the selection rule is written down before results are seen. Every
  protocol change is recorded with its diagnosed reason (see the model card's protocol history).

## 3. A/B testing (comparing two versions of the system)
For example "with vs without the judge" or "prompt v1 vs v2":
1. **Same claims for both arms.** A paired comparison removes "which claims were drawn" noise.
2. **The metric is fixed before running.** E.g. cost-weighted error, or wrongly-approved fraud rate.
3. **Paired bootstrap 95% CI** on the difference (`ml/metrics.paired_bootstrap_diff`). A
   difference counts only if the CI excludes zero.
4. **No peeking.** Decide the sample size, run once, then read the result. Stopping when the numbers
   look good inflates false positives.
5. **Cost control.** The exact-match LLM cache means arms that share nodes reuse cached calls, so an
   ablation that changes only the judge re-spends only judge calls.

## 4. Real-world testing (shadow mode)
Before acting on live claims, the system would run in **shadow mode**:
1. Each new claim goes to both the human adjuster (who decides, as today) and the system, which
   records its decision without acting.
2. After a few weeks, compare the two:
   - agreement rate;
   - fraud caught by the system that adjusters missed, confirmed by investigators;
   - wrongful approvals the system would have made;
   - escalation volume.
3. **Monitor drift** against the training distribution: anomaly-score distribution, feature
   distributions, fraud-rate trend. Retrain or recalibrate when they move.
4. Only then move to limited autonomy: auto-approve low-value, low-risk claims, keep humans on
   the rest, and audit a random sample of auto-decisions.

This project has no live claims, so shadow mode is documented, not run. The locked test window
(H2 2024) and the holdouts stand in for it.
