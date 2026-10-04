# Model card: auto claim fraud scorer

Version `catboost-16b27ae138b7` · trained 2026-10-04 · numbers from
[`fraud_benchmark.md`](fraud_benchmark.md) (regenerate with `uv run python scripts/train_fraud.py`).

## What it is
A CatBoost classifier that ranks auto physical-damage claims by fraud likelihood, plus two
deterministic companions: an Isolation Forest anomaly score and a red-flag rules engine
(`src/autoclaim/lines/auto/fraud_rules.yaml`). Together they make up the **fraud tools** that the
harness's fraud node calls. Tool outputs are the source of truth for every fraud number.

| | |
|---|---|
| Algorithm | CatBoost, 28 trees, depth 6, learning rate 0.05, native categoricals |
| Features | 27 (25 source columns + derived `claim_lag_weeks`; listed in `models/fraud/metadata.json`) |
| Training data | 11,337 claims from 1994–95 ([data notes](data_notes.md)), sha256 `8b6aa597…` |
| Held-out test | 4,083 claims from 1996 (~213 frauds), never used for training or model selection |
| Explanations | Exact TreeSHAP from CatBoost: top-5 features per claim, in log-odds |

## Intended use
- **Use:** rank claims so a limited fraud-review team (SIU) looks at the riskiest ones first.
  In the harness, a high score can **only escalate a claim to a human**. It never denies one.
- **Don't use:** to deny claims, set prices, or judge individuals without review. A score is a
  reason to look closer, not evidence.

## How it was evaluated
- **Time split:** train on 1994–95, test on 1996. The fraud rate drifts down (6.7% → 5.8% → 5.2%),
  and for every model random CV scored 0.06–0.10 PR-AUC higher than the forward-in-time test.
  Random CV would overstate production performance.
- **Headline metric: recall in the top 5% of claims** (the review budget, `carrier_config.yaml`).
  Also PR-AUC (base rate ≈ 0.05). Accuracy is meaningless here.
- **Uncertainty:** 1996 has only ~213 frauds, so every 1996 number has a bootstrap 95% CI. Model
  differences use a *paired* bootstrap (same resampled claims for both models).
- **Early stopping:** the tree count is chosen on the latest 20% of the training period (never 1996),
  by ROC-AUC. See "Protocol history" for why.

## Results (1996 test set)
| Model | PR-AUC | recall@5% [95% CI] | Δ recall vs CatBoost [95% CI] | PR-AUC, 5-fold CV |
|---|---|---|---|---|
| Logistic regression | 0.094 | 0.056 [0.032, 0.094] | −0.070 [−0.108, −0.015] | 0.158 ± 0.011 |
| XGBoost | 0.163 | 0.127 [0.091, 0.175] | +0.000 [−0.043, +0.055] | 0.235 ± 0.024 |
| LightGBM | 0.161 | 0.146 [0.101, 0.193] | +0.019 [−0.023, +0.063] | 0.227 ± 0.030 |
| **CatBoost** | 0.121 | 0.127 [0.082, 0.169] | (reference) | 0.220 ± 0.019 |
| EBM | 0.149 | 0.099 [0.064, 0.138] | −0.028 [−0.067, +0.019] | 0.224 ± 0.010 |

The saved production model (refit on all of 1994–95) scores PR-AUC 0.118 and recall@5% 0.117 on
1996. In plain terms: reviewing the top 5% of claims (~205) catches about 1 in 8 frauds, at about
2.3× the base rate. That's a modest signal. The table only has coarse, pre-binned fields
and no claim amounts or text; the harness adds rules and narrative evidence on top.

### Why CatBoost
The selection rule was declared **before** the final run: ship another model only if it beats
CatBoost on 1996 recall@5% with a paired 95% CI that excludes zero; otherwise ship CatBoost for
native categorical handling and exact built-in SHAP. No model met that bar. The honest summary
is that **the four tree models are statistically tied on this data**, and only logistic regression
is clearly worse. CatBoost was not shown to be best.

### Protocol history (all runs reported)
| Run | Early-stopping metric | What happened | Change made |
|---|---|---|---|
| 1 | Mixed (PR-AUC / aucpr / LightGBM default) | Inconsistent across models; no CIs | Same metric for all; add bootstrap CIs |
| 2 | Log-loss | XGBoost/LightGBM stopped at **12–13 trees**; the rule would have picked them (Δ +0.052, CI excl. 0) | Diagnosed: log-loss also scores calibration, and the validation slice has a lower fraud rate (drift), so it worsens almost at once |
| 3 | ROC-AUC (rank-based, base-rate invariant) | Final table above; no significant differences | — |

CatBoost's own 1996 recall moved between 0.094 and 0.127 depending only on the stopping metric.
That sensitivity is itself a result: **differences between these models are within the noise
of this dataset**. Every protocol change was justified by a diagnosed defect, not by the outcome.

## What drives the score
CatBoost importance: `Fault` 61%, `BasePolicy` 25%, then `Deductible`, `Age`, `DriverRating`,
claim timing and `AddressChange_Claim` (~1–3% each). In plain English: the model mostly learns
"the policyholder caused the accident, under a policy that pays for their own car". Liability-only
claims are almost never fraud here (0.7%). Per-claim SHAP reasons are attached to every assessment.

## Fairness
- `Sex` and `MaritalStatus` are **excluded** (several US states restrict their use in insurance).
- Ablation: adding them back makes 1996 recall **significantly worse**
  (Δ −0.056 [−0.091, −0.014]), and CV is flat (0.234 vs 0.220 PR-AUC, within noise). Excluding them
  costs nothing measurable and generalizes better forward in time.
- Still present: `Age` and `AgeOfPolicyHolder` (age can be a regulated attribute), plus possible
  proxies (`VehicleCategory`, `Make`). Their importance is small (Age 2.3%; Make and
  VehicleCategory ≈ 0%), but a production deployment would need a per-group error audit.

## Isolation Forest: not a fraud signal here
On 1996 it ranks fraud **worse than random** (ROC-AUC 0.44, PR-AUC 0.047 vs 0.052 base rate):
fraudulent claims in this data look *more* typical than average, which fits fraud that is
designed to look ordinary. Recommended role: an **out-of-distribution check**. A high anomaly score
means "this claim is unlike the training data, so trust the model score less", which the router
can use to prefer human review. It should not raise fraud risk. (Decision for Step 5.)

Implementation notes: categoricals are **frequency-encoded**, so unseen or rare values look
anomalous. One-hot with `handle_unknown="ignore"` would make unseen values look *typical*;
a unit test pins this. Known limitation: values beyond the training range (e.g. `Age = 99`) score
like the training extremes, because the forest's split points are drawn within the range it saw.

## Red-flag rules
Seven rules, written in our own words. On all years, lift 1.3–2.3 for most and 12.5 for
`recent_address_change`, but that rule fires on only 4 claims. Caveats, reported rather than tuned
away (re-tuning on 1996 would leak the test set):
- `unverified_at_fault_loss` fires on **70%** of claims, because this dataset records a police
  report on only 2.8% of claims. It's non-specific here; synthetic narratives (Step 4) will carry
  realistic police-report rates.
- `isolated_unverified_loss` (lift 0.85) and `delayed_reporting` (1.01) show no lift on 1996.
  `claim_soon_after_inception` fires on just 8 test claims.

## Limitations
- Small positive count (~213 test frauds): wide intervals, single training seed.
- Mid-1990s data with documented quirks (PolicyType/VehicleCategory mismatch, Age = 0 encoding).
- No claim amounts, free text or exact dates; the score is a weak, coarse signal on its own.
- Synthetic claims (Step 4) are generated from these rows. **Evaluation claims come only from
  1996 rows**, so the fraud score never sees its own training data during harness evaluation.
