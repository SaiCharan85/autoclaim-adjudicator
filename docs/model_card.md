# Model card: auto claim fraud scorer

Production version `catboost-4f7d8bfa06ea` · trained 2026-10-04.
Generated reports: [`fraud_benchmark_sim_us.md`](fraud_benchmark_sim_us.md) (production) and
[`fraud_benchmark_legacy_1990s.md`](fraud_benchmark_legacy_1990s.md) (reality check).
Test-set touches: [`test_set_log.md`](test_set_log.md). Protocol: [`evaluation_protocol.md`](evaluation_protocol.md).

## What it is
A CatBoost classifier that ranks auto physical-damage claims by fraud likelihood, plus two
deterministic companions: an Isolation Forest anomaly score and a red-flag rules engine.
Together they are the **fraud tools** the harness's fraud node calls. Tool outputs are the source
of truth for every fraud number.

| | |
|---|---|
| Algorithm | CatBoost, depth 4, L2 10, learning rate 0.05; 1,346 trees (early stopping); native categoricals |
| Features | 41 first-notice-of-loss features (leakage-tagged, `lines/auto/simulator/columns.py`) |
| Training data | 41,603 grounded-hybrid claims, 2022 to H1 2024 ([docs/simulator.md](simulator.md)) |
| Locked test | 8,397 claims, H2 2024, plus two 8,000-claim holdouts (fresh seed; perturbed world) |
| Reality check | Real 1994–96 fraud labels (15,420 claims): same pipeline, separate model, never shipped |
| Explanations | Exact TreeSHAP: top-5 features per claim, in log-odds |

## Intended use
- **Use:** rank claims so a limited fraud team (SIU) reviews the riskiest first. In the harness, a
  high score can **only escalate a claim to a human**. It never denies one.
- **Don't use:** to deny claims, price policies, or judge individuals without review. Above all,
  **don't quote the simulator metrics as real-world fraud-detection performance**: the fraud
  labels are simulated.

## Results

### Production (grounded-hybrid simulator)
| Evaluation | PR-AUC | recall@5% | precision@5% | base rate |
|---|---|---|---|---|
| Validation, CatBoost (3 seeds) | 0.419 ± 0.002 | 0.414 ± 0.003 | | 6.4% |
| **Locked test, production model** | **0.409** | **0.398** | **0.498** | 6.25% |
| Fresh-seed holdout | 0.464 | 0.448 | 0.550 | 6.1% |
| **Shifted-world holdout** | **0.404** | **0.409** | 0.463 | 5.7% |

In plain terms: reviewing the top 5% of claims catches about 40% of fraud, and half of the reviewed
claims are fraudulent (8× random picking). Net of a $300 review cost, that is about $312k of
fraudulent claims stopped per 1,000 claims (simulated amounts). Performance holds on the shifted
world, so the model is not just memorizing simulator quirks.

Model comparison (validation, where decisions are made):

| Model | PR-AUC | recall@5% | Δ recall vs CatBoost [95% CI] | train − val PR-AUC |
|---|---|---|---|---|
| Logistic regression | 0.287 | 0.313 | −0.108 [−0.139, −0.069] | 0.002 |
| XGBoost | 0.398 | 0.410 | −0.002 [−0.025, +0.020] | 0.317 (flagged) |
| LightGBM | 0.425 | 0.419 | +0.004 [−0.016, +0.024] | 0.229 (flagged) |
| **CatBoost (regularized)** | 0.419 | 0.414 | (reference) | **0.121** |
| EBM | 0.416 | 0.411 | −0.008 [−0.027, +0.016] | 0.143 |

### Reality check (real 1994–96 labels)
| | PR-AUC | recall@5% | base rate |
|---|---|---|---|
| Validation, CatBoost (3 seeds) | 0.179 | 0.098 | ~7% |
| Locked test 1996, CatBoost | 0.112 | 0.103 | 5.2% |
| Locked test 1996, LightGBM | 0.154 | 0.138 | 5.2% |

**Validation and test disagree on the real data.** CatBoost won validation, but on 1996 LightGBM
is better (+0.066 recall [+0.013, +0.112]). We do **not** switch on that basis, because choosing on
the test set is the leakage this protocol forbids. The honest reading: with about 200 real frauds
per year and a drifting fraud rate, model rankings on this data don't carry from one period to the
next. The real data is also much harder than the simulator (PR-AUC about 0.11–0.15 vs 0.41), which
is a reminder of the simulator's limits.

## Two-stage triage: catching TRUE fraud (user target: >= 80%)
First-notice facts alone cannot separate inflated repair estimates (79% of fraud) from normal cost
variation: the same visible damage varies about ±60–80% in true cost, while inflation is ×1.3–2.5.
Reaching 80% of true fraud on stage 1 alone takes reviewing ~27% of claims. Insurers solve this
the way we now do: an **independent appraisal** before payment, then an estimate audit.

- **Stage 1 (first notice):** the CatBoost above; top 5% -> fraud team (SIU).
- **Stage 2 (after appraisal):** CatBoost on first-notice + appraisal features (appraised amount,
  log(claimed / appraised), appraiser noted prior damage). A new `appraisal` availability stage;
  stage 1 stays first-notice only (tests enforce both). Canaries pass.
- **Threshold** fitted on validation for 80% true-fraud recall at the fewest referrals, frozen in
  config, then the locked test run once ([`fraud_triage_validation.md`](fraud_triage_validation.md),
  [`fraud_triage_test.md`](fraud_triage_test.md)).

Re-fit 2026-10-04 after `appraisal.shop_sigma` (honest shop-estimate error) was added: the
appraisal-gap feature alone had ROC-AUC 0.904, over the 0.9 single-feature canary, because honest
shop estimates matched the appraisal too closely. It is now 0.867 (train) / 0.874 (validation), and
the canary passes. The locked test was re-run once on the new data (second look, logged in
[`test_set_log.md`](test_set_log.md); justified because the data changed, not the model choice).

| Measured on TRUE fraud | Validation | **Locked test** | Locked test, old data (superseded) |
|---|---|---|---|
| Recall (intercepted by either stage) | 0.796 [0.767, 0.826] | **0.783** | 0.785 |
| Review rate (stage 1 + stage 2) | 0.095 (0.05 + 0.045) | **0.100** (0.05 + 0.05) | 0.087 |
| Precision of referrals | 0.643 | 0.615 | 0.768 |
| Fraud dollars intercepted | 0.881 | 0.897 | 0.888 |
| ROC-AUC stage 1 / stage 2 | 0.877 / 0.962 | **0.860 / 0.955** | 0.864 / 0.973 |
| Recall by type: inflated / prior damage / staged / give-up | 0.81 / 0.74 / 0.79 / 0.87 | 0.78 / 0.76 / 0.74 / 1.00 | 0.79 / 0.72 / 0.53 / 1.00 |

Stage-2 threshold 0.1745 (was 0.3247). Rolling-origin CV before the locked test agrees: 6 quarterly
folds, recall 0.80 ± 0.04 at review 10.6% ([`fraud_triage_rolling.md`](fraud_triage_rolling.md));
3 yearly folds on the 2002-2024 world, 0.80 ± 0.02 at 10.1%
([`fraud_triage_rolling_decades.md`](fraud_triage_rolling_decades.md)). Recency weighting changes
nothing measurable. Drift: training on 2002-2009 and testing on 2020-2023 loses only ~0.005 ROC-AUC
per stage ([`drift_study.md`](drift_study.md)), because the simulated fraud mechanism is stationary;
real fraud drifts, so this is an upper bound on stability. Staged collisions stay hard and few
(n = 19). The appraisal is simulated: true cost x lognormal noise, spread hypot(0.20 appraiser,
0.18 shop) ~ 0.27, slight under-estimate. That is an assumption to calibrate if real
appraisal-vs-final-cost data becomes available.

## Why CatBoost, and why regularized
- **Rule (declared before the final run):** ship CatBoost unless another model beats it on
  validation recall@5% with a paired 95% CI excluding zero. None did. The tree models are tied;
  only logistic regression is clearly worse.
- **Overfitting check:** the depth-6 CatBoost had a train-vs-validation PR-AUC gap of 0.20, above
  the 0.15 flag. A pre-declared rule: adopt depth 4 + L2 10 if it is not significantly worse on
  validation and its gap drops below 0.15. Result: recall +0.012 [−0.002, +0.023], gap 0.12, so it
  was adopted. XGBoost and LightGBM still show large gaps with their settings, one more reason not
  to ship them.

## Pre-declared rule: improvement levers (written 2026-10-04, before any lever was run)
Levers: the repair-cost residual feature, amount-weighted training, a CatBoost + LightGBM + EBM
rank-averaged ensemble, and a 4-point CatBoost depth/L2 grid. All variants run on validation
only, with 3 seeds; deltas are paired bootstraps on the seed-averaged scores.
1. **Repair-cost residual** is adopted if validation recall@5% vs the current model improves with
   a 95% CI excluding zero, and the train − validation PR-AUC gap stays below 0.15.
2. The other levers are built on top of the repair-cost residual, so they are judged **against
   "+ repair_cost"**. If (1) fails, they are judged against the current model instead.
   - Ensemble and grid: adopted only if recall@5% improves with a CI excluding zero (grid: gap < 0.15).
   - Amount weighting: adopted only if net savings per 1,000 claims improves with a CI excluding
     zero **and** recall@5% is not significantly worse.
3. Ties go to the simpler model (fewer models, current depth 4 / L2 10).
4. The grid tests 4 settings, so one lucky win is possible; a grid win must also hold on all
   3 seeds' means, or it is reported but not adopted.
5. The locked test is run once, after the decision, and logged.

## Protocol history (every change, with its reason)
| When | Change | Why |
|---|---|---|
| Step 2 | Early stopping: mixed metrics → log-loss → ROC-AUC | Log-loss stopped models at ~12 trees under base-rate drift (diagnosed defect) |
| Step 2b | Production data: 1990s Kaggle → grounded hybrid (real CRSS 2022–24 + simulated labels) | User requirement: recent, US data. No real recent fraud labels exist publicly |
| Step 2b | Shuffled-label canary: 1 permutation → mean of 5, one-sided | A single permutation false-alarmed on the real data (runs 0.617, 0.476, 0.463, 0.518, 0.514): strong features make one shuffle's random model align with labels by luck |
| Step 2b | CatBoost depth 6 → 4, L2 3 → 10 | Train-validation gap above 0.15; equal validation performance |
| Step 2b | "No address change" encoded as a known value (10,000 days) instead of missing | Found in the rule report: missing values sort as *smallest* in CatBoost, so "never moved" looked like "moved recently". Fixed in code, then the locked test was re-run (2nd entry in the test log) |
| Backlog | Tested 4 levers on validation under the pre-declared rule above: repair-cost residual, amount weighting, CatBoost+LightGBM+EBM ensemble, depth/L2 grid ([`fraud_experiments_sim_us.md`](fraud_experiments_sim_us.md)) | **None adopted.** Repair-cost residual +0.002 recall@5% [−0.010, +0.028]; amount weighting significantly worse than the residual alone (−0.030 [−0.057, −0.006]); ensemble and grid within noise. The production model is unchanged. The residual may still help the 79% "inflated damage" fraud once real repair-cost data exists |

The 1996 test set was used 3 times during Step 2, before this protocol existed. Since Step 2b, each
use is logged. That's 2 touches per dataset, the second after a correctness fix with no model or
protocol change.

## What drives the score
CatBoost importance:
- claim amount relative to vehicle value: 24%;
- prior claims in 3 years: 13%;
- claimed amount: 10%;
- cause: 7%;
- damage extent: 6%;
- deductible: 5%;
- police-report delay: 4%;
- towed: 4%;
- days from policy start to loss: 4%.

In plain English: a large claim for the car's value, several recent claims, and losses soon after
the policy started raise the score. These mirror the simulator's planted mechanisms, by
construction.

## Fairness
- The simulator generates no sex or marital status. On the real data, `Sex` and `MaritalStatus` are
  excluded. Ablation with the regularized model: **no measurable benefit** from adding them
  (validation +0.006 [−0.013, +0.024]; test +0.023 [−0.019, +0.056]). In Step 2 (depth 6) they were
  significantly worse. Either way, excluding them costs nothing measurable.
- Still present: driver age (age can be a regulated attribute) and possible proxies (vehicle make,
  state). A deployment would need a per-group error audit.

## Isolation Forest
It is weak as a fraud signal: ROC-AUC 0.59 on the simulator and 0.45 (worse than random) on the
real data, where fraud looks more ordinary than average. **Recommended role:** an
out-of-distribution check. A high anomaly score means "unlike the training claims, trust the model
score less, prefer a human". It should not raise fraud risk. (Decision for Step 5.)

## Red-flag rules
On **real 1990s labels** (rules the old data can express), most flags hold up:

| Rule | Lift |
|---|---|
| loss soon after policy start | 1.65× |
| claim soon after policy start | 3.4× |
| isolated unverified loss | 1.6× |
| delayed reporting | 1.4× |
| unverified at-fault loss | 1.3× (fires on 71%, non-specific) |
| recent address change | 12× (only 4 claims) |

On the **simulator**, lifts are higher (5–12× for timing, address and financed-theft flags), but
**that's circular**: the same author wrote the rules and the fraud mechanisms. The out-of-state rule
shows no lift (1.0) because no mechanism plants it. That's an honest gap, not a bug.

## Limitations
- **Simulated labels.** The production model learns our fraud mechanisms. Only the 1990s data has
  real labels, and that data is old, coarse and harder.
- At fault is a police-violation proxy (under-counts). The state is sampled within the real region.
- Real crash records end in December 2024 (NHTSA publication lag).
- Single dataset build (seed 42). Seeds vary the models, not the world.
