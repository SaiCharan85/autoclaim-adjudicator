# Data notes: Vehicle Insurance Claim Fraud Detection

Source: Kaggle `shivamb/vehicle-claim-fraud-detection` (`fraud_oracle.csv`), license **CC0-1.0**
(public domain, as labeled by the uploader; listed origin "Oracle Databases").
Full auto-generated statistics: [`data_profile.md`](data_profile.md). Regenerate with
`uv run python scripts/profile_data.py`.

## At a glance
- **15,420 claims × 33 columns**, 0 schema errors, 0 nulls.
- **Fraud rate 5.99%** (923 frauds). A model that always says "not fraud" is 94% accurate, which is
  why we use PR-AUC (baseline ≈ 0.06) and recall at a fixed review budget (top 5% ≈ 771 claims).
- Almost every column is **pre-binned text**, e.g. `VehiclePrice = "20000 to 29000"` or
  `Days_Policy_Accident = "more than 30"`. Only 9 columns are integers. That suits CatBoost's
  native categorical handling.
- No claim amount, no free text, no exact dates. Narratives and amounts will be synthesized in Step 4,
  conditioned on these rows.

## Findings that change what we build

### 1. Fraud rate drifts down over time → use a time-aware split
| Year | Claims | Fraud rate |
|---|---|---|
| 1994 | 6,142 | 6.66% |
| 1995 | 5,195 | 5.79% |
| 1996 | 4,083 | 5.22% |

A random split would hide this drift. **Plan for Step 2:** train on 1994–95, test on 1996
(4,083 claims, ~213 frauds), and report a stratified 5-fold CV alongside it as a stability check.
Within a year, `Month` + `WeekOfMonth` give a coarse ordering (no exact dates).

### 2. `BasePolicy` decides what is even covered (domain)
In plain English, a US personal auto policy has separate coverages:
- **Liability** pays for damage you cause *to others*. It does **not** fix your own car.
- **Collision** fixes your car after you hit something (another car, a pole).
- **Comprehensive** ("All Perils" here, roughly) also covers non-collision losses: theft, hail,
  fire, **hitting a deer**.

| BasePolicy | Claims | Fraud rate |
|---|---|---|
| All Perils | 4,449 | 10.2% |
| Collision | 5,962 | 7.3% |
| Liability | 5,009 | 0.7% |

**Use in Step 4:** map each row's BasePolicy onto our policy's coverage parts. A Liability-only row
whose narrative describes damage to the policyholder's own car is **not covered under Part D**. A
deer strike on a Collision-only policy is also not covered, because animal strikes fall under
comprehensive (trap 1). These give realistic, label-consistent coverage denials for free.

### 3. `PolicyType` and `VehicleCategory` disagree for 32% of rows (source quirk)
`PolicyType` looks like `"<VehicleCategory> - <BasePolicy>"`, but every one of the 4,987
`"Sedan - Liability"` rows has `VehicleCategory = "Sport"`. So `VehicleCategory = Sport`'s low
fraud rate (1.6%) is really the Liability effect. **Step 2:** treat `PolicyType` as redundant with
`BasePolicy`, keep `VehicleCategory` but don't read its importance as "sports cars are safe", and
document this in the model card.

### 4. `Age = 0` is not random missingness
All 320 rows with `Age = 0` have `AgeOfPolicyHolder = "16 to 17"`, and their fraud rate is 9.7%
(well above average). It's an encoding artifact for the youngest band, not a missing value.
**Step 2:** set `Age` to NaN for these rows (CatBoost handles NaN natively), keep
`AgeOfPolicyHolder`, and let the model learn the band effect. One row has `"0"` for both
`DayOfWeekClaimed` and `MonthClaimed` (unknown claim date); treat it as missing.

### 5. Strong but tiny signals → rules, not just the model
These match well-known fraud red flags directionally, but on small counts:

| Signal | Rows | Fraud rate | Red flag (paraphrased) |
|---|---|---|---|
| `AddressChange_Claim = under 6 months` | 4 | 75% | recent address change before a claim |
| `AddressChange_Claim = 2 to 3 years` | 291 | 17.5% | |
| `Days_Policy_Accident = none` | 55 | 16.4% | loss right after the policy started |
| `Days_Policy_Claim = 8 to 15` | 21 | 14.3% | claim soon after inception |
| `Deductible = 500` | 263 | 17.9% | |
| `Fault = Policy Holder` vs `Third Party` | 11,230 / 4,190 | 7.9% / 0.9% | |

The rare ones (n < 60) are too small for a model to learn reliably, so the **rules engine** in Step 2
will encode the corresponding red flags explicitly, and the synthetic generator will plant them.

### 6. Witnesses and police reports are rare
`PoliceReportFiled = Yes` is only 2.8% of rows and `WitnessPresent = Yes` 0.6%. Trap 5 (hit-and-run
needs a police report) will therefore be driven mostly by the synthetic narratives, not the table.

### 7. Reporting lag is partly recoverable
The claim month differs from the accident month in 25.8% of rows. A coarse **claim-lag feature**
(accident vs. claimed Month/WeekOfMonth, with year wrap) can support trap 6 (late notice violates
Part E duties). It will be built and tested in Step 2.

## Fairness note (decision for Step 2)
`Sex`, `MaritalStatus` and `Age` are in the table, and `Sex` shows a gap (Male 6.3% vs Female 4.3%).
Several US states restrict using gender in insurance decisions. **Recommendation:** exclude `Sex` and
`MaritalStatus` from the production model, and report the PR-AUC cost of dropping them as an ablation.

## Quirks kept as-is
- `Make` has 19 values with source misspellings (`Accura`, `Nisson`, `Porche`, `Mecedes`). Keep them for
  modeling (they're just category labels); normalize spelling only when generating narratives.
- `PolicyNumber` is a unique ID (dropped from features). `RepNumber` (16 values) carries little signal.
- `Deductible` is 96% $400. Our own policy and `carrier_config.yaml` define real deductibles; this
  column is a model feature only.
