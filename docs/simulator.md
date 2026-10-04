# Grounded-hybrid claims dataset

Why it exists, what is real, what is simulated, and how leakage is prevented.
Build with `uv run python scripts/simulate_claims.py` (about 1 s for 50,000 claims).

## Why not "real recent fraud data"?
No real, recent, fraud-labelled auto-claims data is public anywhere (US, UK, EU, Kaggle and recent
research were all checked). Insurers can't release claim files (privacy law, confidentiality), and
recent papers use private insurer data. The "recent" public sets that do exist failed basic checks:

| Candidate | Problem found |
|---|---|
| Kaggle "Car Insurance Fraud Detection" (2023–25) | every column uniformly random and independent; only 16.7% of rows coherent |
| Kaggle "Insurance Claims Dataset 2026" | fraud label = exactly `Fraud_Risk_Score > 75`; without leaked columns ROC-AUC 0.508 (chance); 75% non-auto |
| Kaggle `insuranceData` bundle (AutoBi etc.) | real but ≤2005, no fraud labels, license unknown |

So the dataset is built from the most real data that exists, with the rest simulated and labelled
as such.

## What is real and what is simulated

| | Real (NHTSA CRSS 2022–2024, public domain) | Simulated (`config/simulator_auto.yaml`) |
|---|---|---|
| Incident | date (month + weekday), hour, region, urban/rural, weather, light, cause, vehicle count, injuries, damage extent, towed, at-fault proxy | the exact day of month; causes police data can't contain (glass, hail, theft, vandalism, mechanical, unreported damage) |
| Vehicle | make, model, model year, body type | value (actual cash value), financed, ADAS |
| Policy | — | state within the real region, coverage, deductibles, endorsements, driver, use at time of loss, start date |
| Claim | — | notice delay, police-report delay, witnesses, attorney, amounts |
| Labels | — | fraud (4 mechanisms), traps, correct decision, reasons, payout |

- **38% of claims are real crash records** (70% of vehicle-vs-vehicle claims, 85% of hit-and-runs,
  31% of deer strikes). The rest are unreported or non-crash claims.
- Every simulated claim **borrows its context** (date, hour, location, weather, light, vehicle)
  from a different real record. A model therefore can't learn "real vs simulated" instead of fraud.
- **Real fields are never modified.** Fraud and traps change only simulated fields. A test checks
  every real-backed claim against its source record field by field.

### CRSS → claim mapping
| CRSS field | Becomes |
|---|---|
| body type | personal passenger vehicles only (cars, SUVs, light pickups, vans) |
| most harmful event | vehicle → `collision_vehicle`; live animal → `animal`; fixed object/rollover → `collision_object`; fire → `fire`. Pedestrian/cyclist crashes excluded (liability, out of scope) |
| another vehicle fled (`HIT_RUN`) | the victims become `hit_and_run`; the fleeing vehicle is not a claimant |
| struck parked vehicle (`parkwork.csv`) | `parked_hit` (a hit-and-run if the striker fled) |
| damage extent + towed | drives the (simulated) repair cost |
| violation charged | `at_fault` (moving violations only). **A proxy that under-counts fault**: police charge a violation in only ~20% of vehicle-vs-vehicle crashes |
| region (4 census regions) | a state within that region (weighted) |
| month + weekday | loss date (day of month drawn to match both) |

## Calibration
Where a concept exists in the real 1994–96 fraud labels, the simulator matches it:

| Check | Simulator | Real 1994–96 |
|---|---|---|
| Confirmed fraud rate | 6.1% (true 8.1%, 75% detected) | 6.0% |
| Fraud, liability-only vs comprehensive | 0.6% vs 8.9% | 0.7% vs 10.2% |
| Loss within 15 days of policy start: fraud vs legit | 12.7% vs 1.2% | ~2–3× fraud lift |

Real-world patterns that come through from the CRSS records:
- 32% of deer strikes fall in Oct–Dec (25% if uniform).
- Kia/Hyundai make up 17% of thefts vs 8.5% of other claims (the 2022–23 theft wave).
- Mean collision repair cost is $5,190, in line with typical public US averages of roughly $5–6k.

## Fraud mechanisms (latent truth)
| Type | Share of fraud | Profile | What changes (simulated fields only) |
|---|---|---|---|
| Inflated damage | 79% | any claim | claimed amount 1.3–2.5× true repair cost |
| Prior damage | 14% | unreported single-vehicle / parked damage | no witness, often late notice, claimed 1.0–1.3× |
| Owner give-up | 5% | theft of a financed car | police report filed 1–4 days late, often late notice |
| Staged collision | 3% | real urban multi-vehicle crash with injuries | attorney involved early, "friendly" witnesses |

Fraud is also likelier soon after the policy starts and after a recent address change. Only 75% of
fraud is confirmed by investigators. **Models train on the confirmed label; evaluations use the
true one.**

## Ground truth (the oracle, `simulator/oracle.py`)
1. **Deny** if the policy can't pay: liability-only, excluded driver, mechanical breakdown,
   business use without endorsement, a comprehensive cause without comprehensive cover, or a
   hit-and-run not reported to police within the jurisdiction's window.
2. Otherwise **escalate** if notice was late (most US states then require the insurer to show the
   delay caused harm, a human judgment) or the claim is fraudulent.
3. Otherwise **approve**: repair cost (or vehicle value for a total loss) minus deductible. A loss
   below the deductible is denied.

The result is 67% approve, 24% deny, 9% escalate. Denials are enriched on purpose: the six traps
are planted on about 1–1.5% of claims each, so evaluations see enough of them. The oracle is
**evaluation-only**: the harness must never import it.

The jurisdiction profile changes the answer. A deer strike on a collision-only policy is denied
under the US profile and paid under the UK/EU profile, where own-damage cover includes it.

## Leakage and overfitting safeguards
- **Time-ordered splits:** train 2022–2023 (33,286), validation H1 2024 (8,317), locked test H2
  2024 (8,397).
- **Each real record is used at most once** (weighted sampling without replacement), so no crash
  can appear in two splits.
- **Holdouts:** two 8,000-claim holdouts from the test window, built only from records the main
  set didn't use:
  - *fresh*: same world, new seed;
  - *shift*: a perturbed world (different fraud mix, smaller inflation, more liability-only
    policies). It tests whether a model learned fraud or simulator quirks.
- **Leakage tags:** every column is tagged `fnol` / `post_fnol` / `ground_truth` / `provenance` /
  `id`, and only `fnol` columns can be features (`simulator/columns.py`).
- **Canaries:** the shuffled-label and single-feature canaries run before every training run.
- **Reproducibility:** the same seed gives a byte-identical dataset across Python processes, which
  a test checks.

## Known limitations
- **Fraud labels are simulated.** The model learns our mechanisms, so the real 1994–96 labels
  remain the reality check.
- **The red-flag rules and the fraud mechanisms were written by the same author**, so simulator
  results can't validate the rules. The real-label rule report can.
- At fault is a police-violation proxy (under-counts). State is sampled within the real region,
  and the day of month is sampled.
- Real crash records end in December 2024 (NHTSA publishes with about a 15-month lag). CRSS 2025
  can be added when released (`datasets/sources.py`).
