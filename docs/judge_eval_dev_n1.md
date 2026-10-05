# Judge planted-error evaluation (dev period) — LIVE SMOKE TEST, 1 sample (wiring check only, not a result)

Clean cases are built from simulator truth (true facts, code-derived numbers, real clause text, ground-truth decision, a templated explanation that is correct by construction). JudgeKit plants one error per copy; a catch is a failed verdict, and a catch by the expected item means the judge failed it for the right reason.

Judge `judge`, rubric `adjudication_reasoning` v1.1, seed 42.

| error type | n | judged | caught | caught by expected item |
|---|---|---|---|---|
| contradicted_conclusion | 1 | 1 | 100.0% | 100.0% |
| invented_fact | 1 | 1 | 100.0% | 100.0% |
| misstated_coverage | 1 | 1 | 100.0% | 100.0% |
| omitted_material_fact | 1 | 1 | 0.0% | 0.0% |
| wrong_amount | 1 | 1 | 100.0% | 100.0% |
| all planted | 5 | 5 | 80.0% | 80.0% |

False alarms on clean samples: 0.0% (0 of 1 judged; 0 unavailable).

## With 95% bootstrap CIs (resampled by claim)

| metric | value | 95% CI |
|---|---|---|
| false_alarm | 0.0% | [0.0%, 0.0%] |
| catch | 80.0% | [80.0%, 80.0%] |
| catch_expected | 80.0% | [80.0%, 80.0%] |
| catch:contradicted_conclusion | 100.0% | [100.0%, 100.0%] |
| catch_expected:contradicted_conclusion | 100.0% | [100.0%, 100.0%] |
| catch:invented_fact | 100.0% | [100.0%, 100.0%] |
| catch_expected:invented_fact | 100.0% | [100.0%, 100.0%] |
| catch:misstated_coverage | 100.0% | [100.0%, 100.0%] |
| catch_expected:misstated_coverage | 100.0% | [100.0%, 100.0%] |
| catch:omitted_material_fact | 0.0% | [0.0%, 0.0%] |
| catch_expected:omitted_material_fact | 0.0% | [0.0%, 0.0%] |
| catch:wrong_amount | 100.0% | [100.0%, 100.0%] |
| catch_expected:wrong_amount | 100.0% | [100.0%, 100.0%] |

## Which rubric items fired

```
{
 "clean": {},
 "contradicted_conclusion": {
  "outcome_consistent": 1
 },
 "invented_fact": {
  "facts_supported": 1
 },
 "misstated_coverage": {
  "faithful_to_clauses": 1
 },
 "omitted_material_fact": {},
 "wrong_amount": {
  "numbers_consistent": 1,
  "outcome_consistent": 1
 }
}
```

