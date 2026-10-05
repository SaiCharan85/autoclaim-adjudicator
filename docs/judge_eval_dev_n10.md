# Judge planted-error evaluation (dev period)

Clean cases are built from simulator truth (true facts, code-derived numbers, real clause text, ground-truth decision, a templated explanation that is correct by construction). JudgeKit plants one error per copy; a catch is a failed verdict, and a catch by the expected item means the judge failed it for the right reason.

Judge `judge`, rubric `adjudication_reasoning` v1.1, seed 42.

| error type | n | judged | caught | caught by expected item |
|---|---|---|---|---|
| contradicted_conclusion | 10 | 10 | 90.0% | 90.0% |
| invented_fact | 10 | 10 | 100.0% | 100.0% |
| misstated_coverage | 9 | 7 | 100.0% | 100.0% |
| omitted_material_fact | 10 | 8 | 50.0% | 25.0% |
| wrong_amount | 5 | 5 | 100.0% | 60.0% |
| all planted | 44 | 40 | 87.5% | 77.5% |

False alarms on clean samples: 20.0% (2 of 10 judged; 0 unavailable).

## With 95% bootstrap CIs (resampled by claim)

| metric | value | 95% CI |
|---|---|---|
| false_alarm | 20.0% | [0.0%, 50.0%] |
| catch | 87.5% | [76.7%, 97.2%] |
| catch_expected | 77.5% | [65.0%, 87.8%] |
| catch:contradicted_conclusion | 90.0% | [70.0%, 100.0%] |
| catch_expected:contradicted_conclusion | 90.0% | [70.0%, 100.0%] |
| catch:invented_fact | 100.0% | [100.0%, 100.0%] |
| catch_expected:invented_fact | 100.0% | [100.0%, 100.0%] |
| catch:misstated_coverage | 100.0% | [100.0%, 100.0%] |
| catch_expected:misstated_coverage | 100.0% | [100.0%, 100.0%] |
| catch:omitted_material_fact | 50.0% | [14.3%, 85.7%] |
| catch_expected:omitted_material_fact | 25.0% | [0.0%, 57.1%] |
| catch:wrong_amount | 100.0% | [100.0%, 100.0%] |
| catch_expected:wrong_amount | 60.0% | [14.3%, 100.0%] |

## Which rubric items fired

```
{
 "clean": {
  "numbers_consistent": 1,
  "facts_supported": 1
 },
 "contradicted_conclusion": {
  "outcome_consistent": 9,
  "facts_supported": 1,
  "numbers_consistent": 2
 },
 "invented_fact": {
  "facts_supported": 10,
  "numbers_consistent": 1
 },
 "misstated_coverage": {
  "faithful_to_clauses": 7,
  "facts_supported": 1
 },
 "omitted_material_fact": {
  "faithful_to_clauses": 1,
  "material_facts_addressed": 2,
  "facts_supported": 1,
  "outcome_consistent": 1
 },
 "wrong_amount": {
  "numbers_consistent": 3,
  "faithful_to_clauses": 2,
  "facts_supported": 2
 }
}
```

4 cases were unavailable (quota); re-run to fill them in.
