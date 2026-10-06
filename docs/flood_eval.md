# Flood line on real FEMA NFIP claims (development period)

3000 claims sampled (seed 42) from 92,283 with losses before 2024-07-01, run through the same harness graph as the auto line with the flood line plugged in; no LLM calls (22 s).
Outcomes are compared with what FEMA actually paid. Escalated claims count as not auto-decided.

| metric | value | 95% CI |
|---|---|---|
| auto_rate | 45.4% | [43.7%, 47.2%] |
| auto_agreement | 97.6% | [96.8%, 98.4%] |
| wrongly_denied_rate | 0.6% | [0.3%, 0.9%] |
| wrongly_paid_rate | 0.5% | [0.3%, 0.8%] |
| acv_within_5pct | 82.3% | [79.6%, 85.0%] |
| rc_first_payment_ratio | 0.932 | [0.919, 0.941] |

Why claims went to an adjuster (a claim can have several reasons):

- over_authority_limit: 973
- adjudicator_escalated: 664

_This product uses the Federal Emergency Management Agency's OpenFEMA API, but is not endorsed by FEMA. The Federal Government or FEMA cannot vouch for the data or analyses derived from these data after the data have been retrieved from the Agency's website(s)._ Source: https://www.fema.gov/api/open/v2/FimaNfipClaims. Historical, closed claims only; no determination about any person is made.
