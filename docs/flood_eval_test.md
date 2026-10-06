# Flood line on real FEMA NFIP claims (locked test period)

3000 claims sampled (seed 42) from 104,740 with losses on or after 2024-07-01, run through the same harness graph as the auto line with the flood line plugged in; no LLM calls (23 s).
Outcomes are compared with what FEMA actually paid. Escalated claims count as not auto-decided.

| metric | value | 95% CI |
|---|---|---|
| auto_rate | 41.4% | [39.6%, 43.1%] |
| auto_agreement | 98.6% | [97.8%, 99.2%] |
| wrongly_denied_rate | 0.3% | [0.1%, 0.6%] |
| wrongly_paid_rate | 0.3% | [0.1%, 0.5%] |
| acv_within_5pct | 87.3% | [84.6%, 89.5%] |
| rc_first_payment_ratio | 0.937 | [0.928, 0.949] |

Why claims went to an adjuster (a claim can have several reasons):

- over_authority_limit: 1392
- adjudicator_escalated: 366

_This product uses the Federal Emergency Management Agency's OpenFEMA API, but is not endorsed by FEMA. The Federal Government or FEMA cannot vouch for the data or analyses derived from these data after the data have been retrieved from the Agency's website(s)._ Source: https://www.fema.gov/api/open/v2/FimaNfipClaims. Historical, closed claims only; no determination about any person is made.
