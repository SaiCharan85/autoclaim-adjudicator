# Harness evaluation (dev set, 50 claims)

Arms on the same claims; 95% bootstrap CIs over claims; differences are paired. Headline metrics were fixed before any run (`lines/auto/harness_eval.py`). Escalations answered by the simulated adjuster (error rate 0.05).

| metric | full | no_judge | no_critic | few_shot |
|---|---|---|---|---|
| auto_rate | 50.0% [36.0%, 64.0%] | 56.0% [42.0%, 70.0%] | 56.0% [42.0%, 70.0%] | 54.0% [40.0%, 68.0%] |
| auto_accuracy | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| wrong_auto_approve_rate | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| trap_leak_rate | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| fraud_auto_paid_rate | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| proposal_accuracy | 96.0% [90.0%, 100.0%] | 98.0% [94.0%, 100.0%] | 98.0% [94.0%, 100.0%] | 92.0% [84.0%, 98.0%] |
| calls_per_claim | 5.54 [5.02, 6.04] | 3.08 [3.02, 3.16] | 4.50 [4.22, 4.82] | 4.42 [4.14, 4.76] |
| tokens_per_claim | 9,930.14 [8,903.12, 10,930.59] | 4,181.84 [3,894.31, 4,472.23] | 3,954.76 [2,748.10, 5,278.64] | 5,328.36 [4,364.05, 6,543.65] |
| overpaid_per_claim | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| failsafe_rate | 0.0% [0.0%, 0.0%] | 2.0% [0.0%, 6.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

Paired difference no_judge - full (same claims):

| metric | difference [95% CI] | real gap? |
|---|---|---|
| auto_rate | 6.0% [-2.0%, 16.0%] | no |
| auto_accuracy | 0.0% [0.0%, 0.0%] | no |
| wrong_auto_approve_rate | 0.0% [0.0%, 0.0%] | no |
| trap_leak_rate | 0.0% [0.0%, 0.0%] | no |
| fraud_auto_paid_rate | 0.0% [0.0%, 0.0%] | no |
| proposal_accuracy | 2.0% [-4.0%, 10.0%] | no |
| calls_per_claim | -2.46 [-2.94, -1.98] | yes |

Paired difference no_critic - full (same claims):

| metric | difference [95% CI] | real gap? |
|---|---|---|
| auto_rate | 6.0% [0.0%, 14.0%] | no |
| auto_accuracy | 0.0% [0.0%, 0.0%] | no |
| wrong_auto_approve_rate | 0.0% [0.0%, 0.0%] | no |
| trap_leak_rate | 0.0% [0.0%, 0.0%] | no |
| fraud_auto_paid_rate | 0.0% [0.0%, 0.0%] | no |
| proposal_accuracy | 2.0% [-4.0%, 10.0%] | no |
| calls_per_claim | -1.04 [-1.56, -0.48] | yes |

Paired difference few_shot - full (same claims):

| metric | difference [95% CI] | real gap? |
|---|---|---|
| auto_rate | 4.0% [-2.0%, 12.0%] | no |
| auto_accuracy | 0.0% [0.0%, 0.0%] | no |
| wrong_auto_approve_rate | 0.0% [0.0%, 0.0%] | no |
| trap_leak_rate | 0.0% [0.0%, 0.0%] | no |
| fraud_auto_paid_rate | 0.0% [0.0%, 0.0%] | no |
| proposal_accuracy | -4.0% [-14.0%, 6.0%] | no |
| calls_per_claim | -1.12 [-1.60, -0.68] | yes |

By adjudicator model (quota fallbacks; point estimates, small n):

| arm | model | n | auto_rate | auto_accuracy | proposal_accuracy |
|---|---|---|---|---|---|
| full | groq:openai/gpt-oss-120b | 6 | 33.3% | 100.0% | 100.0% |
| full | groq:qwen/qwen3.8-27b | 44 | 52.3% | 100.0% | 95.5% |
| no_judge | groq:openai/gpt-oss-120b | 40 | 62.5% | 100.0% | 100.0% |
| no_judge | groq:qwen/qwen3.8-27b | 9 | 33.3% | 100.0% | 100.0% |
| no_judge | none | 1 | 0.0% | n/a | 0.0% |
| no_critic | groq:openai/gpt-oss-120b | 33 | 63.6% | 100.0% | 100.0% |
| no_critic | groq:qwen/qwen3.8-27b | 17 | 41.2% | 100.0% | 94.1% |
| few_shot | google:gemma-4-26b-a4b-it | 25 | 72.0% | 100.0% | 92.0% |
| few_shot | groq:qwen/qwen3.8-27b | 25 | 36.0% | 100.0% | 92.0% |
