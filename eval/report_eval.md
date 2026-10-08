# Harness evaluation (eval set, 300 claims)

Arms on the same claims; 95% bootstrap CIs over claims; differences are paired. Headline metrics were fixed before any run (`lines/auto/harness_eval.py`). Escalations answered by the simulated adjuster (error rate 0.05).

| metric | full |
|---|---|
| auto_rate | 55.3% [49.7%, 61.0%] |
| auto_accuracy | 97.6% [95.0%, 99.4%] |
| wrong_auto_approve_rate | 1.3% [0.3%, 2.7%] |
| trap_leak_rate | 1.3% [0.0%, 3.5%] |
| fraud_auto_paid_rate | 15.0% [0.0%, 33.3%] |
| proposal_accuracy | 84.7% [80.7%, 88.7%] |
| calls_per_claim | 4.35 [4.24, 4.46] |
| tokens_per_claim | 8,425.19 [8,055.71, 8,794.99] |
| overpaid_per_claim | 52.71 [1.48, 123.37] |
| failsafe_rate | 0.0% [0.0%, 0.0%] |

By adjudicator model (quota fallbacks; point estimates, small n):

| arm | model | n | auto_rate | auto_accuracy | proposal_accuracy |
|---|---|---|---|---|---|
| full | google:gemini-3.8-flash | 3 | 33.3% | 100.0% | 33.3% |
| full | google:gemma-4-26b-a4b-it | 239 | 57.7% | 97.8% | 85.4% |
| full | groq:openai/gpt-oss-120b | 46 | 47.8% | 95.5% | 89.1% |
| full | groq:qwen/qwen3.8-27b | 12 | 41.7% | 100.0% | 66.7% |
