# ML vs LLM vs hybrid: sim_us, TEST (locked), 200 claims

LLM groq:openai/gpt-oss-120b; ML catboost-5e5590cf3e5e; fraud share in sample 0.25, population rate 0.061 (budget metrics reweighted to it).

| arm | roc_auc | pr_auc | recall_at_budget | n_scored |
|---|---|---|---|---|
| ml | 0.855 | 0.532 | 0.42 | 195 |
| llm | 0.631 | 0.146 | 0.1 | 195 |
| hybrid | 0.853 | 0.525 | 0.4 | 195 |

| comparison | delta_roc_auc | delta_recall@budget |
|---|---|---|
| ml - llm | +0.225 [+0.141, +0.309] | +0.320 [+0.148, +0.447] |
| ml - hybrid | +0.002 [-0.003, +0.009] | +0.020 [-0.000, +0.045] |
| llm - hybrid | -0.222 [-0.309, -0.137] | -0.300 [-0.444, -0.145] |
