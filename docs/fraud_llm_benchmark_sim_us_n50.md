# ML vs LLM vs hybrid: sim_us, validation, 50 claims

LLM groq:openai/gpt-oss-120b; ML catboost-6f935607c6b3; fraud share in sample 0.24, population rate 0.061 (budget metrics reweighted to it).

| arm | roc_auc | pr_auc | recall_at_budget | n_scored |
|---|---|---|---|---|
| ml | 0.807 | 0.216 | 0.333 | 50 |
| llm | 0.698 | 0.261 | 0.167 | 50 |
| hybrid | 0.822 | 0.234 | 0.333 | 50 |

| comparison | delta_roc_auc | delta_recall@budget |
|---|---|---|
| ml - llm | +0.109 [-0.101, +0.343] | +0.167 [-0.333, +0.462] |
| ml - hybrid | -0.015 [-0.068, +0.007] | +0.000 [-0.182, +0.000] |
| llm - hybrid | -0.124 [-0.344, +0.066] | -0.167 [-0.455, +0.333] |
