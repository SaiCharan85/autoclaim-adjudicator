# ML vs LLM vs hybrid: sim_us, validation, 200 claims

LLM groq:openai/gpt-oss-120b; ML catboost-6f935607c6b3; fraud share in sample 0.25, population rate 0.061 (budget metrics reweighted to it).

| arm | roc_auc | pr_auc | recall_at_budget | n_scored |
|---|---|---|---|---|
| ml | 0.869 | 0.474 | 0.485 | 130 |
| llm | 0.68 | 0.154 | 0.182 | 130 |
| hybrid | 0.869 | 0.505 | 0.485 | 130 |

| comparison | delta_roc_auc | delta_recall@budget |
|---|---|---|
| ml - llm | +0.189 [+0.072, +0.299] | +0.303 [+0.057, +0.536] |
| ml - hybrid | +0.000 [-0.004, +0.006] | +0.000 [-0.063, +0.000] |
| llm - hybrid | -0.189 [-0.298, -0.074] | -0.303 [-0.548, -0.067] |
