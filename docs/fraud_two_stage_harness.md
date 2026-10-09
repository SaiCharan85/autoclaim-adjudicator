# Two-stage fraud triage in the harness: validation

Validation-period claims 8,284 (638 true frauds); models trained on the training period only (`models/fraud_val`, `models/fraud_appraisal_val`). No LLM calls; the locked test is not touched.

Policy: refer when the first-notice score is in the top 5% (score >= 0.2264, fitted here) or the post-appraisal score >= 0.174461 (frozen in config).

| Routing | Claims reviewed | True fraud caught | Fraud that could be auto-paid |
|---|---|---|---|
| First notice only, line 0.12 | 10.0% | 61.3% | 38.7% |
| **Two-stage (with the appraisal)** | 9.5% | **79.5%** | 20.5% |

Wiring check: the auto line's fraud node referred the same claims as the toolkit on 298 of 300 sampled claims (99.3%). Facts come from the simulator truth here; with live intake, extraction errors can change a score.
