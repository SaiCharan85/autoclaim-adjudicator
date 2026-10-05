# ML vs LLM-only vs hybrid fraud scoring: protocol

Written 2026-10-04, **before** any validation result. Code: `src/autoclaim/lines/auto/fraud_benchmark.py`,
`scripts/fraud_llm_benchmark.py`.

## Question
Does an LLM add value to fraud triage at first notice of loss, on its own or combined with the ML tools?

| Arm | What scores the claim |
|---|---|
| **ml** | CatBoost fraud probability (the deterministic tool) |
| **llm** | LLM reading only the claim's first-notice facts (the same fields the ML model uses) |
| **hybrid** | LLM reading the facts **plus** the tool outputs: CatBoost score, top-5 SHAP reasons, Isolation Forest percentile, fired red flags |

## Data
- Grounded-hybrid claims: real NHTSA CRSS 2022–24 crash records with simulated policy and fraud
  labels. **The fraud labels are simulated**, and no public recent US claims data has real fraud labels.
- **Validation** (H1 2024) for development and decisions. The **locked test** (H2 2024) is run once
  with `--final` and logged in `test_set_log.md`.
- Stratified sample: 200 claims, 25% fraud (50 frauds). At the natural 6% rate, 200 claims would hold
  about 13 frauds, too few to compare arms. Claim order is stratified so that every prefix keeps the
  25% share. The 50-claim pilot is the first quarter of the same run and is reused from the cache.
- ML arm on validation: a CatBoost + Isolation Forest fit on the **training period only**. The
  production model was fit on train + validation, so scoring validation with it would be in-sample.
  With `--final`, the production model is used.

## LLM settings
- One model per run, pinned with no fallback mid-run: `groq:openai/gpt-oss-120b` (the fraud role's
  first choice), reasoning effort low, temperature 0.
- 10 claims per request. Batch ids are C01–C10, and real identifiers are never sent.
- Exact-match disk cache: a re-run costs nothing. A hard stop at 90% of the free daily quota; a run
  that stops resumes the next day from the cache.

## Metrics (fixed in advance)
- **Primary: ROC-AUC.** Ranking quality, unaffected by the over-sampling of fraud.
- Secondary:
  - recall@5%: the share of frauds caught when the fraud team reviews the riskiest 5%;
  - PR-AUC.

  Both are reweighted to the real validation fraud rate.
- Differences between arms: paired bootstrap, 1,000 resamples, 95% CI.

## Decision rule
- An LLM arm "adds value" only if its ROC-AUC beats **ml** with a paired 95% CI that excludes zero.
- The harness's fraud node uses the hybrid design only if hybrid beats ml by that rule. Otherwise the
  fraud number stays the ML score, and the LLM only writes the explanation around it. That is the
  design principle "tool numbers win".
- Prompts may change after the pilot. Once the 200-claim validation run starts, they are frozen.
