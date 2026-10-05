# CLAUDE.md — autoclaim-adjudicator

## 1. Project summary
Auto insurers review many routine claims by hand while some fraud and out-of-policy claims slip through.
**autoclaim-adjudicator** auto-resolves clear-cut claims, catches fraud and coverage violations, and routes
only genuinely ambiguous cases to a human adjuster with a ready-made case summary. It learns from adjuster corrections.
Scope: auto physical damage (collision + comprehensive) under a personal auto policy. Health/property/flood are out
of scope, but the design is plug-in ready (shared core + pluggable line-of-business module). RLHF/DPO is out of scope.
**The crux is the agentic harness**: deterministic control flow, verified LLM judgment, HITL, audit, evals.

## 2. Working agreement (non-negotiable)
- **The user owns git.** Never run `git commit`, `git push`, `git rebase`, `git reset`, or anything that rewrites
  history. Read-only git (`status`, `diff`, `log`) is fine.
- **No AI attribution anywhere**: no `Co-Authored-By`, no "Generated with Claude Code", in code, docs, or messages.
- **Stop after every step** and print this block, then wait for "pushed" or feedback:
  ```
  ✅ STEP <n> COMPLETE — <title>
  What changed:   <short bullet list of files/modules>
  How to verify:  <exact commands to run>
  Decisions made: <trade-offs chosen, 1 line each>
  Needs from me:  <API keys, data downloads, choices — or "nothing">
  Suggested commit message:
    <conventional-commit message>
  ```
- **Ask for secrets/accounts** (API keys, credentials, paid services, Kaggle). Never hardcode; use `.env` + `.env.example`.
- **Ask before big decisions**: heavy dependencies, architecture changes, deviating from the step plan.
  Small choices are mine to make; list them under "Decisions made".
- **Don't invent APIs.** Check the installed library's source/docs (LangGraph changes often).
- Optimized, production-minded code: vectorized pandas/numpy, cached + batched + rate-limited LLM calls, profile slow paths.

## 3. Design principles
- **Code decides control flow and facts; LLMs decide judgment and language; independent checks verify.**
- Tool outputs are the source of truth. Deterministic numbers overwrite any LLM-stated numbers.
- Deterministic multi-agent workflow. **No LLM supervisor** picking the next agent.
- The judge evaluates **reasoning only, never the decision**. Labels evaluate outcomes; judges evaluate explanations.
- Fail safe: primary model → backup model → `human_review`. Never silently output garbage.

## 4. Architecture
```
guardrails → intake → [coverage ∥ fraud] → adjudicator → critic → judge → router
                                               ↑   fail (≤2 retries)  │      ├→ auto_decide → END
                                               └──────────────────────┘      │
                                          retries exhausted → human_review ←─┘ → feedback memory → END
```
| Node | Type | Job |
|---|---|---|
| guardrails | code | Validate schema/length/policy fields, flag PII, `reject_input` early |
| intake | LLM (cheap) | Narrative → `ClaimFacts` (incl. `missing_info`) |
| coverage | LLM + retrieval | BM25 + FAISS HNSW (RRF) → policy-graph 1–2 hop expansion → `CoverageResult` with reasoning chain |
| fraud | tools + LLM | CatBoost + SHAP, Isolation Forest, red-flag rules → LLM assessment; tool numbers win |
| adjudicator | LLM (strong) | `Decision` + self-check checklist; few-shot from feedback memory; gets critic/judge issues on retry |
| critic | code | Clauses/edges exist, no contradictions, denial cites ≥1 clause, numbers match policy/config |
| judge | LLM (≠ adjudicator) | Yes/no rubric on faithfulness, completeness, unsupported facts, consistency |
| router | code | Hard rules first; escalate on low confidence, approve + high fraud, amount > authority limit |
| auto_decide | code | Finalize; **idempotent** (never decides/pays a claim twice) |
| human_review | interrupt | `interrupt()` + checkpointer; resumed by Streamlit console or oracle adjuster |
| feedback memory | store | Episodic (narrative, decision, correction, reason); FAISS HNSW; swappable backend |

**Boundary rule:** nothing auto-specific in `src/autoclaim/core/`. Auto lives in `src/autoclaim/lines/auto/`.
Cross-cutting: Pydantic v2 structured output (retry once with the error), append-only audit log per node
(node, ts, outputs, model, tokens, cost, latency), tracing via env vars, per-claim budget caps (trip to human review),
on-disk LLM cache, all thresholds and model-per-role in `config/carrier_config.yaml`.

## 5. Coding standards
- Type hints everywhere; `mypy --strict` on `src/` (lenient on scripts/eval/tests).
- Pydantic models for **all** LLM inputs/outputs.
- Small, pure functions where possible; no premature abstraction, but keep the seams the architecture names.
- **Every piece of functionality gets pytest unit tests in the same step that adds it** — happy path, edge
  cases, and failure paths. No step is complete with untested code. Test files mirror modules:
  `src/autoclaim/x/y.py` → `tests/x/test_y.py`.
- Mock LLMs in tests (no network, no API keys); mark real-API tests `@pytest.mark.live_llm` (excluded in CI).
- Seeded randomness (`harness.seed` in config). Config over constants. No secrets in code.
- ruff (lint + format), line length 100.

## 5b. Leakage and overfitting rules (user requirement; enforced by tests)
- **First-notice-only features:** every simulator column is tagged `fnol` / `appraisal` / `post_fnol` /
  `ground_truth`; only `fnol` columns may be stage-1 features; the stage-2 (post-appraisal) fraud
  model may also use `appraisal` columns (user decision 2026-10-04). **The harness never imports the ground-truth oracle.**
- **Canaries every training run:** shuffled-label ROC-AUC ≈ 0.5, and flag any single feature with ROC-AUC > 0.9.
- **Time order:** train on the past, test on the future; every encoder, scaler, vocab and reference fit on train only.
- **3-way split:** train / validation (all decisions) / **locked test**, touched only with `--final` and
  logged in `docs/test_set_log.md`. Plus a fresh-seed simulator holdout and a perturbed-world shift test.
- **Overfitting:** early stopping on validation; report train-vs-validation gap; 3 seeds (mean ± std).
- **Harness evals:** eval claims from the test period only; few-shot, feedback memory and fine-tuning
  data from train/validation only. A/B = same claims, metric fixed in advance, paired bootstrap, no peeking.

## 6. Data and copyright rules
- Never commit data, model weights, FAISS indexes, or caches (`data/`, `models/`, `.cache/`, `*.faiss` are ignored).
- Never paste ISO policy forms, NICB publications, or Kaggle data into the repo.
- Policy text is written in our own words; fraud red flags are paraphrased.
- Update `DATA.md` whenever a data source changes, and ask the user before changing any source.

## 7. Cost rules
- **HARD RULE: free resources only.** Every model, API, service, and dataset must be usable at $0
  (free tiers, local/open-weight models, open-source tools). Never wire in a paid-only service; if
  something free is unavailable, stop and ask. Design for free-tier rate limits (throttle, retry, cache).
  Trial credits count as free only while they last: enforce a hard token cap below the quota so nothing
  ever bills, and fall back to the next provider when the cap is hit.
- Cache every LLM call on disk; re-running evals must not re-pay (or re-spend rate limit) for identical calls.
  The primary cache is **exact-match** (hash of model + messages + params). **Never semantic-cache decision-path
  calls** (intake, coverage, fraud, adjudicator, judge): near-identical claims differ precisely on the traps.
  Similarity is used to *retrieve context* (feedback memory), never to *reuse answers*.
- **Token + request frugality (user rule): never waste tokens or requests/day.**
  - Persistent per-model daily counter (requests, tokens) with a hard stop below the free-tier limit;
    limits live in config. Tripping it falls back to the next provider or stops; it never silently retries.
  - Every batch job has a `--dry-run` that prints estimated requests/tokens/time before spending anything.
  - Batch several items per request where quality allows (requests/day is usually the binding limit).
  - Compact prompts: static prefix first (provider prefix caching), compact JSON schemas, only the
    graph-expanded clauses, few-shot only when measured to help. Cap `max_tokens` per role; lowest
    reasoning effort that works.
  - Code decides first: deterministic checks (guardrails, hard router rules, critic) run before any
    LLM call, and an LLM is skipped when its answer can't change the outcome.
  - Development: mocks in tests; at most one tiny live smoke test per integration; never a live call
    just to look something up (read the console or docs instead).
- Cheapest adequate model per role.
- Pilot (~50) before any full run.
- Before any large run, show the estimated request/token count against free-tier limits and the wall-clock time.
- **Free quotas are shared with 2 other projects.** Before any GPU job, state the estimated GPU hours; log actual
  usage in `docs/compute_budget.md`. CPU work (data prep, scoring) never runs on a GPU session; batch GPU work so
  one session does several jobs (load the teacher model once); use background commit runs, never idle sessions.

## 8. Commands
| Task | Command |
|---|---|
| Setup | `uv sync` then `uv run pre-commit install` |
| Lint | `uv run ruff check . && uv run ruff format --check .` |
| Type check | `uv run mypy` |
| Test | `uv run pytest` (real-data tests auto-skip if `data/` is absent) |
| Download data | `uv run python scripts/download_data.py` (CRSS 2022–24 + Kaggle; Kaggle needs `KAGGLE_API_TOKEN`) |
| Build claims | `uv run python scripts/simulate_claims.py` → `data/sim/` (grounded hybrid, ~1 s) |
| Profile data | `uv run python scripts/profile_data.py` → `docs/data_profile.md` |
| Train fraud model | `uv run python scripts/train_fraud.py [--dataset sim_us\|legacy_1990s] [--skip-benchmark]` → validation only |
| Locked test | `uv run python scripts/train_fraud.py --final` (scores test + holdouts once; appends `docs/test_set_log.md`) |
| Two-stage fraud triage | `uv run python scripts/fraud_triage.py [--final]` (validation fits + freezes nothing; set `fraud_model.triage.stage2_threshold`) |
| Fraud lever experiments | `uv run python scripts/fraud_experiments.py --levers repair_cost weighting ensemble grid` (validation only) |
| Fraud ML vs LLM vs hybrid | `uv run python scripts/fraud_llm_benchmark.py --n 200 [--dry-run] [--final --model groq:openai/gpt-oss-120b]` |
| Retrieval benchmark | `uv run python scripts/retrieval_benchmark.py` → `docs/retrieval_benchmark.md` |
| Build claim narratives | `uv run python scripts/build_narratives.py --set eval\|dev --n 300 [--dry-run]` → `data/sim/packages/` |
| Run claims (smoke) | `uv run python scripts/run_claims.py --set dev --n 3 [--dry-run]` |
| Run demo | `uv run python scripts/run_claims.py --set dev --n 3` then `uv run streamlit run src/autoclaim/ui/console.py` |
| Judge planted-error eval | `uv run python scripts/judge_eval.py --n 10 [--dry-run] [--final]` → `docs/judge_eval_<set>_n<N>.md` (checkpointed, resumable) |
| Review queue | `uv run python scripts/review_queue.py --list` / `--adjuster oracle [--error-rate 0.05]` (no LLM calls) |
| Fine-tuning data / Kaggle | `uv run python scripts/build_finetune_data.py` then `scripts/package_kaggle.py --user <name>` (docs/finetune.md) |
| Run eval | `uv run python scripts/run_eval.py --set dev --n 50 --arms full no_judge no_critic few_shot [--dry-run]`; memory first: `--build-memory 40`; locked test: `--set eval --final` → `eval/report_<set>.md` |

## 9. Step plan
- ☑ Step 0 — Foundations (CLAUDE.md, pyproject, skeleton, CI, DATA.md, README stub)
- ☑ Step 1 — Data layer (Kaggle download, schemas, profile → `docs/data_notes.md`)
- ☑ Step 2 — Fraud ML (5-model benchmark, CatBoost + SHAP, Isolation Forest, rules, model card)
- ☑ Step 2b — Real recent data: NHTSA CRSS 2022–24 grounded-hybrid claims, dataset specs, leakage
  canaries, 3-way splits + locked test log, seeds, savings metric (`docs/simulator.md`)
- ☑ Step 3 — Policy & retrieval (75-clause policy YAML, NetworkX graph, BM25 + bge-large HNSW + RRF, 1-hop, `docs/retrieval_benchmark.md`)
- ☑ Step 4 — Synthetic claims (row-conditioned, style cards, traps enriched to 50%; 300 eval + 300 dev)
- ☑ Step 5 — Harness core (state, nodes, edges, checkpointer, router, audit, fallback, budgets, idempotency; `docs/harness.md`)
- ☑ Step 6 — Judge on JudgeKit (../JudgeKit, editable; CI checks out tag v0.1.0): adapter in `core/judge.py`, rubric severity tiers, insurance planted errors (`lines/auto/judge_eval.py`), `scripts/judge_eval.py` (live smoke n=1 done; pilot n=10 = 54 calls pending OK). Local judge: add an `ollama:` model to the judge chain once one exists (Step 7.5 exports the fine-tuned small judge); Ollama is not installed/running yet
- ☑ Step 7 — Memory & oracle: `core/memory.py` (episodes, HNSW, cutoff 2024-07-01, read-only for evals, few-shot off until A/B), `simulator/adjuster.py` (noisy oracle), `core/review.py` + `scripts/review_queue.py` (pause/resume e2e, free)
- ◐ Step 7.5 — Fine-tuning (docs/finetune.md): data built (no LLM), Qwen3-4B-Instruct-2507 QLoRA script + Kaggle kernel ready; GPU run pending user push (~2 h). Distillation deferred; deps pinned in kaggle/requirements-finetune.txt (user decisions 2026-10-05)
- ◐ Step 8 — Evaluation: code done (`lines/auto/harness_eval.py` pre-registered metrics, ablations no_judge/no_critic, few_shot learning loop, paired bootstrap; `scripts/run_eval.py` per-arm state + resume); live runs pending OK (`run_eval.py`, metrics, ablations incl. base vs. fine-tuned, learning loop, `eval/report.md`)
  + fraud A/B: ML-only vs LLM-only vs hybrid on ~200 test claims (cached; user decision 2026-10-04)
- ☐ Step 8.5 — Flood line module on real FEMA NFIP claims (plug-in demo; after the auto harness works)
- ☑ Step 9 — Adjuster console: `ui/console.py` (Streamlit 1.65) over tested `ui/review_view.py`; queue, case view, decision form -> `core.review.resume`, audit trail; AppTest smoke passes
- ☐ Step 10 — Polish (final README, architecture doc, playbook, CI badge, limitations)

Decisions 2026-10-04: two-stage fraud triage (first notice + independent appraisal) for >= 80% TRUE-fraud
interception: locked test 78.3% at 10.0% review rate, ROC-AUC 0.860 / 0.955 after the shop-estimate realism fix (model card); one OpenAI-compatible httpx adapter for all providers (no vendor SDKs);
Gemini 2.5 is closed to new keys, so Gemini 3.x (`gemini-3.8-flash`, `gemini-3.5-flash-lite`);
roles spread across free models (intake Flash-Lite, coverage/judge Gemini 3.8 Flash, adjudicator
gpt-oss-120b); a fraud-model lever experiment adopted nothing (model card); free-tier throughput is
~40–50 harness claims/day (`docs/harness.md`).
Decisions so far: LLMs = hybrid free (Groq + Google AI Studio free tiers, local Ollama fallback;
judge from a different model family than the adjudicator), tracing = LangSmith (free Developer tier),
embeddings = local only: bge-large-en-v1.5 via fastembed (user choice; the retrieval benchmark still
reports small/base for reference; no API embeddings for now), large batch generation + fine-tuning = Kaggle free GPU notebooks (vLLM / QLoRA), free Colab as overflow
(GPU jobs must be platform-agnostic and checkpoint often so they can resume after a disconnect).

**Backlog (user: "later")**: repair-cost residual tested 2026-10-04, not adopted (CI includes 0);
older real crashes CRSS 2016–2021 for a drift study. Headroom (validation): model at ~55% of the label-noise
ceiling (PR-AUC 0.418 vs 0.761); vs true fraud, precision@5% = 0.675.

## 10. Explanation style
The user is an experienced ML/NLP engineer with little insurance domain knowledge.
Explain domain concepts in plain English, briefly. Prefer a concrete step-by-step trace
(e.g. "claim says deer → comprehensive → $250 deductible") over abstractions.
