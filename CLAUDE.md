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
- Token hygiene: static content (system prompt, rubric, schema) first so provider prefix caching applies; send only
  graph-expanded clauses, never the whole policy; cap output tokens per role; code checks run before LLM checks.
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
| Test | `uv run pytest` |
| Run demo | _TBD (Step 9)_ |
| Run eval | _TBD (Step 8)_ |

## 9. Step plan
- ☑ Step 0 — Foundations (CLAUDE.md, pyproject, skeleton, CI, DATA.md, README stub)
- ☐ Step 1 — Data layer (Kaggle download, schemas, profile → `docs/data_notes.md`)
- ☐ Step 2 — Fraud ML (5-model benchmark, CatBoost + SHAP, Isolation Forest, rules, model card)
- ☐ Step 3 — Policy & retrieval (policy YAML, NetworkX graph, BM25 + HNSW + RRF, k-hop, benchmark doc)
- ☐ Step 4 — Synthetic claims (row-conditioned, red flags, 6 traps, messiness; pilot 50 → full 300–500)
- ☐ Step 5 — Harness core (state, nodes, edges, checkpointer, router, audit, fallback, budgets, idempotency)
- ☐ Step 6 — Judge (`Judge` protocol, local rubric adapter, YAML rubrics, planted-error tests)
- ☐ Step 7 — Memory & oracle (feedback memory, noisy oracle adjuster, interrupt/resume end to end)
- ☐ Step 7.5 — Fine-tuning on Kaggle (free GPU, QLoRA): intake extractor, small judge, adjudicator
  (distillation). Leakage-safe splits by source row, `finetune` dependency group only, GGUF → Ollama export
- ☐ Step 8 — Evaluation (`run_eval.py`, metrics, ablations incl. base vs. fine-tuned, learning loop, `eval/report.md`)
- ☐ Step 9 — Adjuster console (Streamlit, resume interrupted claims, demo script)
- ☐ Step 10 — Polish (final README, architecture doc, playbook, CI badge, limitations)

Decisions so far: LLMs = hybrid free (Groq + Google AI Studio free tiers, local Ollama fallback;
judge from a different model family than the adjudicator), tracing = LangSmith (free Developer tier),
large batch generation + fine-tuning = Kaggle free GPU notebooks (vLLM / QLoRA), free Colab as overflow
(GPU jobs must be platform-agnostic and checkpoint often so they can resume after a disconnect).

## 10. Explanation style
The user is an experienced ML/NLP engineer with little insurance domain knowledge.
Explain domain concepts in plain English, briefly. Prefer a concrete step-by-step trace
(e.g. "claim says deer → comprehensive → $250 deductible") over abstractions.
