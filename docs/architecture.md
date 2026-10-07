# Architecture

## The idea in one line

**Code decides control flow and facts; LLMs decide judgment and language; independent checks
verify.** There is no LLM supervisor choosing the next step: the graph is fixed, and every edge is
a code decision.

## The graph

```
guardrails -> intake -> [coverage || fraud] -> adjudicator -> critic -> judge -> router
                                                  ^   fail (<= 2 retries)  |      |-> auto_decide -> END
                                                  +------------------------+      |
                                         retries exhausted -> human_review <------+ -> feedback memory -> END
```

| Node | Kind | Job |
|---|---|---|
| guardrails | code | Validate the claim schema, lengths and policy fields; redact PII; flag prompt injection; reject bad input early |
| intake | LLM (cheap) | Narrative -> typed `ClaimFacts`, including what the story leaves out (`missing_info`) |
| (derive) | code | From the facts: coverage part, deductible, total-loss test, payout, late notice, driver status. Every number lives here |
| coverage | LLM + retrieval | BM25 + FAISS HNSW (local bge-large embeddings), fused with RRF, then 1-hop expansion on the policy graph; returns a reasoning chain that cites clauses |
| fraud | tools + LLM | CatBoost score with SHAP reasons, Isolation Forest, red-flag rules; an LLM only writes the assessment, and only when it can change the outcome |
| adjudicator | LLM (strong) | Outcome, reasons, cited clauses, explanation, confidence and a self-check; on retry it gets the critic's or judge's issues; code overwrites the payout |
| critic | code | Cited clauses exist; reason codes are valid; a denial cites a clause; no contradiction with the derived facts (e.g. approving comprehensive on a collision-only policy) |
| judge | LLM, another family | Yes/no rubric on the *reasoning* (faithful to clauses, facts supported, numbers consistent, material facts addressed, outcome consistent); never the decision |
| router | code | Hard rules first (late notice, unclear facts, input flags), then thresholds: low confidence, approve with a high fraud score, payout over the authority limit, checks failed |
| auto_decide | code | Finalize, idempotently: a claim is never decided or paid twice |
| human_review | interrupt | LangGraph `interrupt()` with a SQLite checkpointer; resumed from the Streamlit console or by the simulated adjuster in evaluations |
| feedback memory | store | One episode per human review (case, proposal, adjuster's decision and reason, correction flag), FAISS HNSW, used to retrieve context, never to reuse answers |

## Core vs line of business

`src/autoclaim/core/` knows nothing about cars or floods. It owns the graph, routing, retries,
audit, budgets, fail-safe handling, idempotent finalization, human review and memory. A line of
business implements one small protocol (`core/lob.py`):

```python
class LineOfBusiness(Protocol):
    name: str

    def guardrails(self, state) -> NodeResult: ...
    def intake(self, state) -> NodeResult: ...
    def coverage(self, state) -> NodeResult: ...
    def fraud(self, state) -> NodeResult: ...
    def adjudicate(self, state) -> NodeResult: ...
    def critic(self, state) -> NodeResult: ...
    def judge(self, state) -> NodeResult: ...
    def hard_escalations(self, state) -> list[str]: ...
    def fraud_score(self, state) -> float | None: ...
    def case_summary(self, state) -> dict: ...
```

- **Auto** (`lines/auto/`): narratives, LLM intake, retrieval-backed coverage, the fraud models.
- **Flood** (`lines/flood/`, [flood.md](flood.md)): structured FEMA records, code-only intake and
  coverage, a template or LLM explanation, no fraud model. It runs on the same core unchanged, and a
  test checks that `core/` contains no flood code.

Seams that are swappable by design: the judge (`core/judge.py`, our own interface with JudgeKit
behind it), the memory backend (exact search or FAISS HNSW), the LLM provider per role (one
OpenAI-compatible HTTP adapter for Groq, Google AI Studio and Ollama).

## Cross-cutting guarantees

| Guarantee | How |
|---|---|
| Numbers come from code | Derived facts compute every amount; the adjudicator's stated payout is overwritten and logged |
| Fail safe | Each role has a fallback chain (API models, then the local fine-tuned model on Ollama); if all fail, or any node raises, the claim fails safe to a human. Garbage is never auto-decided |
| Structured output | Pydantic v2 models for every LLM input and output; one retry with the validation error |
| Audit | Append-only JSONL per claim and node: time, outputs, model, tokens, cost (always $0), latency |
| Budgets | Per-claim caps on LLM calls and tokens trip the claim to a human; a persistent per-model daily ledger stops below each free-tier limit |
| Cache | Exact-match on-disk cache (model + messages + parameters): re-running costs nothing. Decision-path calls are never semantically cached |
| Idempotency | A finalization ledger keyed by claim id; replays and double resumes are refused or return the first decision |
| Judge independence | The judge never uses the adjudicator's model family (enforced per call); panels keep members on different families |
| Leakage | The harness never imports simulation truth or eval-only code (an AST import test); memory never stores test-period cases; evaluations open memory read-only and use API models only |

## Where the models run

| Role | Primary (free tier) | Fallbacks |
|---|---|---|
| intake | Gemini 3.5 Flash-Lite | gpt-oss-20b (Groq), local fine-tuned Qwen3-4B (Ollama) |
| coverage | Gemma 4 26B (AI Studio) | Gemini 3.5 Flash-Lite, gpt-oss-20b, Gemini 3.8 Flash |
| fraud explanation | gpt-oss-20b | Gemini 3.5 Flash-Lite |
| adjudicator | gpt-oss-120b (Groq) | qwen3.8-27b, Gemma 4 26B, Gemini 3.8 Flash |
| judge | Gemma 4 26B (AI Studio) | Gemini 3.5 Flash-Lite, qwen3.8-27b, gpt-oss-20b, Gemini 3.8 Flash, local fine-tuned Qwen3-4B judge |

All of this, plus thresholds, deductible rules and budgets, lives in `config/carrier_config.yaml`.
Gemma 4 replaced Gemini 3.8 Flash as the coverage and judge primary on 2026-10-06: Flash's small
free daily quota ran out (429 all day), while Gemma's is far larger. Gemma always thinks before
answering, so its catalog entry adds `thinking_tokens` on top of each role's `max_tokens`; the
thought block is stripped before the JSON is parsed.
Embeddings are local (bge-large via fastembed); nothing is sent to an embedding API.
