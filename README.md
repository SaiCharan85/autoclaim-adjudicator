# autoclaim-adjudicator

> 🚧 Work in progress.

## Problem

Auto insurers receive a large volume of claims. Most are routine, but adjusters still review many of them by
hand, while some fraudulent or out-of-policy claims slip through.

**autoclaim-adjudicator** auto-resolves clear-cut auto physical-damage claims (collision and comprehensive),
catches fraud and coverage violations, and routes only genuinely ambiguous cases to a human adjuster with a
ready-made case summary. It improves over time from adjuster corrections.

The core idea: **code decides control flow and facts, LLMs decide judgment and language, and independent
checks verify.** It is a deterministic multi-agent workflow built on LangGraph, not an autonomous agent.

## Architecture

```mermaid
flowchart LR
    A[guardrails] -->|valid| B[intake]
    A -->|invalid| R[reject_input]
    B --> C[coverage]
    B --> D[fraud]
    C --> E[adjudicator]
    D --> E
    E --> F[critic]
    F -->|pass| G[judge]
    F -->|fail, retries left| E
    G -->|fail, retries left| E
    G -->|pass| H{router}
    F -->|retries exhausted| K[human_review]
    G -->|retries exhausted| K
    H -->|clear-cut| I[auto_decide]
    H -->|escalate| K
    K --> M[(feedback memory)]
    M -.->|similar past corrections| E
    I --> Z([END])
    M --> Z
    R --> Z
```

| Component | What it does |
|---|---|
| **guardrails** | Validates input and rejects bad claims early (code) |
| **intake** | Turns the claim narrative into typed facts (cheap LLM) |
| **coverage** | Hybrid BM25 + FAISS HNSW retrieval plus policy-graph expansion; outputs an explicit reasoning chain |
| **fraud** | CatBoost + SHAP, Isolation Forest, and a red-flag rules engine; an LLM writes the assessment |
| **adjudicator** | Approve / deny / escalate, with rationale, cited clauses, and a self-check checklist (strong LLM) |
| **critic** | Deterministic checks: clauses exist, numbers match, no contradictions (code) |
| **judge** | A different LLM grades the *reasoning* against a rubric (never the decision itself) |
| **router** | Config-driven thresholds decide between auto-decide and human review (code) |
| **human_review** | LangGraph interrupt/resume; answered by a Streamlit console or an oracle adjuster in evals |

## Status

Work in progress. Setup, results, and demo instructions will be added as the project lands.

## Data

See [DATA.md](DATA.md). No data is committed to this repository.

## License

MIT. See [LICENSE](LICENSE).
