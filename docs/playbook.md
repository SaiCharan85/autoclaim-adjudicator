# Operator playbook

How to run this day to day on free quotas. Commands assume `uv run` from the repo root.

## 1. One-time setup

1. `uv sync && uv run pre-commit install`
2. `cp .env.example .env` and fill in the free keys you have: `GROQ_API_KEY`, `GOOGLE_API_KEY`
   (AI Studio), optionally `LANGSMITH_API_KEY` (tracing) and `KAGGLE_API_TOKEN` (fine-tuning).
3. Data: `python scripts/download_data.py` (NHTSA CRSS; add `fema_nfip` for the flood line), then
   `python scripts/simulate_claims.py`. Narratives already built: `data/sim/packages/`.
4. Optional local fallback: install [Ollama](https://ollama.com/download), then create the two
   fine-tuned models (commands in [finetune.md](finetune.md)).
5. Check: `uv run pytest` (no network, no keys).

## 2. Processing claims

```bash
python scripts/run_claims.py --set dev --n 3 --dry-run   # what it would spend
python scripts/run_claims.py --set dev --n 3             # run (cached: re-runs are free)
streamlit run src/autoclaim/ui/console.py                # answer the escalated ones
python scripts/review_queue.py --list                    # or list them in the terminal
```

Each claim either finishes (`auto_decide`) or pauses at human review. A paused claim waits in the
checkpointer until someone answers it in the console. Finalization is idempotent, so answering
twice or re-running a claim never pays twice.

## 3. Living with free quotas

Free tiers allow roughly **40-50 harness claims per day** (about 5.5 LLM calls per claim). The binding
limits are requests per day and, on Groq, tokens per minute and output tokens per minute.

| You see | It means | Do |
|---|---|---|
| A claim escalated with `failsafe:` mentioning `daily safety limit` or `rate limited` | Every model in a role's chain is out of quota (our ledger stops below each limit) | Nothing: it is safe with a human. Normal runs fall back to the local Ollama model first if it is installed |
| An eval script exits with code 2: "quota exhausted ... re-run tomorrow" | Evaluations use API models only and stop instead of scoring an outage | Re-run the same command after the daily reset; finished claims and cached calls are free |
| `429 ... OTPM limit` from Groq | The request's `max_tokens` exceeds the model's output-tokens-per-minute cap | Set `otpm` for that model in the catalog; the client clamps `max_tokens` below it |
| A model fails every call all day (e.g. 429 "exceeded quota") | The free quota for that model on your key is exhausted or zero | Move it later in the chains in `config/carrier_config.yaml` |
| A thinking model's reply ends with `finish_reason: length` and an empty answer | Its thinking used up `max_tokens` | Set `thinking_tokens` for that model in the catalog (Gemma 4 has 2048) |

Providers reset their daily quotas at different times. The usage ledger counts each day from
Pacific midnight (`models.day_reset_utc_offset_hours: -8`), the conservative choice, so it never
assumes a quota has reset before the provider has.

## 4. Running evaluations

The rules (fixed metrics, paired comparisons, locked test touched only with `--final`) are in
[evaluation_protocol.md](evaluation_protocol.md).

```bash
python scripts/run_eval.py --build-memory 40 --dry-run              # fill feedback memory (train period)
python scripts/run_eval.py --set dev --n 50 --arms full no_judge --dry-run
python scripts/run_eval.py --set dev --n 50 --arms full no_judge no_critic few_shot
python scripts/judge_eval.py --n 10                                 # judge planted-error test
python scripts/flood_eval.py --n 3000                               # flood line, no LLM calls
```

- Always `--dry-run` first: it prints the requests and tokens against today's headroom.
- Pilot (about 50 claims) before any full run.
- Long runs go in the background (`.cache/runs/pipeline_*.sh`); every step appends to
  `.cache/runs/CONTEXT.md`, so a stopped session can be picked up.
- `--final` (locked test) only once the method is frozen; each use is appended to
  `docs/test_set_log.md` with the code version. Never re-run a finished `--final` step.

## 5. Retraining

| What | Command | Notes |
|---|---|---|
| Fraud models | `python scripts/train_fraud.py` then `--final` | Canaries run on every training; decisions on validation only |
| Two-stage triage threshold | `python scripts/fraud_triage.py`, freeze `fraud_model.triage.stage2_threshold` | Then `--final` once |
| Local judge / intake | `python scripts/build_finetune_data.py`, `python scripts/package_kaggle.py --user <name>`, then the two `kaggle` commands in [finetune.md](finetune.md) | State the GPU hours first (estimate at ~450 tokens/s on a T4) and log them in `compute_budget.md` |

Kaggle tips learned the hard way: a kernel cannot reuse a dataset's slug (409 Conflict); the 20 GB
working disk does not fit two GGUF exports plus checkpoints, so export runs free the checkpoints
first; on Windows, set `PYTHONIOENCODING=utf-8` before `kaggle kernels logs`.

## 6. Adding a line of business

1. Create `src/autoclaim/lines/<line>/` with a class implementing `LineOfBusiness`
   (`core/lob.py`): guardrails, intake, coverage, fraud, adjudicate, critic, judge,
   hard_escalations, fraud_score, case_summary, plus `memory_text` if it uses feedback memory.
2. Write the policy in your own words as a clause YAML (`retrieval/corpus.py` schema).
3. Give it a router and budget section in `carrier_config.yaml` and a builder like
   `lines/flood/build.py` (`Harness(line, config, audit, ledger)`).
4. Keep `core/` untouched; the flood line is the worked example ([flood.md](flood.md)).
