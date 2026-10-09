# Fine-tuning: local intake extractor, small judge and adjudicator

**Why:** free-tier API quotas (~40-50 claims/day) are the binding limit. Local fine-tuned models
on Ollama become the fallback at the end of each role's model chain, so a quota-exhausted day still
processes claims at $0, and the Step 8 ablation measures base vs fine-tuned.

| | intake extractor | small judge |
|---|---|---|
| task | claim narrative -> `ClaimFacts` JSON | judge case -> yes/no per rubric item |
| training data | 236 dev narratives, targets = simulator truth (a fact the story leaves out is null + `missing_info`) | 400 clean reference cases + 1,736 planted-error copies |
| validation | 64 (validation period) | 535 (validation period, 100 clean) |
| prompts | exactly the production intake prompt | exactly the production judge prompt |

- **Base model:** `Qwen3-4B-Instruct-2507` (Apache-2.0, text-only), 4-bit QLoRA via Unsloth;
  LoRA r=16; early stopping on validation loss; train-vs-validation loss gap reported.
- **Leakage:** training rows are from the train period, validation rows from the validation
  period, and a crash record used in training never appears in validation. The locked test period is
  refused by the builder (`autoclaim.finetune.datasets.split`). Data is built with no LLM calls.
- **Adjudicator:** added 2026-10-08 with gold targets instead of teacher outputs (no API calls);
  see the adjudicator section below.
- **Environment:** pinned in `kaggle/requirements-finetune.txt` (unsloth 2026.9.14 needs
  trl <= 0.24.0); nothing heavy is added to this project's dependencies.

## Run it

```bash
uv run python scripts/build_finetune_data.py            # free, ~1 min, CPU
uv run python scripts/package_kaggle.py --user <kaggle-username>
kaggle datasets create -p .cache/kaggle/dataset         # private
kaggle kernels push -p .cache/kaggle/kernel             # GPU batch job, ~2 h (docs/compute_budget.md)
kaggle kernels status <kaggle-username>/autoclaim-finetune-run
kaggle kernels output <kaggle-username>/autoclaim-finetune-run -p models/finetune
```

The kernel scores the base model, trains, scores the fine-tuned model on the same held-out
examples, exports `q4_k_m` GGUF + an Ollama `Modelfile`, and writes `run_log.json`. On Colab or a
local GPU: `python kaggle/finetune_qlora.py --data data/finetune --out finetune_out` (re-running
resumes from the last checkpoint and skips finished stages).

## Results (Kaggle T4, 2026-10-05; held-out validation-period examples)

| judge (120 cases) | base Qwen3-4B | fine-tuned |
|---|---|---|
| planted errors caught | 30% | **95%** |
| false alarms on clean cases | 18% | **6%** |
| rubric items right | 90.2% | 98.8% |
| valid JSON | 100% | 100% |

| intake (64 narratives) | base | fine-tuned |
|---|---|---|
| coverage-deciding fields right (12) | 58.3% | **95.1%** |
| valid JSON | 84% | 100% |
| weakest field | | driver name 62.5% |

Training: judge 450 steps (early stopping before 2 full epochs), final train loss 0.0004 vs best
validation loss 0.0014; intake 90 steps, 0.037 vs 0.035 (gap +0.002). Both near zero: the
targets are templated, so these numbers measure fit to the synthetic task, not real-world quality.

**Caveat:** the judge was trained and tested on the same kind of data (template explanations with
planted errors). Real adjudicator explanations are worded differently, so expect lower numbers in
production; Step 8 measures it on real harness explanations before the local judge is trusted.

Status: both GGUF models exported (q4_k_m, 2.5 GB each, with Ollama Modelfiles):
`models/finetune/out/judge/gguf_gguf/` and `models/finetune/export/out/intake/gguf_gguf/`. The
first run's intake export failed on Kaggle's 20 GB disk ("Not enough disk space to convert to
GGUF"); a 7-minute export-only run on a fresh disk produced it. GPU total: ~4.3 h.

## Local fallback on Ollama (wired 2026-10-05)

```bash
cd models/finetune/out/judge/gguf_gguf && ollama create autoclaim-judge -f Modelfile
cd models/finetune/export/out/intake/gguf_gguf && ollama create autoclaim-intake -f Modelfile
```

Both are the **last** entry of their role chains in `config/carrier_config.yaml`
(`ollama:autoclaim-judge`, `ollama:autoclaim-intake`, family `qwen` so the judge still never grades
a Qwen-written decision). They are used only when every free API model for the role is out of
quota, so a quota-exhausted day still processes claims at $0. On this CPU (no GPU): judge ~17-30 s
per call, intake ~34 s; if Ollama is not running, the call fails fast and the claim fails safe to
a human as before. Smoke test through the project's own client: a planted omitted fact was flagged
on exactly the right rubric item (`material_facts_addressed`), no validation retry, $0.

## Adjudicator (2026-10-08)

**Data (no LLM calls).** Each example is the exact production adjudicator prompt, built by the real
line code: policy declarations, true facts, deterministic facts, retrieved clauses (the production
retriever), fraud signals (the production fraud models) and a coverage analysis that agrees with the
truth. Only the LLM steps before the adjudicator are replaced, by gold answers
(`autoclaim.finetune.datasets.GoldClient`). The target is the true decision: outcome, reason codes,
cited clauses and a templated explanation (`judge_eval.explain`, the same templates the judge data
uses), with the deterministic payout.

- 650 train-period examples, 150 validation-period; approvals 35%, the rest spread evenly over the
  9 deny/escalate reasons (~47 each), so rare reasons are not swamped.
- A fraud escalation is a target only when the fraud score in the prompt is at or above the review
  threshold (0.24). True fraud with a low score looks like any other claim to the adjudicator, so
  those claims are left out rather than teaching it to escalate at random.
- `missing_information` escalations are left out: gold facts have nothing missing.

```bash
uv run python scripts/build_finetune_data.py --tasks adjudicator      # ~3 min, CPU
uv run python scripts/package_kaggle.py --user <name> --tasks adjudicator  # own kernel
```

| adjudicator (120 held-out validation claims) | base Qwen3-4B | fine-tuned |
|---|---|---|
| outcome right (approve / deny / escalate) | 85.0% | **99.2%** |
| reason codes exactly right | 80.0% | **97.5%** |
| reference clauses cited (recall) | 50.2% | **100%** |
| approvals of claims that should not be approved | 22.5% | **0%** |
| payout right on approvals | 100% | 100% |
| valid JSON | 100% | 100% |

Training: 164 steps (2 epochs, batch 1 x 8, max length 3,584), 113.5 min at 440 tokens/s on a T4;
final train loss 0.0034 vs best validation loss 0.0076. Whole session ~2.8 GPU h.

**Caveat:** training and test prompts both come from a perfect intake and coverage step. In
production the adjudicator reads LLM-written facts and coverage analyses, which are noisier, so
expect lower accuracy there; the base model's 22.5% wrong-approval rate shows why a fine-tune (or a
strong API model) matters for this role.

**Local fallback.** `ollama create autoclaim-adjudicator -f Modelfile` in
`models/finetune/adjudicator/out/adjudicator/gguf_gguf/` (q4_k_m, 2.5 GB, 6,144-token context); the
last entry of the adjudicator chain (`ollama:autoclaim-adjudicator`, family `qwen`, so the judge
still uses another family). Smoke test through the project client on a held-out denial: right
outcome, reason and clause, no validation retry, 56 s on this CPU (Ollama runs it 100% on CPU).
