# Fine-tuning (Step 7.5): local intake extractor and small judge

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
- **Adjudicator distillation:** deferred (user decision 2026-10-05): it needs teacher outputs on
  real harness inputs, about 900 free-tier requests.
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
