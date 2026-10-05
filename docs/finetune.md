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
kaggle kernels status <kaggle-username>/autoclaim-finetune
kaggle kernels output <kaggle-username>/autoclaim-finetune -p models/finetune
```

The kernel scores the base model, trains, scores the fine-tuned model on the same held-out
examples, exports `q4_k_m` GGUF + an Ollama `Modelfile`, and writes `run_log.json`. On Colab or a
local GPU: `python kaggle/finetune_qlora.py --data data/finetune --out finetune_out` (re-running
resumes from the last checkpoint and skips finished stages).

## Results

_Pending the GPU run._
