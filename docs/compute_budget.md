# Compute budget (free GPU quota, shared with 2 other projects)

Kaggle gives 30 GPU hours per week across all three projects. Every GPU job is estimated first and
its actual usage is logged here (from `run_log.json`).

| date | job | platform / GPU | estimate | actual | notes |
|---|---|---|---|---|---|
| 2026-10-05 | Step 7.5: QLoRA judge (2 epochs, 2,136 ex) + intake (3 epochs, 236 ex), base + fine-tuned eval, 2x GGUF | Kaggle T4 (script kernel, background) | ~2 h | _pending_ | Qwen3-4B-Instruct-2507, r=16, early stopping on validation loss |

How the estimate was made: ~2.1M judge + 0.28M intake training tokens per epoch, at an assumed
~1,000-1,500 tokens/s for 4B QLoRA on a T4 with Unsloth, plus ~25 min of batched evaluation and
~20 min of GGUF conversion. `run_log.json` records the measured tokens/s so later estimates use it.
