# Compute budget (free GPU quota, shared with 2 other projects)

Kaggle gives 30 GPU hours per week across all three projects. Every GPU job is estimated first and
its actual usage is logged here (from `run_log.json`).

| date | job | platform / GPU | estimate | actual | notes |
|---|---|---|---|---|---|
| 2026-10-05 | Step 7.5: QLoRA judge (2 epochs, 2,136 ex) + intake (3 epochs, 236 ex), base + fine-tuned eval, 2x GGUF | Kaggle T4 (script kernel, background) | ~2 h | **~4.0 h** (12:20-16:21; judge training 158.7 min, intake 26.9 min, rest scoring + export) | ended in ERROR at the intake GGUF export (disk); judge GGUF + both adapters + all scores saved |
| 2026-10-05 | Step 7.5: intake GGUF export only (`--export-only`, fresh disk) | Kaggle T4 (script kernel) | 15-20 min | **~0.3 h** (export 7.3 min + install/download) | complete; intake GGUF 2.5 GB |
| 2026-10-08 | Adjudicator QLoRA (2 epochs, 650 ex, ~1.6M tokens/epoch) + base vs fine-tuned scoring + GGUF | Kaggle T4 (script kernel, background) | ~2.5 h | **~2.8 h** (15:30-18:21; training 113.5 min at 440 tokens/s) | complete; adapter + GGUF 2.5 GB |

How the estimate was made: ~2.1M judge + 0.28M intake training tokens per epoch, at an assumed
~1,000-1,500 tokens/s for 4B QLoRA on a T4 with Unsloth, plus ~25 min of batched evaluation and
~20 min of GGUF conversion. `run_log.json` records the measured tokens/s so later estimates use it.

Lesson (2026-10-05): measured throughput was 425 (judge) and 517 (intake) tokens/s, not the assumed
~1,000-1,500, and per-step validation on all 535 judge examples added ~40 min. Fixed for later runs:
validation loss on a 150-example sample, checkpoints deleted before export, `--export-only` mode.
Use 450 tokens/s for T4 estimates of 4B QLoRA.
