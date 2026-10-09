"""QLoRA fine-tuning of the local intake extractor, small judge and adjudicator. Free GPU only.

One GPU session does everything (load once, no idle time):
  for each task (judge, intake, adjudicator; `--tasks` picks):
    1. score the BASE model on held-out validation examples (for the base-vs-fine-tuned ablation)
    2. QLoRA-train on the train split, early stopping on validation loss, checkpoint every 50 steps
    3. score the FINE-TUNED model on the same held-out examples
    4. export GGUF (q4_k_m) + an Ollama Modelfile
  write run_log.json: GPU, minutes, tokens/s, library versions, metrics.

Platform-agnostic: Kaggle (/kaggle/input, /kaggle/working) or Colab/local (--data/--out). Re-running
after a disconnect resumes from the last checkpoint and skips finished stages.

Kaggle notebook cell (GPU T4/P100, internet on):
  !pip install -q -r /kaggle/input/autoclaim-finetune/requirements-finetune.txt
  !python /kaggle/input/autoclaim-finetune/finetune_qlora.py

Base model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit (Apache-2.0, text-only, verified on
the Hugging Face API 2026-10-05). Data: data/finetune/*.jsonl from scripts/build_finetune_data.py
(simulator truth, train period only; validation period for validation).
"""

import argparse
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
CHAT_TEMPLATE = "qwen3-instruct"
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
TASKS = ("judge", "intake", "adjudicator")
DEFAULT_TASKS = ("judge", "intake")
EPOCHS = {"judge": 2, "intake": 3, "adjudicator": 2}
MAX_NEW = {"judge": 700, "intake": 900, "adjudicator": 700}
# adjudicator prompts carry ~2.5k tokens of clauses: a longer window, and batch 1 x 8 (the same
# effective batch) so a T4's 16 GB holds it
MAX_LEN = {"adjudicator": 3584}
BATCH = {"judge": (2, 4), "intake": (2, 4), "adjudicator": (1, 8)}
SEED = 42

# ---------------------------------------------------------------- pure helpers (tested on CPU)


def find_dirs(data: str | None, out: str | None) -> tuple[Path, Path]:
    """Data and output folders: explicit flags win, then Kaggle's layout, then ./data/finetune."""
    if data:
        data_dir = Path(data)
    else:
        kaggle = sorted(Path("/kaggle/input").rglob("judge_train.jsonl"))  # any nesting depth
        data_dir = kaggle[0].parent if kaggle else Path("data/finetune")
    out_dir = Path(out) if out else (
        Path("/kaggle/working/out") if Path("/kaggle/working").exists() else Path("finetune_out")
    )  # fmt: skip
    return data_dir, out_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def split_prompt(example: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """(messages without the answer, the reference answer)."""
    msgs = example["messages"]
    if msgs[-1]["role"] != "assistant":
        raise ValueError("the last message must be the reference answer")
    return msgs[:-1], msgs[-1]["content"]


def extract_json(text: str) -> dict[str, Any] | None:
    """The JSON object in a generation (fences and stray prose tolerated); None if invalid."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if not 0 <= start < end:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score_judge(
    predictions: Sequence[str], references: Sequence[str], error_types: Sequence[str | None]
) -> dict[str, Any]:
    """JSON validity, per-item accuracy, catch rate on planted copies, false alarms on clean."""
    valid = items = correct = 0
    caught = planted = alarms = clean = 0
    for pred, ref, etype in zip(predictions, references, error_types, strict=True):
        want = {a["id"]: a["answer"] for a in json.loads(ref)["answers"]}
        got_obj = extract_json(pred)
        got = {}
        if got_obj is not None and isinstance(got_obj.get("answers"), list):
            got = {a.get("id"): a.get("answer") for a in got_obj["answers"] if isinstance(a, dict)}
            valid += 1
        items += len(want)
        correct += sum(got.get(i) == v for i, v in want.items())
        flagged = any(got.get(i) == "no" for i in want) or not got
        if etype is None:
            clean += 1
            alarms += flagged
        else:
            planted += 1
            caught += any(got.get(i) == "no" for i, v in want.items() if v == "no")
    n = len(predictions)
    return {
        "n": n,
        "json_valid": valid / n if n else None,
        "item_accuracy": correct / items if items else None,
        "catch_rate": caught / planted if planted else None,
        "false_alarm_rate": alarms / clean if clean else None,
    }


INTAKE_FIELDS = ("cause", "use_at_loss", "driver_name", "vehicle_parked", "police_report",
                 "police_report_hours", "witnesses", "injuries", "towed", "other_driver_fled",
                 "insured_at_fault", "loss_date")  # fmt: skip


def score_intake(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """JSON validity and exact match per coverage-deciding field (hours within 3 h)."""
    valid = 0
    hits = dict.fromkeys(INTAKE_FIELDS, 0)
    for pred, ref in zip(predictions, references, strict=True):
        want = json.loads(ref)
        got = extract_json(pred)
        if got is None:
            continue
        valid += 1
        for f in INTAKE_FIELDS:
            w, g = want.get(f), got.get(f)
            if f == "police_report_hours" and w is not None and isinstance(g, int | float):
                hits[f] += abs(float(g) - float(w)) <= 3
            else:
                hits[f] += g == w
    n = len(predictions)
    per_field = {f: (hits[f] / n if n else None) for f in INTAKE_FIELDS}
    known = [v for v in per_field.values() if v is not None]
    return {"n": n, "json_valid": valid / n if n else None,
            "field_accuracy": sum(known) / len(known) if known else None,
            "per_field": per_field}  # fmt: skip


def score_adjudicator(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """JSON validity, outcome accuracy, exact reason codes, cited-clause recall, payout match on
    approvals (production overwrites it anyway) and wrong approvals (the costly error)."""
    valid = outcome = reasons = approvals = payout_ok = wrong_approve = not_approve = 0
    recall: list[float] = []
    for pred, ref in zip(predictions, references, strict=True):
        want = json.loads(ref)
        got = extract_json(pred) or {}
        valid += "outcome" in got
        outcome += got.get("outcome") == want["outcome"]
        reasons += sorted(got.get("reasons") or []) == sorted(want["reasons"])
        if want["cited_clauses"]:
            cited = set(got.get("cited_clauses") or [])
            recall.append(len(cited & set(want["cited_clauses"])) / len(want["cited_clauses"]))
        if want["outcome"] == "approve":
            approvals += 1
            p = got.get("payout")
            payout_ok += isinstance(p, int | float) and abs(float(p) - want["payout"]) <= 1.0
        else:
            not_approve += 1
            wrong_approve += got.get("outcome") == "approve"
    n = len(predictions)
    return {"n": n, "json_valid": valid / n if n else None,
            "outcome_accuracy": outcome / n if n else None,
            "reasons_exact": reasons / n if n else None,
            "cited_recall": sum(recall) / len(recall) if recall else None,
            "payout_match": payout_ok / approvals if approvals else None,
            "wrong_approve_rate": wrong_approve / not_approve if not_approve else None}  # fmt: skip


def ollama_modelfile(gguf_name: str, task: str) -> str:
    """Modelfile for `ollama create`; greedy decoding, the task's longest answers fit."""
    num_ctx = 6144 if task == "adjudicator" else 4096
    return (f"FROM ./{gguf_name}\n"
            "PARAMETER temperature 0\n"
            f"PARAMETER num_predict {MAX_NEW[task]}\n"
            f"PARAMETER num_ctx {num_ctx}\n")  # fmt: skip


def find_gguf(task_dir: Path) -> Path | None:
    """The exported GGUF wherever Unsloth wrote it (it appends "_gguf" to the target folder)."""
    found = sorted(task_dir.rglob("*.gguf"), key=lambda p: p.stat().st_size)
    return found[-1] if found else None


def write_modelfile(gguf: Path, task: str) -> Path:
    """An Ollama Modelfile next to the GGUF, referring to it by its real name."""
    path = gguf.parent / "Modelfile"
    path.write_text(ollama_modelfile(gguf.name, task), encoding="utf-8")
    return path


def done_marker(out_dir: Path, task: str, stage: str) -> Path:
    return out_dir / task / f".done_{stage}"


# ---------------------------------------------------------------- GPU work (Kaggle/Colab only)


def load(max_len: int) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=max_len, dtype=None, load_in_4bit=True,
        random_state=SEED,
    )  # fmt: skip
    return model, get_chat_template(tok, chat_template=CHAT_TEMPLATE)


def generate(model: Any, tok: Any, examples: Sequence[dict[str, Any]], max_new: int,
             batch: int = 8) -> list[str]:  # fmt: skip
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    outs: list[str] = []
    for i in range(0, len(examples), batch):
        prompts = [tok.apply_chat_template(split_prompt(e)[0], tokenize=False,
                                           add_generation_prompt=True)
                   for e in examples[i : i + batch]]  # fmt: skip
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        outs += tok.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return outs


def evaluate(task: str, model: Any, tok: Any, val: list[dict[str, Any]]) -> dict[str, Any]:
    preds = generate(model, tok, val, MAX_NEW[task], batch=4 if task == "adjudicator" else 8)
    refs = [split_prompt(e)[1] for e in val]
    if task == "judge":
        return score_judge(preds, refs, [e["meta"].get("error_type") for e in val])
    if task == "adjudicator":
        return score_adjudicator(preds, refs)
    return score_intake(preds, refs)


def train(task: str, model: Any, tok: Any, train_rows: list[dict[str, Any]],
          val_rows: list[dict[str, Any]], out: Path, max_len: int) -> dict[str, Any]:  # fmt: skip
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    from datasets import Dataset

    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0.0,
                                             random_state=SEED)  # fmt: skip

    def to_text(rows: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list([{"text": tok.apply_chat_template(r["messages"], tokenize=False)}
                                  for r in rows])  # fmt: skip

    ckpt = out / task / "checkpoints"
    args = SFTConfig(
        output_dir=str(ckpt), dataset_text_field="text", max_length=max_len, packing=False,
        per_device_train_batch_size=BATCH[task][0], gradient_accumulation_steps=BATCH[task][1],
        num_train_epochs=EPOCHS[task], learning_rate=2e-4, lr_scheduler_type="linear",
        warmup_ratio=0.03, weight_decay=0.01, logging_steps=10, eval_strategy="steps",
        eval_steps=50, save_strategy="steps", save_steps=50, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        seed=SEED, report_to="none", per_device_eval_batch_size=2,
    )  # fmt: skip
    trainer = SFTTrainer(model=model, processing_class=tok, args=args,
                         train_dataset=to_text(train_rows), eval_dataset=to_text(val_rows),
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])  # fmt: skip
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART,
                                      response_part=RESPONSE_PART)  # fmt: skip
    resume = any(ckpt.glob("checkpoint-*"))
    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=resume or None)
    minutes = (time.time() - t0) / 60
    logs = trainer.state.log_history
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
    tokens = sum(len(tok(x["text"])["input_ids"]) for x in to_text(train_rows)) * EPOCHS[task]
    model.save_pretrained(str(out / task / "lora"))
    tok.save_pretrained(str(out / task / "lora"))
    return {"minutes": round(minutes, 1), "resumed": resume,
            "steps": stats.global_step, "final_train_loss": train_loss[-1] if train_loss else None,
            "best_eval_loss": min(eval_loss) if eval_loss else None,
            "train_minus_eval_loss": (train_loss[-1] - min(eval_loss))
            if train_loss and eval_loss else None,
            "tokens_per_s": round(tokens / (minutes * 60), 1) if minutes else None,
            "model": model}  # fmt: skip


def export(task: str, model: Any, tok: Any, out: Path) -> str:
    import shutil

    # the merge needs ~8 GB of scratch: free the training checkpoints first (20 GB disk on Kaggle)
    shutil.rmtree(out / task / "checkpoints", ignore_errors=True)
    model.save_pretrained_gguf(str(out / task / "gguf"), tok, quantization_method="q4_k_m")
    gguf = find_gguf(out / task)
    if gguf is None:
        raise RuntimeError(f"{task}: GGUF export produced no .gguf file")
    write_modelfile(gguf, task)
    return str(gguf.relative_to(out))


def versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("unsloth", "unsloth_zoo", "trl", "transformers", "peft", "bitsandbytes", "torch"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "missing"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=TASKS)
    ap.add_argument("--eval-n", type=int, default=120, help="held-out examples scored per task")
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--skip-base-eval", action="store_true")
    ap.add_argument(
        "--train-eval-n",
        type=int,
        default=150,
        help="validation examples for the training loss (early stopping)",
    )
    ap.add_argument(
        "--export-only", action="store_true", help="only export GGUF from <data>/<task>_lora"
    )
    args = ap.parse_args(argv)

    import torch

    data_dir, out = find_dirs(args.data, args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run_log.json"
    log: dict[str, Any] = json.loads(log_path.read_text()) if log_path.exists() else {}
    log.update({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                "base_model": BASE_MODEL, "versions": versions(), "seed": SEED})  # fmt: skip
    session_start = time.time()
    if args.export_only:
        from unsloth import FastLanguageModel

        for task in args.tasks:
            model, tok = FastLanguageModel.from_pretrained(str(data_dir / f"{task}_lora"),
                                                           max_seq_length=args.max_len,
                                                           load_in_4bit=True)  # fmt: skip
            log.setdefault(task, {})["gguf"] = export(task, model, tok, out)
            log_path.write_text(json.dumps(log, indent=1))
            del model
            torch.cuda.empty_cache()
        log.setdefault("session_minutes", []).append(round((time.time() - session_start) / 60, 1))
        log_path.write_text(json.dumps(log, indent=1))
        print(json.dumps({t: log.get(t, {}).get("gguf") for t in args.tasks}))
        return 0
    for task in args.tasks:
        entry = log.setdefault(task, {})
        tr = read_jsonl(data_dir / f"{task}_train.jsonl")
        va = read_jsonl(data_dir / f"{task}_val.jsonl")
        held = va[: args.eval_n]
        (out / task).mkdir(parents=True, exist_ok=True)
        max_len = max(args.max_len, MAX_LEN.get(task, 0))
        model, tok = load(max_len)
        if not args.skip_base_eval and not done_marker(out, task, "base_eval").exists():
            entry["base"] = evaluate(task, model, tok, held)
            done_marker(out, task, "base_eval").touch()
            log_path.write_text(json.dumps(log, indent=1))
        if not done_marker(out, task, "train").exists():
            result = train(task, model, tok, tr, va[: args.train_eval_n], out, max_len)
            model = result.pop("model")
            entry["train"] = result
            done_marker(out, task, "train").touch()
            log_path.write_text(json.dumps(log, indent=1))
        else:
            from unsloth import FastLanguageModel

            model, tok = FastLanguageModel.from_pretrained(str(out / task / "lora"),
                                                           max_seq_length=max_len,
                                                           load_in_4bit=True)  # fmt: skip
        if not done_marker(out, task, "ft_eval").exists():
            entry["fine_tuned"] = evaluate(task, model, tok, held)
            done_marker(out, task, "ft_eval").touch()
            log_path.write_text(json.dumps(log, indent=1))
        if not done_marker(out, task, "gguf").exists():
            entry["gguf"] = export(task, model, tok, out)
            done_marker(out, task, "gguf").touch()
            log_path.write_text(json.dumps(log, indent=1))
        del model
        torch.cuda.empty_cache()
    sessions = log.setdefault("session_minutes", [])
    sessions.append(round((time.time() - session_start) / 60, 1))
    log_path.write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "versions"}, indent=1))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
