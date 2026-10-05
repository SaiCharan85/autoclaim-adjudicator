"""Build chat-format fine-tuning sets (no LLM calls) -> data/finetune/<task>_<split>.jsonl.

Tasks: intake (narrative -> true facts) and judge (case -> yes/no answers, from planted errors).
Train = train period, val = validation period; the locked test period is never used.

Usage:
  uv run python scripts/build_finetune_data.py [--judge-samples 400] [--stats-only]
"""

import argparse
import json
import sys
from collections import Counter

import pandas as pd

from autoclaim.config import load_carrier_config
from autoclaim.finetune import datasets as ft
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.policy import POLICY_PATH
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.llm.usage import estimate_tokens
from autoclaim.paths import data_dir
from autoclaim.retrieval.corpus import load_policy


def tokens(examples: list[dict]) -> int:
    return sum(estimate_tokens("".join(m["content"] for m in ex["messages"])) for ex in examples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--judge-samples", type=int, default=400, help="clean train cases")
    ap.add_argument("--judge-val-samples", type=int, default=100, help="clean validation cases")
    ap.add_argument("--stats-only", action="store_true", help="print sizes, write nothing")
    args = ap.parse_args()

    cfg = load_carrier_config()
    split_cfg = cfg.fraud_model.datasets[cfg.fraud_model.production_dataset]
    assert split_cfg.val_start is not None and split_cfg.test_start is not None
    val_start, test_start = split_cfg.val_start, split_cfg.test_start
    claims = sim_build.load("claims")
    seed = cfg.harness.seed

    path = data_dir() / "sim" / "packages" / "dev.jsonl"
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    intake = ft.split(ft.intake_examples(records, claims), val_start, test_start)

    policy, jur = load_policy(POLICY_PATH), cfg.active_jurisdiction
    srcs = dict(zip(claims["claim_id"], claims["source_record"].astype(str), strict=True))
    dates = {
        cid: str(pd.Timestamp(d).date())
        for cid, d in zip(claims["claim_id"], claims["loss_date"], strict=True)
    }
    judge: dict[str, list[dict]] = {}
    windows = {"train": ("2000-01-01", str(val_start)), "val": (str(val_start), str(test_start))}
    sizes = {"train": args.judge_samples, "val": args.judge_val_samples}
    for part, (start, end) in windows.items():
        samples = je.select_samples(claims, start, end, sizes[part], jur, policy, seed)
        examples = ft.judge_examples(samples, je.ERROR_TYPES, seed, srcs, dates)
        judge[part] = ft.split(examples, val_start, test_start)[part]
    used = {ex["meta"]["source_record"] for ex in judge["train"]}
    judge["val"] = [ex for ex in judge["val"] if ex["meta"]["source_record"] not in used]

    out = data_dir() / "finetune"
    for task, parts in (("intake", intake), ("judge", judge)):
        for part, examples in parts.items():
            kinds = (
                Counter(str(ex["meta"].get("error_type")) for ex in examples)
                if task == "judge"
                else ""
            )
            size = f"{len(examples)} examples, ~{tokens(examples):,} tokens"
            print(f"{task}/{part}: {size} {dict(kinds) or ''}")
            if not args.stats_only:
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{task}_{part}.jsonl").write_text(ft.to_jsonl(examples), encoding="utf-8")
    if not args.stats_only:
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
