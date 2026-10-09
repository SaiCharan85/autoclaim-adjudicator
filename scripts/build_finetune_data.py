"""Build chat-format fine-tuning sets (no LLM calls) -> data/finetune/<task>_<split>.jsonl.

Tasks: intake (narrative -> true facts), judge (case -> yes/no answers, from planted errors) and
adjudicator (the production adjudicator prompt -> the true decision; loads the retriever and the
fraud models, ~0.3 s per example on CPU).
Train = train period, val = validation period; the locked test period is never used.

Usage:
  uv run python scripts/build_finetune_data.py [--tasks intake judge adjudicator] [--stats-only]
"""

import argparse
import json
import sys
from collections import Counter
from typing import cast

import pandas as pd

from autoclaim.config import CarrierConfig, load_carrier_config
from autoclaim.core.judge import build_judge, load_rubric
from autoclaim.finetune import datasets as ft
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH, build_retriever, harness_config
from autoclaim.lines.auto.fraud_tools import FraudToolkit
from autoclaim.lines.auto.line import AutoLine
from autoclaim.lines.auto.policy import POLICY_PATH
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.llm.client import LLMClient
from autoclaim.llm.usage import estimate_tokens
from autoclaim.paths import data_dir
from autoclaim.retrieval.corpus import load_policy


def tokens(examples: list[dict]) -> int:
    return sum(estimate_tokens("".join(m["content"] for m in ex["messages"])) for ex in examples)


TASKS = ("intake", "judge", "adjudicator")


def gold_line(cfg: CarrierConfig) -> tuple[AutoLine, float]:
    """The production auto line (retriever, fraud models) with the gold stand-in client."""
    r, hc = build_retriever(cfg), harness_config(cfg)
    client = cast(LLMClient, ft.GoldClient())
    line = AutoLine(client=client, policy=r.policy, retriever=r, toolkit=FraudToolkit.load(),
                    judge_impl=build_judge(client, load_rubric(RUBRIC_PATH)),
                    jurisdiction=cfg.active_jurisdiction,
                    fraud_review_score=hc.router.fraud_review_score, top_k=cfg.retrieval.top_k,
                    graph_hops=cfg.retrieval.graph_hops,
                    max_expanded=cfg.retrieval.max_expanded)  # fmt: skip
    return line, hc.router.fraud_review_score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--judge-samples", type=int, default=400, help="clean train cases")
    ap.add_argument("--judge-val-samples", type=int, default=100, help="clean validation cases")
    ap.add_argument("--adjudicator-samples", type=int, default=650, help="train examples")
    ap.add_argument("--adjudicator-val-samples", type=int, default=150)
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--stats-only", action="store_true", help="print sizes, write nothing")
    args = ap.parse_args()

    cfg = load_carrier_config()
    split_cfg = cfg.fraud_model.datasets[cfg.fraud_model.production_dataset]
    assert split_cfg.val_start is not None and split_cfg.test_start is not None
    val_start, test_start = split_cfg.val_start, split_cfg.test_start
    claims = sim_build.load("claims")
    seed = cfg.harness.seed

    sets: dict[str, dict[str, list[dict]]] = {}
    if "intake" in args.tasks:
        path = data_dir() / "sim" / "packages" / "dev.jsonl"
        records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        sets["intake"] = ft.split(ft.intake_examples(records, claims), val_start, test_start)

    policy, jur = load_policy(POLICY_PATH), cfg.active_jurisdiction
    srcs = dict(zip(claims["claim_id"], claims["source_record"].astype(str), strict=True))
    dates = {
        cid: str(pd.Timestamp(d).date())
        for cid, d in zip(claims["claim_id"], claims["loss_date"], strict=True)
    }
    windows = {"train": ("2000-01-01", str(val_start)), "val": (str(val_start), str(test_start))}
    if "judge" in args.tasks:
        judge: dict[str, list[dict]] = {}
        sizes = {"train": args.judge_samples, "val": args.judge_val_samples}
        for part, (start, end) in windows.items():
            samples = je.select_samples(claims, start, end, sizes[part], jur, policy, seed)
            examples = ft.judge_examples(samples, je.ERROR_TYPES, seed, srcs, dates)
            judge[part] = ft.split(examples, val_start, test_start)[part]
        used = {ex["meta"]["source_record"] for ex in judge["train"]}
        judge["val"] = [ex for ex in judge["val"] if ex["meta"]["source_record"] not in used]
        sets["judge"] = judge
    if "adjudicator" in args.tasks:
        line, review = gold_line(cfg)
        adj: dict[str, list[dict]] = {}
        sizes = {"train": args.adjudicator_samples, "val": args.adjudicator_val_samples}
        for part, (start, end) in windows.items():
            examples = ft.adjudicator_examples(claims, start, end, sizes[part], line, jur,
                                               review, seed)  # fmt: skip
            adj[part] = ft.split(examples, val_start, test_start)[part]
        used = {ex["meta"]["source_record"] for ex in adj["train"]}
        adj["val"] = [ex for ex in adj["val"] if ex["meta"]["source_record"] not in used]
        sets["adjudicator"] = adj

    out = data_dir() / "finetune"
    for task, parts in sets.items():
        for part, examples in parts.items():
            key = {"judge": "error_type", "adjudicator": "primary_reason"}.get(task)
            kinds = Counter(str(ex["meta"].get(key)) for ex in examples) if key else ""
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
