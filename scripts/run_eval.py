"""Harness evaluation: arms on the same claims, scored against truth -> eval/report_<set>.md.

Arms (each has its own ledger/checkpoints, so arms never share decisions; identical LLM calls
are shared through the exact-match cache, so later arms pay only where their path differs):
  full       the production harness
  no_judge   judge always passes (what does the judge add?)
  no_critic  critic always passes (what do the code checks add?)
  few_shot   adjudicator gets similar past adjuster decisions from feedback memory (k=3)

Sets: dev = validation-period claims (development; repeatable); eval = test-period claims (the
locked test; needs --final and is logged). Escalations are answered by the simulated adjuster.
Feedback memory is read-only during evaluation; fill it first from TRAIN-period claims:
  uv run python scripts/run_eval.py --build-memory 40 [--dry-run]

Usage:
  uv run python scripts/run_eval.py --set dev --n 50 --arms full no_judge --dry-run
  uv run python scripts/run_eval.py --set dev --n 50 --arms full no_judge no_critic few_shot
  uv run python scripts/run_eval.py --set eval --final --n 300 --arms full few_shot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from autoclaim.config import load_carrier_config
from autoclaim.core.review import resume
from autoclaim.lines.auto import harness_eval as he
from autoclaim.lines.auto.build import build
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.lines.auto.simulator.adjuster import OracleAdjuster
from autoclaim.ml import test_log
from autoclaim.paths import REPO_ROOT, data_dir

CACHE = REPO_ROOT / ".cache"
ARMS = ("full", "no_judge", "no_critic", "few_shot")
# New (uncached) LLM calls per claim, measured on the first live smoke run (docs/harness.md):
# the first arm pays everything; later arms reuse identical calls from the cache.
NEW_CALLS = {"full": 4.5, "no_judge": 0.3, "no_critic": 0.8, "few_shot": 2.3}
TOKENS_PER_CALL = 2700
FEW_SHOT_K = 3


def packages(name: str, start: str, end: str) -> list[dict[str, Any]]:
    path = data_dir() / "sim" / "packages" / f"{name}.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return [r for r in recs if start <= r["package"]["report_date"] < end]


def run_claim(app: Any, pkg: dict[str, Any], adjuster: OracleAdjuster) -> dict[str, Any]:
    cid = pkg["claim_id"]
    config = {"configurable": {"thread_id": cid}}
    out: dict[str, Any] = app.graph.invoke({"claim_id": cid, "claim": pkg}, config)
    if "final" not in out:
        out = resume(app.graph, cid, adjuster.decide(out["__interrupt__"][0].value))
    return out


def load_scores(path: Path) -> list[he.ClaimScore]:
    if not path.exists():
        return []
    return [he.ClaimScore.model_validate_json(x) for x in path.read_text("utf-8").splitlines() if x]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", choices=["dev", "eval"], default="dev")
    ap.add_argument("--final", action="store_true", help="required for the test-period set")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=["full"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument(
        "--build-memory",
        type=int,
        default=0,
        metavar="N",
        help="first run N train-period claims (full arm) to fill feedback memory",
    )
    ap.add_argument("--error-rate", type=float, default=0.05, help="simulated adjuster mistakes")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.set == "eval" and not args.final:
        ap.error("the eval set is the locked test period: pass --final (it is logged)")

    cfg = load_carrier_config()
    split = cfg.fraud_model.datasets[cfg.fraud_model.production_dataset]
    val, test = str(split.val_start), str(split.test_start)
    claims = sim_build.load("claims")
    truth = {r["claim_id"]: he.truth_of(r) for _, r in claims.iterrows()}
    adjuster = OracleAdjuster(claims, args.error_rate, cfg.harness.seed)
    load_dotenv(REPO_ROOT / ".env")

    if args.build_memory:
        pool = packages("dev", "2000-01-01", val)[: args.build_memory]
        calls = len(pool) * NEW_CALLS["full"]
        print(f"memory build: {len(pool)} train-period claims, ~{calls:.0f} LLM calls, "
              f"~{calls * TOKENS_PER_CALL:,.0f} tokens")  # fmt: skip
        if args.dry_run:
            return 0
        app = build(cfg, state_dir=CACHE / "eval" / "memory_build")
        for rec in pool:
            if he.quota_exhausted(run_claim(app, rec["package"], adjuster)):
                print(
                    "free-tier quota exhausted: stopping; re-run tomorrow (cached calls are free)"
                )
                return 2
        print(f"feedback memory: {len(app.harness.memory)} episodes")  # type: ignore[arg-type]
        return 0

    name, (start, end) = (("eval", (test, "2030-01-01")) if args.set == "eval"
                          else ("dev", (val, test)))  # fmt: skip
    pool = packages(name, start, end)[: args.n]
    root = CACHE / "eval" / args.set
    todo = {a: [r for r in pool if r["package"]["claim_id"]
                not in {s.claim_id for s in load_scores(root / a / "scores.jsonl")}]
            for a in args.arms}  # fmt: skip
    calls = sum(len(todo[a]) * NEW_CALLS[a] for a in args.arms)
    print(f"{len(pool)} claims ({args.set} set), arms {args.arms}; still to run "
          f"{ {a: len(t) for a, t in todo.items()} }; ~{calls:.0f} new LLM calls, "
          f"~{calls * TOKENS_PER_CALL:,.0f} tokens (later arms reuse cached calls)")  # fmt: skip
    if args.dry_run:
        return 0

    arms: dict[str, list[he.ClaimScore]] = {}
    for arm in args.arms:
        out_path = root / arm / "scores.jsonl"
        scores = load_scores(out_path)
        arm_cfg = cfg
        if arm == "few_shot" and cfg.memory is not None:
            arm_cfg = cfg.model_copy(update={"memory": cfg.memory.model_copy(
                update={"few_shot_k": FEW_SHOT_K})})  # fmt: skip
        wrap = None
        if arm in ("no_judge", "no_critic"):

            def wrap(line: Any, arm: str = arm) -> Any:
                return he.AblatedLine(line, no_judge=arm == "no_judge",
                                      no_critic=arm == "no_critic")  # fmt: skip

        app = build(arm_cfg, read_only_memory=True, state_dir=root / arm, wrap_line=wrap)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for rec in todo[arm]:
                pkg = rec["package"]
                state = run_claim(app, pkg, adjuster)
                if he.quota_exhausted(state):
                    print(f"  [{arm}] free-tier quota exhausted at {pkg['claim_id']}: stopping; "
                          "re-run tomorrow to resume (finished claims are kept)")  # fmt: skip
                    return 2
                s = he.score(state, truth[pkg["claim_id"]], arm)
                fh.write(s.model_dump_json() + "\n")
                fh.flush()
                scores.append(s)
                print(f"  [{arm}] {s.claim_id}: {'auto' if s.auto else 'human'} {s.final_outcome}"
                      f" (truth {s.truth.outcome}) calls={s.llm_calls}", flush=True)  # fmt: skip
        ids = {r["package"]["claim_id"] for r in pool}
        arms[arm] = [s for s in scores if s.claim_id in ids]

    report = REPO_ROOT / "eval" / f"report_{args.set}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    table = he.markdown_table(arms, args.boot, cfg.harness.seed)
    header = (
        f"# Harness evaluation ({args.set} set, {len(pool)} claims)\n\n"
        f"Arms on the same claims; 95% bootstrap CIs over claims; differences are paired. "
        f"Headline metrics were fixed before any run (`lines/auto/harness_eval.py`). "
        f"Escalations answered by the simulated adjuster (error rate {args.error_rate}).\n\n"
    )
    report.write_text(header + table + "\n", encoding="utf-8")
    print(table)
    print(f"report -> {report}")
    if args.final:
        full = arms.get(args.arms[0], [])
        test_log.append(
            "harness (claims, H2 2024)",
            f"arm {args.arms[0]}: auto_rate {he.metric(full, 'auto_rate')}, "
            f"auto_accuracy {he.metric(full, 'auto_accuracy')}, n={len(full)}",
        )
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    sys.exit(main())
