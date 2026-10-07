"""Planted-error evaluation of the claim judge -> docs/judge_eval_<set>.md.

Clean reference cases come from simulator truth (no LLM); JudgeKit plants five insurance error
types into copies and measures, per type, how often the judge catches them, plus false alarms on
the clean originals, with bootstrap CIs. Every judgment is checkpointed, so an interrupted run
(free-tier quota) resumes without re-spending requests; unavailable cases are retried.

Usage:
  uv run python scripts/judge_eval.py --n 10 --dry-run     # estimate requests/tokens, spend nothing
  uv run python scripts/judge_eval.py --n 10               # pilot (dev period)
  uv run python scripts/judge_eval.py --n 50               # validation run (dev period)
  uv run python scripts/judge_eval.py --n 50 --final       # test period, once; logged
"""

import argparse
import json
import sys
from pathlib import Path

import judgekit
from dotenv import load_dotenv

from autoclaim.config import load_carrier_config, without_providers
from autoclaim.core.judge import ClientLLM, load_rubric
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH
from autoclaim.lines.auto.policy import POLICY_PATH
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.llm.client import LLMClient, system_prompt
from autoclaim.ml import test_log
from autoclaim.paths import REPO_ROOT
from autoclaim.retrieval.corpus import load_policy

# dev = validation window only, so the small judge can train on the train period (no overlap)
PERIODS = {"dev": ("2024-01-01", "2024-07-01"), "eval": ("2024-07-01", "2030-01-01")}
ROLE = "judge"
CACHE = REPO_ROOT / ".cache"


def load_saved(path: Path) -> list[judgekit.CaseResult]:
    """Checkpointed judgments that can be reused (unavailable ones are judged again)."""
    if not path.exists():
        return []
    rows = [
        judgekit.CaseResult.model_validate_json(x)
        for x in path.read_text("utf-8").splitlines()
        if x
    ]
    return [r for r in rows if r.available]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--n", type=int, default=50, help="clean samples (each gets up to 5 planted copies)"
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--final", action="store_true", help="test period; logged in docs/test_set_log.md"
    )
    ap.add_argument("--rerun-final", action="store_true", help="allow a second locked-test look")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    cfg = without_providers(load_carrier_config(), frozenset({"ollama"}))  # API models only
    seed = cfg.harness.seed if args.seed is None else args.seed
    name = "eval" if args.final else "dev"
    if args.final and not args.dry_run:
        test_log.guard_final("judge planted errors (claims, H2 2024)", "catch", args.rerun_final)
    start, end = PERIODS[name]
    policy = load_policy(POLICY_PATH)
    rubric = load_rubric(RUBRIC_PATH)
    jur = cfg.active_jurisdiction
    samples = je.select_samples(sim_build.load("claims"), start, end, args.n, jur, policy, seed)
    suite = judgekit.plant(samples, je.ERROR_TYPES, seed=seed)

    load_dotenv(REPO_ROOT / ".env")
    client = LLMClient.from_config(cfg.models)
    system = system_prompt(judgekit.render_system(rubric), judgekit.JudgeResponse)
    cases = [je.render(s) for s in suite.clean] + [je.render(p.sample) for p in suite.planted]
    tokens = sum(client.estimate_tokens(ROLE, system, c) for c in cases)
    ckpt = CACHE / "runs" / f"judge_eval_{name}_n{args.n}_s{seed}.jsonl"
    saved = load_saved(ckpt)
    todo = suite.n_cases - len({c.case_id for c in saved})
    first = cfg.models.roles[ROLE].chain[0]
    print(
        f"{len(samples)} clean samples ({name} period), planted {suite.counts()}, "
        f"not applicable {suite.not_applicable}"
    )
    print(f"{suite.n_cases} judge cases, {todo} still to judge (checkpoint {ckpt.name}); "
          f"<= {tokens} tokens in total, up to 2 requests per case on a validation retry; "
          f"model {first}, headroom today {client.ledger.headroom(first)}")  # fmt: skip
    if args.dry_run:
        return 0

    adjudicator_family = cfg.models.catalog[cfg.models.roles["adjudicator"].chain[0]].family
    judge = judgekit.LLMJudge(ClientLLM(client, ROLE), rubric, name=ROLE)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with ckpt.open("a", encoding="utf-8") as out:

        def save(case: judgekit.CaseResult) -> None:
            out.write(case.model_dump_json() + "\n")
            out.flush()
            status = "ok" if case.available else f"unavailable ({case.verdict.error[:60]})"
            print(
                f"  {case.case_id}: {'flagged' if case.flagged else 'passed'} {status}", flush=True
            )

        result = judgekit.run_suite(judge, suite, je.render, rubric,
                                    avoid_families=frozenset({adjudicator_family}),
                                    on_case=save, resume=saved)  # fmt: skip

    est = judgekit.suite_estimates(result, n_boot=args.boot, seed=seed)
    unavailable = sum(not c.available for c in result.cases)
    lines = [
        f"# Judge planted-error evaluation ({name} period)",
        "",
        "Clean cases are built from simulator truth (true facts, code-derived numbers, real clause "
        "text, ground-truth decision, a templated explanation that is correct by construction). "
        "JudgeKit plants one error per copy; a catch is a failed verdict, and a catch by the "
        "expected item means the judge failed it for the right reason.",
        "",
        result.to_markdown(),
        "",
        "## With 95% bootstrap CIs (resampled by claim)",
        "",
        "| metric | value | 95% CI |",
        "|---|---|---|",
    ]
    for metric, e in est.items():
        value = "n/a" if e.value is None else f"{e.value:.1%}"
        ci = "n/a" if e.low is None or e.high is None else f"[{e.low:.1%}, {e.high:.1%}]"
        lines.append(f"| {metric} | {value} | {ci} |")
    lines += [
        "",
        "## Which rubric items fired",
        "",
        "```",
        json.dumps(result.item_flags(), indent=1),
        "```",
        "",
    ]
    if unavailable:
        lines.append(f"{unavailable} cases were unavailable (quota); re-run to fill them in.")
    report = REPO_ROOT / "docs" / f"judge_eval_{name}_n{len(samples)}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_markdown())
    print(f"report -> {report}")
    if args.final:
        catch, fa = est["catch"].value, est["false_alarm"].value
        test_log.append(
            "judge planted errors (claims, H2 2024)",
            f"catch {catch:.3f} / false alarm {fa:.3f}, n={len(samples)} samples"
            if catch is not None and fa is not None else "incomplete (unavailable cases)",
        )  # fmt: skip
    return 0 if not unavailable else 1


if __name__ == "__main__":
    sys.exit(main())
