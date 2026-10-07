"""Flood line on real FEMA NFIP claims, through the same harness graph as auto.

Report: docs/flood_eval.md (development) or docs/flood_eval_test.md (--final).
Template mode makes no LLM calls (code decides and explains), so it runs thousands of real claims
at $0. Development uses losses before the flood test start (fixed in config); --final scores the
later period once and logs it in docs/test_set_log.md.

Usage:
  uv run python scripts/download_data.py fema_nfip       # once (OpenFEMA, free)
  uv run python scripts/flood_eval.py --n 3000            # development period
  uv run python scripts/flood_eval.py --n 3000 --final    # locked test period, once
"""

import argparse
import sys
import time

from autoclaim.config import load_carrier_config
from autoclaim.datasets import fema_nfip
from autoclaim.lines.flood import evaluate as fe
from autoclaim.lines.flood.build import build_flood
from autoclaim.lines.flood.claim import from_nfip
from autoclaim.ml import test_log
from autoclaim.paths import REPO_ROOT, raw_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=3000, help="claims sampled from the period")
    ap.add_argument("--final", action="store_true", help="locked test period (logged)")
    ap.add_argument("--rerun-final", action="store_true", help="allow a second locked-test look")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_carrier_config()
    test_start = (cfg.model_extra or {})["flood"]["test_start"]
    if args.final:
        test_log.guard_final(
            f"flood (FEMA NFIP, losses >= {test_start})", "auto_rate", args.rerun_final
        )
    claims = fema_nfip.load(raw_dir("fema_nfip"))
    period = claims["dateOfLoss"] >= test_start if args.final else claims["dateOfLoss"] < test_start
    pool = claims[period]
    sample = pool.sample(n=min(args.n, len(pool)), random_state=cfg.harness.seed)
    _, graph = build_flood(cfg)
    t0 = time.time()
    scores = []
    for _, row in sample.iterrows():
        claim = from_nfip(row)
        payload = claim.model_dump(mode="json")
        out = graph.invoke({"claim_id": claim.claim_id, "claim": payload},
                           {"configurable": {"thread_id": claim.claim_id}})  # fmt: skip
        scores.append(fe.score(out, row, claim.replacement_cost_basis))
    secs = time.time() - t0
    est = fe.summarize(scores, args.boot, cfg.harness.seed)
    name = "locked test" if args.final else "development"
    lines = [
        f"# Flood line on real FEMA NFIP claims ({name} period)",
        "",
        f"{len(scores)} claims sampled (seed {cfg.harness.seed}) from {len(pool):,} with losses "
        f"{'on or after' if args.final else 'before'} {test_start}, run through the same harness "
        f"graph as the auto line with the flood line plugged in; no LLM calls ({secs:.0f} s).",
        "Outcomes are compared with what FEMA actually paid. Escalated claims count as not "
        "auto-decided.",
        "",
        "| metric | value | 95% CI |",
        "|---|---|---|",
    ]
    for m, e in est.items():
        v = (
            "n/a"
            if e.value is None
            else f"{e.value:.1%}"
            if m != "rc_first_payment_ratio"
            else f"{e.value:.3f}"
        )
        ci = ("n/a" if e.low is None or e.high is None
              else f"[{e.low:.3f}, {e.high:.3f}]" if m == "rc_first_payment_ratio"
              else f"[{e.low:.1%}, {e.high:.1%}]")  # fmt: skip
        lines.append(f"| {m} | {v} | {ci} |")
    lines += ["", "Why claims went to an adjuster (a claim can have several reasons):", ""]
    lines += [f"- {r}: {c}" for r, c in list(fe.route_counts(scores).items())[:8]]
    lines += ["", f"_{fema_nfip.DISCLAIMER}_ Source: {fema_nfip.API}. Historical, closed claims "
              "only; no determination about any person is made."]  # fmt: skip
    out_path = REPO_ROOT / "docs" / ("flood_eval_test.md" if args.final else "flood_eval.md")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:]))
    print(f"report -> {out_path}")
    if args.final:
        test_log.append("flood (FEMA NFIP, losses >= " + test_start + ")",
                        f"auto_rate {fe.metric(scores, 'auto_rate'):.3f}, auto_agreement "
                        f"{fe.metric(scores, 'auto_agreement'):.3f}, n={len(scores)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
