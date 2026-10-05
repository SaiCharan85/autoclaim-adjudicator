"""Claims waiting for human review: list them, or answer them with the simulated adjuster.

Resuming costs no LLM calls: the graph finalizes the adjuster's decision and feedback memory
stores the episode (local embeddings). Claims from the evaluation period are finalized but never
remembered (memory cutoff).

Usage:
  uv run python scripts/review_queue.py --list
  uv run python scripts/review_queue.py --adjuster oracle [--error-rate 0.05] [--limit 20]
"""

import argparse
import sys

from autoclaim.config import load_carrier_config
from autoclaim.core.memory import FeedbackMemory
from autoclaim.core.review import pending, resume
from autoclaim.lines.auto.build import build
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.lines.auto.simulator.adjuster import OracleAdjuster


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show waiting claims and exit")
    ap.add_argument("--adjuster", choices=["oracle"], help="answer waiting claims")
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not args.list and not args.adjuster:
        ap.error("use --list or --adjuster")

    cfg = load_carrier_config()
    app = build(cfg)
    queue = pending(app.graph)
    print(f"{len(queue)} claims waiting for human review")
    for claim_id, req in queue:
        prop = req.get("proposed_decision") or {}
        print(f"  {claim_id}: {', '.join(req['route_reasons'])}; proposed "
              f"{prop.get('outcome')} {prop.get('reasons')}")  # fmt: skip
    if args.list:
        return 0

    adjuster = OracleAdjuster(sim_build.load("claims"), args.error_rate, cfg.harness.seed)
    for claim_id, req in queue[: args.limit]:
        decision = adjuster.decide(req)
        out = resume(app.graph, claim_id, decision)
        final = out.get("final", {})
        print(f"  {claim_id}: {final.get('decided_by')} {final.get('outcome')} "
              f"payout={final.get('payout')} ({decision.reason})")  # fmt: skip
    memory = app.harness.memory
    if isinstance(memory, FeedbackMemory):
        skipped = memory.skipped_after_cutoff
        print(f"feedback memory: {len(memory)} episodes "
              f"({skipped} evaluation-period claims not remembered)")  # fmt: skip
    print(f"simulated adjuster mistakes: {len(adjuster.errors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
