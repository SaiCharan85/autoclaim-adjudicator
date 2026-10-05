"""Run claim packages through the harness (smoke test / demo). Interrupted claims wait for an
adjuster (console or oracle) and can be resumed later by claim id.

Usage: uv run python scripts/run_claims.py --set dev --n 3 [--dry-run]
"""

import argparse
import json
import sys

from dotenv import load_dotenv

from autoclaim.lines.auto.build import build
from autoclaim.paths import REPO_ROOT, data_dir

# Typical per-claim usage (pilot measurements go in docs/harness.md): intake, coverage,
# adjudicator, judge, sometimes fraud; retries add an adjudicator + judge call each.
EST_CALLS, EST_TOKENS = 4.5, 12_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="dev", choices=["dev", "eval"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = data_dir() / "sim" / "packages" / f"{args.set}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records = records[args.offset : args.offset + args.n]
    print(f"{len(records)} claims: ~{EST_CALLS * len(records):.0f} LLM calls, "
          f"~{EST_TOKENS * len(records):,} tokens (before cache)")  # fmt: skip
    if args.dry_run:
        return 0
    load_dotenv(REPO_ROOT / ".env")
    app = build()
    for rec in records:
        pkg = rec["package"]
        config = {"configurable": {"thread_id": pkg["claim_id"]}}
        out = app.graph.invoke({"claim_id": pkg["claim_id"], "claim": pkg}, config)
        if "final" in out:
            f = out["final"]
            print(f"{pkg['claim_id']}: {f['decided_by']} {f['outcome']} payout={f.get('payout')} "
                  f"reasons={f.get('reasons')}")  # fmt: skip
        else:
            req = out["__interrupt__"][0].value
            prop = req.get("proposed_decision") or {}
            print(f"{pkg['claim_id']}: HUMAN REVIEW ({', '.join(req['route_reasons'])}); "
                  f"proposed {prop.get('outcome')} {prop.get('reasons')}")  # fmt: skip
        print(f"   llm_calls={out.get('llm_calls')} tokens={out.get('tokens')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
