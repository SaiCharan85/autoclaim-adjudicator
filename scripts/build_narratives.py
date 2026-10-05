"""Build claim packages with LLM-written narratives -> data/sim/packages/<set>.jsonl.

Sets (time order is the leakage rule): `eval` = test period (H2 2024) for harness evaluation;
`dev` = train + validation periods, for prompt development, few-shot, feedback memory and
fine-tuning data.

Usage:
  uv run python scripts/build_narratives.py --set eval --n 50 --dry-run
  uv run python scripts/build_narratives.py --set eval --n 50      # pilot (reused by the full run)
  uv run python scripts/build_narratives.py --set eval --n 300
"""

import argparse
import json
import sys

from dotenv import load_dotenv

from autoclaim.config import load_carrier_config
from autoclaim.lines.auto.narratives import (
    INSTRUCTIONS,
    ROLE,
    NarrativeBatch,
    batch_prompt,
    generate,
    select_claims,
)
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.llm.client import LLMClient, system_prompt
from autoclaim.paths import REPO_ROOT, data_dir

SETS = {"eval": ("2024-07-01", "2030-01-01"), "dev": ("2000-01-01", "2024-07-01")}


def main() -> int:
    cfg = load_carrier_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    claims = sim_build.load("claims")
    start, end = SETS[args.set]
    seed = cfg.harness.seed + (0 if args.set == "eval" else 1)
    # select from the full-size draw, then take a prefix: a pilot is the first N of the full run
    rows = select_claims(claims, start, end, max(args.n, 300), seed).iloc[: args.n]
    load_dotenv(REPO_ROOT / ".env")
    client = LLMClient.from_config(cfg.models)
    system = system_prompt(INSTRUCTIONS, NarrativeBatch)
    est_tokens, n_req = 0, 0
    for i in range(0, len(rows), args.batch_size):
        chunk = [r for _, r in rows.iloc[i : i + args.batch_size].iterrows()]
        est_tokens += client.estimate_tokens(ROLE, system, batch_prompt(chunk)[0])
        n_req += 1
    first = cfg.models.roles[ROLE].chain[0]
    print(f"{len(rows)} claims ({args.set}), {n_req} requests, <= {est_tokens} tokens, "
          f"model {first}, headroom today {client.ledger.headroom(first)}")  # fmt: skip
    if args.dry_run:
        return 0
    records, failed = generate(client, rows, args.batch_size)
    for rec in records:
        rec["set"] = args.set
    out = data_dir() / "sim" / "packages" / f"{args.set}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    models = sorted({r["generator"] for r in records})
    print(f"wrote {len(records)} packages to {out} (generators {models}); failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
