"""Build the multi-decade claims dataset (2002-2024, recent years emphasized) -> data/sim_decades/.

Real crash context per year: NHTSA GES 2002-2015, CRSS 2016-2024. Prices per year: BLS CPI.
Usage: uv run python scripts/simulate_decades.py [--n 141000]
"""

import argparse
import sys

from autoclaim.config import load_carrier_config
from autoclaim.datasets.bls import load_cpi
from autoclaim.lines.auto import crss, ges
from autoclaim.lines.auto.simulator import build
from autoclaim.lines.auto.simulator.eras import build_decades, era_overrides, recency_weight
from autoclaim.lines.auto.simulator.world import load_world
from autoclaim.paths import data_dir, raw_dir

YEARS = tuple(range(2002, 2025))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=141_000)
    args = ap.parse_args()
    world = load_world()
    jur = load_carrier_config().active_jurisdiction
    cpi = load_cpi(raw_dir("bls_cpi"))
    incidents = {y: (ges.load_year(y) if y <= 2015 else crss.load_year(y)) for y in YEARS}
    claims = build_decades(world, incidents, cpi, jur, args.n)
    out = data_dir() / "sim_decades"
    build.write(
        {"claims": claims},
        out,
        {
            "years": [YEARS[0], YEARS[-1]],
            "sources": "GES 2002-2015 + CRSS 2016-2024 (NHTSA, public domain); BLS CPI",
            "recency_weights": {str(y): recency_weight(y) for y in YEARS},
            "era_price_factors": {
                str(y): era_overrides(world, y, cpi)["new_vehicle_price"] for y in YEARS
            },
            "real_incident_pool": {str(y): len(v) for y, v in incidents.items()},
            "seed": world.seed,
        },
    )
    by_year = claims.groupby(claims["loss_date"].dt.year).agg(
        n=("claim_id", "size"),
        real_share=("record_origin", lambda s: (s == "crss").mean()),
        fraud=("gt_is_fraud", "mean"),
        median_acv=("vehicle_acv", "median"),
        median_claim=("claimed_amount", "median"),
        total_loss=("gt_total_loss", "mean"),
        rideshare=("use_at_loss", lambda s: s.isin(["rideshare_active", "delivery_active"]).mean()),
        adas=("adas", "mean"),
    )
    print(by_year.round(3).to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
