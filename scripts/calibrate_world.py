"""Calibrate the simulator's assumed amount dials to cited real 2024 figures (grid search).

Targets (config/calibration_sources.yaml): average paid collision claim, average paid
comprehensive claim (III / ISO-Verisk, 2024) and total-loss share (CCC, ~23%). Measured on the
approved claims of a 2022-24 build, 2024 losses only. Prints the grid and the best dials; the
chosen values are then written into config/simulator_auto.yaml by hand (reviewed, not automatic).

Usage: uv run python scripts/calibrate_world.py [--n 50000]
"""

import argparse
import itertools
import sys

import pandas as pd
import yaml

from autoclaim.config import load_carrier_config
from autoclaim.datasets.profile import to_markdown
from autoclaim.lines.auto.crss import load_incidents
from autoclaim.lines.auto.simulator.build import with_ground_truth
from autoclaim.lines.auto.simulator.generate import build_claims
from autoclaim.lines.auto.simulator.world import World, load_world, with_overrides
from autoclaim.paths import REPO_ROOT

COLLISION = ("collision_vehicle", "collision_object", "parked_hit", "hit_and_run")
COMPREHENSIVE = ("animal", "vandalism", "hail_weather", "glass", "fire")


def candidate(
    world: World, m_coll: float, s_add: float, m_comp: float, shift: float, dep: float
) -> World:
    frac = {c: list(v) for c, v in world.damage_fraction.items()}
    for c in COLLISION:
        frac[c] = [round(frac[c][0] * m_coll, 4), round(frac[c][1] + s_add, 3)]
    for c in COMPREHENSIVE:
        frac[c] = [round(frac[c][0] * m_comp, 4), frac[c][1]]
    extent = {
        e: [round(m * m_coll, 4), round(s + s_add, 3)]
        for e, (m, s) in world.damage_by_extent.items()
    }
    causes = dict(world.loss_causes)
    causes["theft"] = round(causes["theft"] - shift, 4)
    causes["glass"] = round(causes["glass"] + shift, 4)
    return with_overrides(
        world,
        {
            "damage_fraction": frac,
            "damage_by_extent": extent,
            "loss_causes": causes,
            "depreciation_per_year": dep,
        },
    )


def measure(claims: pd.DataFrame) -> dict[str, float]:
    c = claims[(claims["loss_date"].dt.year == 2024) & (claims["gt_decision"] == "approve")]
    part = c["gt_coverage_part"]
    return {
        "collision_severity": float(c.loc[part == "collision", "gt_payout"].mean()),
        "comprehensive_severity": float(c.loc[part == "comprehensive", "gt_payout"].mean()),
        "total_loss_share": float(c["gt_total_loss"].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50_000)
    args = ap.parse_args()
    src = yaml.safe_load((REPO_ROOT / "config" / "calibration_sources.yaml").read_text("utf-8"))
    target = {
        "collision_severity": src["collision_claim_severity_usd"][0]["value"],
        "comprehensive_severity": src["comprehensive_claim_severity_usd"][0]["value"],
        "total_loss_share": src["total_loss_share"][0]["value"],
    }
    world, jur = load_world(), load_carrier_config().active_jurisdiction
    incidents = load_incidents()
    causes = tuple(jur.comprehensive_causes)
    rows = []
    # Refined after a first pass (m_coll 1-1.5, s_add 0-0.3, m_comp 0.7/1, shift 0/0.02): cost
    # alone could not reach the total-loss share without overshooting collision severity.
    # Depreciation is the missing dial: older cars are worth less, so more cross 75% of value.
    grid = itertools.product((1.1, 1.25, 1.4), (0.3, 0.45), (0.7,), (0.02,), (0.12, 0.15, 0.18))
    for m_coll, s_add, m_comp, shift, dep in grid:
        w = candidate(world, m_coll, s_add, m_comp, shift, dep)
        claims = with_ground_truth(
            build_claims(w, incidents, n=args.n, comprehensive_causes=causes), jur
        )
        m = measure(claims)
        err = sum((m[k] / target[k] - 1) ** 2 for k in target)
        rows.append(
            {
                "depreciation": dep,
                "m_coll": m_coll,
                "s_add": s_add,
                "m_comp": m_comp,
                "theft_to_glass": shift,
                **{k: round(v, 3) for k, v in m.items()},
                "sq_rel_error": round(err, 4),
            }
        )
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows).sort_values("sq_rel_error")
    out = REPO_ROOT / "docs" / "calibration.md"
    out.write_text(
        "# Simulator calibration (2024 targets)\n\n"
        f"Targets: {target} (sources: config/calibration_sources.yaml). Approved claims, "
        f"2024 losses, n={args.n} build.\n\n"
        + to_markdown(res.head(12).set_index("depreciation"), "depreciation")
        + "\n",
        encoding="utf-8",
    )
    print(res.head(5).to_string(index=False))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
