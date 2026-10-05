"""Multi-decade world (2002-2024): one simulated year per real crash year, era-correct.

The base world (config/simulator_auto.yaml) describes the 2023-26 market. For an earlier year:
- vehicle prices scale with the BLS new-vehicle index; minimum value with used vehicles;
- repair costs are fractions of vehicle value, scaled by repair prices RELATIVE to new-vehicle
  prices (repairs rose far faster: BLS repair index 2002 -> 2024 is x2.1, new vehicles x1.27);
- era facts: rideshare/delivery use starts 2012, rideshare endorsements 2015, driver-assist
  sensors spread from 2012; the Kia/Hyundai theft wave stays 2022-24 (keyed by year already).
Fraud mechanisms are the same every year (no real per-year fraud labels exist). Recent years get
more claims (recency emphasis; user requirement 2026-10-04).
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from autoclaim.config import JurisdictionProfile
from autoclaim.lines.auto.simulator.generate import build_claims
from autoclaim.lines.auto.simulator.oracle import adjudicate
from autoclaim.lines.auto.simulator.world import World, with_overrides

BASE_YEAR = 2024  # the base world's price level


def _ramp(year: int, start: int, full: int) -> float:
    """0 before `start`, linear to 1 at `full`, 1 after."""
    return float(np.clip((year - start) / max(full - start, 1), 0.0, 1.0))


def era_overrides(world: World, year: int, cpi: pd.DataFrame) -> dict[str, object]:
    base, now = cpi.loc[BASE_YEAR], cpi.loc[year]
    price = now["new_vehicles"] / base["new_vehicles"]
    used = now["used_vehicles"] / base["used_vehicles"]
    repair_rel = (now["repair"] / base["repair"]) / price  # repair cost relative to car value
    gig = _ramp(year, 2012, 2020)  # Uber/Lyft from ~2012, delivery apps from ~2013
    use = dict(world.use_at_loss)
    business = use["rideshare_active"] + use["delivery_active"]
    use["rideshare_active"] *= gig
    use["delivery_active"] *= gig
    use["personal"] += business * (1 - gig)
    adas = dict(world.adas_rate_by_age)
    adas_scale = _ramp(year, 2012, 2022)
    adas["new"] = round(adas["new"] * adas_scale, 4)
    adas["old"] = round(adas["old"] * adas_scale, 4)
    return {
        "loss_date_start": f"{year}-01-01",
        "loss_date_end": f"{year}-12-31",
        "body_new_price": {k: round(v * price, 2) for k, v in world.body_new_price.items()},
        "new_vehicle_price": {
            "median": round(world.new_vehicle_price.median * price, 2),
            "sigma": world.new_vehicle_price.sigma,
        },
        "min_acv": round(world.min_acv * used, 2),
        "damage_fraction": {
            c: [round(m * repair_rel, 4), s]
            for c, (m, s) in world.damage_fraction.items()
            if c != "theft"
        }
        | {"theft": list(world.damage_fraction["theft"])},
        "damage_by_extent": {
            e: [round(m * repair_rel, 4), s] for e, (m, s) in world.damage_by_extent.items()
        },
        "adas_glass_recalibration_usd": round(
            world.adas_glass_recalibration_usd * now["repair"] / base["repair"], 2
        ),
        "use_at_loss": use,
        "rideshare_endorsement_rate": round(
            world.rideshare_endorsement_rate * _ramp(year, 2015, 2022), 4
        ),
        "adas_rate_by_age": adas,
        # the business-use trap needs rideshare/delivery apps to exist
        "traps": world.traps
        | {
            "business_use_no_endorsement": round(
                world.traps["business_use_no_endorsement"] * gig, 5
            )
        },
    }


def year_counts(years: list[int], total: int, weights: Mapping[int, float]) -> dict[int, int]:
    w = np.array([weights[y] for y in years], dtype=float)
    n = np.floor(total * w / w.sum()).astype(int)
    n[-1] += total - n.sum()  # rounding remainder to the newest year
    return dict(zip(years, n.tolist(), strict=True))


def recency_weight(year: int) -> float:
    """Claims per year relative to the 2000s: 2000s 1, 2010-15 1.5, 2016-19 2.5, 2020s 4."""
    if year >= 2020:
        return 4.0
    if year >= 2016:
        return 2.5
    return 1.5 if year >= 2010 else 1.0


def build_decades(
    world: World,
    incidents_by_year: Mapping[int, pd.DataFrame],
    cpi: pd.DataFrame,
    jur: JurisdictionProfile,
    total: int,
) -> pd.DataFrame:
    years = sorted(incidents_by_year)
    counts = year_counts(years, total, {y: recency_weight(y) for y in years})
    causes = tuple(jur.comprehensive_causes)
    frames = []
    for year in years:
        w = with_overrides(world, era_overrides(world, year, cpi))
        claims = build_claims(
            w,
            incidents_by_year[year],
            n=counts[year],
            seed=world.seed + year,
            comprehensive_causes=causes,
        )
        claims["claim_id"] = f"CLM-{year}-" + claims["claim_id"].str[4:]
        claims["policy_id"] = f"POL-{year}-" + claims["policy_id"].str[4:]
        frames.append(pd.concat([claims, adjudicate(claims, jur)], axis=1))
    return pd.concat(frames, ignore_index=True)
