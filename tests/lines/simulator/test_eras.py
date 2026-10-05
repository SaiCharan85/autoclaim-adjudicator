import pandas as pd
import pytest

from autoclaim.lines.auto.simulator import eras
from autoclaim.lines.auto.simulator.world import World, with_overrides

CPI = pd.DataFrame(
    {"new_vehicles": [140.0, 180.0], "used_vehicles": [150.0, 180.0], "repair": [190.0, 400.0]},
    index=pd.Index([2002, 2024], name="year"),
)


def test_ramp_and_recency_weights() -> None:
    assert eras._ramp(2010, 2012, 2020) == 0 and eras._ramp(2016, 2012, 2020) == 0.5
    assert eras._ramp(2030, 2012, 2020) == 1
    assert [eras.recency_weight(y) for y in (2005, 2012, 2018, 2022)] == [1.0, 1.5, 2.5, 4.0]


def test_year_counts_sum_exactly() -> None:
    counts = eras.year_counts([2002, 2010, 2020], 1000, {2002: 1, 2010: 1.5, 2020: 4})
    assert sum(counts.values()) == 1000 and counts[2020] > counts[2010] > counts[2002]


def test_era_overrides_2002(world: World) -> None:
    o = eras.era_overrides(world, 2002, CPI)
    w = with_overrides(world, o)  # must stay a valid world
    price = 140 / 180
    assert w.new_vehicle_price.median == pytest.approx(world.new_vehicle_price.median * price, 0.01)
    repair_rel = (190 / 400) / price  # repairs relatively cheaper than cars in 2002
    m0, _ = world.damage_fraction["collision_vehicle"]
    assert w.damage_fraction["collision_vehicle"][0] == pytest.approx(m0 * repair_rel, 0.01)
    assert w.damage_fraction["theft"] == world.damage_fraction["theft"]  # theft = whole car
    assert w.use_at_loss["rideshare_active"] == 0 and w.use_at_loss["delivery_active"] == 0
    assert sum(w.use_at_loss.values()) == pytest.approx(1)
    assert w.rideshare_endorsement_rate == 0 and w.adas_rate_by_age["new"] == 0
    assert w.traps["business_use_no_endorsement"] == 0  # no rideshare apps yet
    assert str(w.loss_date_start) == "2002-01-01"


def test_era_overrides_base_year_is_identity_for_prices(world: World) -> None:
    w = with_overrides(world, eras.era_overrides(world, 2024, CPI))
    assert w.new_vehicle_price.median == pytest.approx(world.new_vehicle_price.median)
    assert w.use_at_loss == pytest.approx(world.use_at_loss)
    assert w.traps == world.traps
