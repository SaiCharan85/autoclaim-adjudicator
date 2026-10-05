import numpy as np
import pytest

from autoclaim.lines.auto import ges


@pytest.mark.parametrize(
    ("label", "cause"),
    [("Motor Vehicle In-Transport", "collision_vehicle"), ("Parked MV", "collision_vehicle"),
     ("Parked Motor Vehicle", "collision_vehicle"), ("Live Animal", "animal"), ("Deer", "animal"),
     ("Ridden Animal or Animal Drawn Conveyance", None), ("Fire/Explosion", "fire"),
     ("Fire Hydrant", "collision_object"), ("Tree-standing only", "collision_object"),
     ("Guardrail Face", "collision_object"), ("Pedestrian", None), ("Rollover/Overturn", None)],
)  # fmt: skip
def test_event_cause(label, cause) -> None:
    assert ges.event_cause(label) == cause


def test_damage_towed_light_area() -> None:
    assert ges.damage_extent("Disabling (Severe)") == "disabling"
    assert ges.damage_extent("Functional Damage") == "functional"
    assert ges.damage_extent("No Damage") == "none" and ges.damage_extent("Unknown") == "unknown"
    assert ges.towed_flag("Driven Away") == 0.0 and ges.towed_flag("Towed Due To Damage") == 1.0
    assert ges.towed_flag("Towed Not Due To Disabling Damage") == 1.0  # towed for any reason
    assert np.isnan(ges.towed_flag("Unknown If Towed"))
    assert ges.light_group("Dark But Lighted") == "dark" and ges.light_group("Dusk") == "dawn_dusk"
    assert ges.area_group("Within area of population 100,000+") == "urban"
    assert ges.area_group("Other area") == "rural" and np.isnan(ges.area_group("Unknown"))


def test_borrowed_labels_keep_formats_with_same_codes(monkeypatch) -> None:
    y05 = {"A": {1.0: "x"}, "B": {1.0: "old"}, "C": {1.0: "c"}}
    y07 = {"A": {1.0: "x"}, "B": {1.0: "new wording"}, "C": {1.0: "c", 2.0: "added"}}
    years = {2005: y05, 2007: y07}
    monkeypatch.setattr(ges, "year_labels", lambda y, root=None: years[y])
    assert ges.borrowed_labels(2006) == {"A": {1.0: "x"}, "B": {1.0: "old"}}  # C: codes differ
