"""CRSS loader on tiny hand-written fixture files (no real data needed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoclaim.lines.auto import crss


def _write(dir_: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(dir_ / name, index=False, encoding="latin-1")


def _acc(case: int, **kw: object) -> dict:
    base = {"CASENUM": case, "WEIGHT": 100.0, "REGION": 3, "URBANICITY": 1, "VE_TOTAL": 2,
            "YEAR": 2024, "MONTH": 11, "DAY_WEEK": 2, "HOUR": 17, "LGT_COND": 1,
            "WEATHERNAME": "Clear", "MAN_COLL": 1}  # fmt: skip
    return base | kw


def _veh(case: int, no: int, **kw: object) -> dict:
    base = {"CASENUM": case, "VEH_NO": no, "HIT_RUN": 0, "MOD_YEAR": 2019, "MAKENAME": "Toyota",
            "VPICMODELNAME": "Camry", "BODY_TYP": 4, "M_HARM": 12, "DEFORMED": 4, "TOWED": 5,
            "NUM_INJV": 0}  # fmt: skip
    return base | kw


@pytest.fixture
def crss_dir(tmp_path: Path) -> Path:
    accidents = [
        _acc(1),  # two-vehicle crash; vehicle 1 charged with following too closely
        _acc(2, VE_TOTAL=1, HOUR=99),  # single vehicle hits a deer; unknown hour
        _acc(3),  # hit-and-run: vehicle 2 fled
        _acc(4, VE_TOTAL=1),  # single vehicle hits a tree
        _acc(5, VE_TOTAL=1),  # parked car struck; striker fled
        _acc(6, VE_TOTAL=1),  # motorcycle + pedestrian: out of scope
    ]
    vehicles = [
        _veh(1, 1),
        _veh(1, 2, MAKENAME="Nissan/Datsun", BODY_TYP=14),
        _veh(2, 1, M_HARM=11, DEFORMED=6, TOWED=6, MOD_YEAR=9999),
        _veh(3, 1),
        _veh(3, 2, HIT_RUN=1),
        _veh(4, 1, M_HARM=42, NUM_INJV=1, MAKENAME="KIA", VPICMODELNAME="Unknown"),
        _veh(5, 1, HIT_RUN=1),
        _veh(6, 1, BODY_TYP=80, M_HARM=8),
    ]
    parked = [{"CASENUM": 5, "VEH_NO": 1, "PHARM_EV": 14, "PMODYEAR": 2015, "PMAKENAME": "Honda",
               "PVPICMODELNAME": "Civic", "PBODYTYP": 4, "PVEH_SEV": 2, "PTOWED": 5}]  # fmt: skip
    violations = [
        {"CASENUM": 1, "VEH_NO": 1, "VIOLATION": 58},  # following too closely -> at fault
        {"CASENUM": 1, "VEH_NO": 2, "VIOLATION": 76},  # uninsured: paperwork, not fault
        {"CASENUM": 3, "VEH_NO": 2, "VIOLATION": 7},
    ]
    _write(tmp_path, "accident.csv", accidents)
    _write(tmp_path, "vehicle.csv", vehicles)
    _write(tmp_path, "parkwork.csv", parked)
    _write(tmp_path, "violatn.csv", violations)
    return tmp_path


@pytest.fixture
def incidents(crss_dir: Path) -> pd.DataFrame:
    return crss.load_year(2024, root=crss_dir).set_index("source_record")


def test_claimant_selection(incidents: pd.DataFrame) -> None:
    assert sorted(incidents.index) == [
        "crss:2024:1:1",
        "crss:2024:1:2",
        "crss:2024:2:1",
        "crss:2024:3:1",  # victim of the hit-and-run; the fleeing vehicle 3:2 is not a claimant
        "crss:2024:4:1",
        "crss:2024:5:1",  # struck parked car (parkwork)
    ]  # motorcycle/pedestrian crash 6 excluded


def test_cause_mapping(incidents: pd.DataFrame) -> None:
    assert incidents.loc["crss:2024:1:1", "cause"] == "collision_vehicle"
    assert incidents.loc["crss:2024:2:1", "cause"] == "animal"
    assert incidents.loc["crss:2024:3:1", "cause"] == "hit_and_run"
    assert incidents.loc["crss:2024:4:1", "cause"] == "collision_object"
    assert incidents.loc["crss:2024:5:1", "cause"] == "hit_and_run"  # parked + striker fled
    assert incidents.loc["crss:2024:5:1", "vehicle_role"] == "parked"


def test_at_fault_from_moving_violations_only(incidents: pd.DataFrame) -> None:
    assert incidents.loc["crss:2024:1:1", "at_fault"] == 1.0
    assert incidents.loc["crss:2024:1:2", "at_fault"] == 0.0  # uninsured is not a fault violation
    assert incidents.loc["crss:2024:4:1", "at_fault"] == 1.0  # single vehicle into a tree
    assert incidents.loc["crss:2024:2:1", "at_fault"] == 0.0  # animal
    assert incidents.loc["crss:2024:3:1", "at_fault"] == 0.0  # hit-and-run victim


def test_unknown_codes_become_missing(incidents: pd.DataFrame) -> None:
    deer = incidents.loc["crss:2024:2:1"]
    assert np.isnan(deer["loss_hour"])
    assert np.isnan(deer["vehicle_model_year"])
    assert deer["damage_extent"] == "disabling"
    assert deer["towed"] == 1.0
    assert pd.isna(incidents.loc["crss:2024:4:1", "vehicle_model"])


def test_make_and_body_normalization(incidents: pd.DataFrame) -> None:
    assert incidents.loc["crss:2024:1:2", "vehicle_make"] == "Nissan"
    assert incidents.loc["crss:2024:1:2", "body_class"] == "suv"
    assert incidents.loc["crss:2024:4:1", "vehicle_make"] == "Kia"


def test_region_area_weather_light(incidents: pd.DataFrame) -> None:
    row = incidents.loc["crss:2024:1:1"]
    assert (row["region"], row["area"], row["weather"], row["light"]) == (
        "south",
        "urban",
        "clear",
        "daylight",
    )


def test_assigned_date_keeps_real_month_and_weekday(incidents: pd.DataFrame) -> None:
    dates = pd.to_datetime(incidents["loss_date"])
    assert (dates.dt.year == 2024).all()
    assert (dates.dt.month == 11).all()
    assert (dates.dt.day_name() == "Monday").all()  # CRSS DAY_WEEK 2 = Monday


def test_assign_day_unknown_weekday_stays_in_month() -> None:
    rng = np.random.default_rng(0)
    n = 200
    out = crss.assign_day(pd.Series([2024] * n), pd.Series([2] * n), pd.Series([9] * n), rng)
    assert (out.dt.month == 2).all()
    assert out.dt.day.max() <= 29


@pytest.mark.parametrize(("dow", "name"), [(1, "Sunday"), (4, "Wednesday"), (7, "Saturday")])
def test_assign_day_every_weekday(dow: int, name: str) -> None:
    rng = np.random.default_rng(1)
    out = crss.assign_day(pd.Series([2023] * 50), pd.Series([3] * 50), pd.Series([dow] * 50), rng)
    assert (out.dt.day_name() == name).all()
    assert out.dt.day.nunique() > 1  # spread across the matching days


def test_seeded_day_assignment_is_reproducible(crss_dir: Path) -> None:
    a = crss.load_year(2024, root=crss_dir, seed=3)
    b = crss.load_year(2024, root=crss_dir, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_weather_group() -> None:
    names = pd.Series(
        ["Clear", "Rain", "Blowing Snow", "Fog, Smog, Smoke", "Not Reported", "Other"]
    )
    assert crss.weather_group(names).tolist() == [
        "clear",
        "rain",
        "snow_ice",
        "fog",
        "unknown",
        "other",
    ]


def test_normalize_make_unknown() -> None:
    out = crss.normalize_make(pd.Series(["Unknown Make", "Jeep / Kaiser-Jeep / Willys- Jeep"]))
    assert pd.isna(out.iat[0])
    assert out.iat[1] == "Jeep"
