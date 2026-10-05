"""Turn NHTSA CRSS police-reported crash records into real incident rows for auto claims.

One row per *personal passenger vehicle that could file an own-damage claim*:
- vehicles in transport (except a vehicle that fled: it is the hit-and-run *cause*, not a
  claimant)
- parked vehicles that were struck (parkwork.csv)

Every field comes from the real record except the exact day of month: CRSS publishes month and
weekday only, so a day matching both is drawn at random. Codes follow the CRSS Analytical
User's Manual; the value lists were checked against the 2022-2024 files.
"""

from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from autoclaim.datasets.sources import CRSS_YEARS
from autoclaim.paths import raw_dir

# Personal passenger vehicles (BODY_TYP): cars, SUVs, light pickups and vans.
BODY_CLASS: dict[int, str] = {
    **dict.fromkeys((1, 2, 3, 4, 5, 6, 7, 8, 9, 17), "car"),
    **dict.fromkeys((14, 15, 16, 19), "suv"),
    **dict.fromkeys((10, 33, 34, 39, 48), "pickup"),
    **dict.fromkeys((20, 21, 22, 28, 29), "van"),
    49: "light_unknown",
}

# Most harmful event (M_HARM / PHARM_EV) -> cause of loss. Unlisted codes are out of scope
# (pedestrian/cyclist crashes, immersion, falling objects, unknown events, jackknife).
VEHICLE_EVENTS = (12, 14, 45, 54)
FIXED_OBJECTS = (1, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 38, 39, 40,
                 41, 42, 43, 46, 50, 52, 53, 57, 58, 59, 91, 93)  # fmt: skip
EVENT_CAUSE: dict[int, str] = {
    **dict.fromkeys(VEHICLE_EVENTS, "collision_vehicle"),
    **dict.fromkeys(FIXED_OBJECTS, "collision_object"),
    11: "animal",
    2: "fire",
}
PARKED_EVENTS = (12, 14, 45)

DAMAGE_EXTENT = {0: "none", 2: "minor", 4: "functional", 6: "disabling"}  # 7/8/9 -> unknown
LIGHT = {1: "daylight", 2: "dark", 3: "dark", 6: "dark", 4: "dawn_dusk", 5: "dawn_dusk"}
# 2022+: 5 not towed, 6 towed. 2016-2021 split towing by reason (2, 3, 7): all mean towed.
TOWED = {5: 0.0, 6: 1.0, 2: 1.0, 3: 1.0, 7: 1.0}
REGION = {1: "northeast", 2: "midwest", 3: "south", 4: "west"}
URBANICITY = {1: "urban", 2: "rural"}

# Violations that say nothing about who caused the crash (paperwork, unknown, no driver).
NON_FAULT_VIOLATIONS = frozenset({0, 71, 72, 74, 75, 76, 83, 95, 99})
OLD_NOT_REPORTED = 97  # MVIOLATN (2016-2019 CRSS, GES): 'Not Reported'

MAKE_FIXES = {"Nissan/Datsun": "Nissan", "KIA": "Kia", "Mercedes-Benz": "Mercedes-Benz"}

ACCIDENT_COLS = ["CASENUM", "WEIGHT", "REGION", "URBANICITY", "VE_TOTAL", "YEAR", "MONTH",
                 "DAY_WEEK", "HOUR", "LGT_COND", "WEATHERNAME", "MAN_COLL"]  # fmt: skip
VEHICLE_COLS = ["CASENUM", "VEH_NO", "HIT_RUN", "MOD_YEAR", "MAKENAME", "VPICMODELNAME",
                "BODY_TYP", "M_HARM", "DEFORMED", "TOWED", "NUM_INJV"]  # fmt: skip
PARKED_COLS = ["CASENUM", "VEH_NO", "PHARM_EV", "PMODYEAR", "PMAKENAME", "PVPICMODELNAME",
               "PBODYTYP", "PVEH_SEV", "PTOWED"]  # fmt: skip


def _read(path: Path, cols: Sequence[str]) -> pd.DataFrame:
    """Read the wanted columns that exist (older years lack some *NAME columns; see below)."""
    have = set(pd.read_csv(path, encoding="latin-1", nrows=0).columns)
    return pd.read_csv(path, encoding="latin-1", usecols=[c for c in cols if c in have],
                       low_memory=False)  # fmt: skip


# Before 2019 the files carry only NHTSA codes for make, make-model and weather. The codes are one
# national scheme across years, so names are filled from years that publish both (each code maps
# to exactly one name in 2019-2024: checked when the maps are built).
NAMED_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)


# Make-model wording changed across years (e.g. spacing, abbreviations); the newest wins.
LATEST_WORDING_WINS = frozenset({"MAK_MODNAME", "PMAK_MODNAME"})


@cache
def code_names(code_col: str, name_col: str, file: str) -> dict[int, str]:
    pairs = []
    for year in NAMED_YEARS:  # oldest first, so a later year's wording overwrites
        path = raw_dir(f"crss_{year}") / file
        if not path.exists():
            continue
        have = set(pd.read_csv(path, encoding="latin-1", nrows=0).columns)
        if {code_col, name_col} <= have:
            pairs.append(pd.read_csv(path, usecols=[code_col, name_col], encoding="latin-1"))
    if not pairs:
        raise FileNotFoundError(f"no CRSS year with {name_col} to build the code map from")
    df = pd.concat(pairs).dropna().drop_duplicates()
    if name_col in LATEST_WORDING_WINS:
        df = df.drop_duplicates(subset=code_col, keep="last")
    elif df[code_col].duplicated().any():
        raise ValueError(f"{code_col} -> {name_col} is not one-to-one across {NAMED_YEARS}")
    return dict(zip(df[code_col].astype(int), df[name_col].astype(str), strict=True))


def clean_model_name(names: pd.Series) -> pd.Series:
    """'Honda Accord (Note: For Crosstour ...)' -> 'Honda Accord' (drop NHTSA editorial notes)."""
    return names.astype("string").str.replace(r"\s*\(.*$", "", regex=True).str.strip()


def _fill_names(df: pd.DataFrame, pairs: Sequence[tuple[str, str, str]]) -> pd.DataFrame:
    for code_col, name_col, file in pairs:
        if name_col not in df and code_col in df:
            df[name_col] = df[code_col].map(code_names(code_col, name_col, file))
    return df


def normalize_make(names: pd.Series) -> pd.Series:
    """'Nissan/Datsun' -> 'Nissan', 'Jeep / Kaiser-Jeep / ...' -> 'Jeep', unknown -> NaN."""
    fixed = names.replace(MAKE_FIXES)
    first = fixed.astype("string").str.split("/").str[0].str.strip()
    return first.where(~first.str.contains("Unknown|Not Reported", case=False, na=True))


def weather_group(names: pd.Series) -> pd.Series:
    n = names.astype("string").str.lower()
    out = pd.Series("other", index=names.index, dtype="object")
    out[n.str.contains("clear", na=False)] = "clear"
    out[n.str.contains("cloud", na=False)] = "cloudy"
    out[n.str.contains("rain|drizzle", na=False)] = "rain"
    out[n.str.contains("snow|sleet|hail", na=False)] = "snow_ice"
    out[n.str.contains("fog", na=False)] = "fog"
    out[n.str.contains("unknown|not reported", na=True)] = "unknown"
    return out


def assign_day(
    year: pd.Series, month: pd.Series, dow: pd.Series, rng: np.random.Generator
) -> pd.Series:
    """A date in (year, month) on CRSS weekday `dow` (1=Sunday..7=Saturday), chosen at random.

    Unknown weekday (9) -> any day of the month.
    """
    first = pd.to_datetime(dict(year=year, month=month, day=1))
    days_in_month = first.dt.days_in_month.to_numpy()
    # pandas weekday: Monday=0..Sunday=6; CRSS: Sunday=1..Saturday=7
    target = ((dow.to_numpy() + 5) % 7).astype(int)
    first_wd = first.dt.weekday.to_numpy()
    offset0 = (target - first_wd) % 7  # first day of month with that weekday (0-based)
    n_matches = (days_in_month - 1 - offset0) // 7 + 1
    pick = offset0 + 7 * (rng.random(len(year)) * n_matches).astype(int)
    unknown = (dow.to_numpy() < 1) | (dow.to_numpy() > 7)
    pick = np.where(unknown, (rng.random(len(year)) * days_in_month).astype(int), pick)
    return first + pd.to_timedelta(pick, unit="D")


def _at_fault(violations: pd.DataFrame) -> pd.Series:
    moving = violations[~violations["VIOLATION"].isin(NON_FAULT_VIOLATIONS)]
    keys = moving[["CASENUM", "VEH_NO"]].drop_duplicates()
    return pd.Series(1.0, index=pd.MultiIndex.from_frame(keys))


def load_year(year: int, root: Path | None = None, seed: int = 0) -> pd.DataFrame:
    """Real claimant incidents for one CRSS year (vehicles in transport + struck parked)."""
    base = root if root is not None else raw_dir(f"crss_{year}")
    acc = _fill_names(
        _read(base / "accident.csv", [*ACCIDENT_COLS, "WEATHER"]),
        [("WEATHER", "WEATHERNAME", "accident.csv")],
    )
    veh = _fill_names(
        _read(base / "vehicle.csv", [*VEHICLE_COLS, "MAKE", "MAK_MOD"]),
        [("MAKE", "MAKENAME", "vehicle.csv"), ("MAK_MOD", "MAK_MODNAME", "vehicle.csv")],
    )
    park = _fill_names(
        _read(base / "parkwork.csv", [*PARKED_COLS, "PMAKE", "PMAK_MOD"]),
        [("PMAKE", "PMAKENAME", "parkwork.csv"), ("PMAK_MOD", "PMAK_MODNAME", "parkwork.csv")],
    )
    # vPIC model names start in 2020; earlier years use NHTSA's make-model names
    if "VPICMODELNAME" not in veh:
        veh["VPICMODELNAME"] = clean_model_name(veh["MAK_MODNAME"])
    if "PVPICMODELNAME" not in park:
        park["PVPICMODELNAME"] = clean_model_name(park["PMAK_MODNAME"])
    viol = _read(base / "violatn.csv", ["CASENUM", "VEH_NO", "VIOLATION", "MVIOLATN"])
    if "VIOLATION" not in viol:  # before 2020: same codes as MVIOLATN, plus 97 = not reported
        viol = viol.rename(columns={"MVIOLATN": "VIOLATION"})
        viol = viol[viol["VIOLATION"] != OLD_NOT_REPORTED]

    fled_cases = set(veh.loc[veh["HIT_RUN"] == 1, "CASENUM"])
    fault = _at_fault(viol)

    moving = veh[(veh["HIT_RUN"] != 1) & veh["M_HARM"].isin(EVENT_CAUSE)].copy()
    moving["cause"] = moving["M_HARM"].map(EVENT_CAUSE)
    moving["role"] = "in_transport"
    moving = moving.rename(
        columns={"DEFORMED": "damage_code", "TOWED": "towed_code", "NUM_INJV": "injuries"}
    )

    parked = park[park["PHARM_EV"].isin(PARKED_EVENTS)].copy()
    parked["cause"] = "parked_hit"
    parked["role"] = "parked"
    parked["injuries"] = 0
    parked = parked.rename(
        columns={
            "PMODYEAR": "MOD_YEAR",
            "PMAKENAME": "MAKENAME",
            "PVPICMODELNAME": "VPICMODELNAME",
            "PBODYTYP": "BODY_TYP",
            "PVEH_SEV": "damage_code",
            "PTOWED": "towed_code",
        }
    )

    cols = ["CASENUM", "VEH_NO", "cause", "role", "MOD_YEAR", "MAKENAME", "VPICMODELNAME",
            "BODY_TYP", "damage_code", "towed_code", "injuries"]  # fmt: skip
    rows = pd.concat([moving[cols], parked[cols]], ignore_index=True)
    rows = rows[rows["BODY_TYP"].isin(BODY_CLASS)].merge(acc, on="CASENUM", how="inner")

    fled = rows["CASENUM"].isin(fled_cases) & rows["cause"].isin(
        ["collision_vehicle", "parked_hit"]
    )
    rows.loc[fled, "cause"] = "hit_and_run"

    key = pd.MultiIndex.from_frame(rows[["CASENUM", "VEH_NO"]])
    charged = pd.Series(fault.reindex(key).fillna(0.0).to_numpy(), index=rows.index)
    single = rows["VE_TOTAL"] == 1
    at_fault = np.select(
        [
            rows["cause"].isin(["animal", "fire", "hit_and_run", "parked_hit"]),
            (rows["cause"] == "collision_object") & single,
        ],
        [0.0, 1.0],
        default=charged,
    )

    rng = np.random.default_rng(seed + year)
    hour = rows["HOUR"].where(rows["HOUR"].between(0, 23))
    model_year = rows["MOD_YEAR"].where(rows["MOD_YEAR"] < 9000)
    injuries = rows["injuries"].where(rows["injuries"] < 90)
    return pd.DataFrame(
        {
            "source_record": "crss:"
            + str(year)
            + ":"
            + rows["CASENUM"].astype(str)
            + ":"
            + rows["VEH_NO"].astype(str),
            "weight": rows["WEIGHT"].astype(float),
            "loss_date": assign_day(rows["YEAR"], rows["MONTH"], rows["DAY_WEEK"], rng),
            "loss_hour": hour,
            "region": rows["REGION"].map(REGION),
            "area": rows["URBANICITY"].map(URBANICITY),
            "cause": rows["cause"],
            "vehicle_role": rows["role"],
            "n_vehicles": rows["VE_TOTAL"].astype(int),
            "injury_count": injuries,
            "damage_extent": rows["damage_code"].map(DAMAGE_EXTENT).fillna("unknown"),
            "towed": rows["towed_code"].map(TOWED),
            "light": rows["LGT_COND"].map(LIGHT).fillna("unknown"),
            "weather": weather_group(rows["WEATHERNAME"]),
            "vehicle_make": normalize_make(rows["MAKENAME"]),
            "vehicle_model": rows["VPICMODELNAME"].where(
                ~rows["VPICMODELNAME"]
                .astype("string")
                .str.contains("Unknown|Not Reported|Not Applicable", case=False, na=True)
            ),
            "vehicle_model_year": model_year,
            "body_class": rows["BODY_TYP"].map(BODY_CLASS),
            "at_fault": at_fault,
        }
    ).reset_index(drop=True)


def load_incidents(years: Sequence[int] = CRSS_YEARS, seed: int = 0) -> pd.DataFrame:
    """All real incidents across CRSS years, newest data last."""
    return pd.concat([load_year(y, seed=seed) for y in years], ignore_index=True)
