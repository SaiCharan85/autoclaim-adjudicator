"""NHTSA GES 2002-2015 (CRSS's predecessor) -> the same real-incident rows as crss.load_year.

GES codes changed over the years (e.g. the harmful-event scheme in 2010), so fields are mapped by
their official LABEL text, read from each release's format catalog. Differences from CRSS:
- no urban/rural flag: places of 25,000+ people (LAND_USE) count as urban, "other area" as rural;
- damage severity is VEH_SEV before 2010 and DEFORMED after (same four labels);
- struck parked vehicles have their own file only from 2005 (earlier: simulated, like any cause
  police data lacks);
- 2006 ships no labels: borrowed from the nearest year whose format of the same name agrees.
"""

import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from autoclaim.datasets.sas_labels import Labels, column_formats, read_labels
from autoclaim.datasets.sources import GES
from autoclaim.lines.auto.crss import (
    BODY_CLASS,
    REGION,
    assign_day,
    normalize_make,
    weather_group,
)
from autoclaim.paths import raw_dir

GES_YEARS = tuple(sorted(GES))
GES_BODY_CLASS = BODY_CLASS | dict.fromkeys((30, 31, 32), "pickup")  # compact/standard pickups

_VEHICLE = r"motor vehicle in[- ]?transport|\bmvit\b|in motion outside|parked m(otor )?v|working m"
_FIXED = (r"building|attenuator|bridge|guardrail|barrier|sign support|signal support|pole|post|"
          r"culvert|ditch|snow bank|mail ?box|curb|embankment|fence|wall|hydrant|shrub|tree|"
          r"boulder|fixed object|object not fixed")  # fmt: skip
NON_FAULT = (
    r"^none|no violation|license|registration|insurance|restraint|unknown|not reported|no driver"
)


# 2011-2015 GES uses CRSS-style names; map them onto the older GES names used below.
ALIASES = {"VEH_NO": "VEHNO", "MOD_YEAR": "MODEL_YR", "M_HARM": "V_EVENT", "VE_TOTAL": "VEH_INVL"}
ALIASES_PARKED = {"VEH_NO": "PVEHNO", "PMODYEAR": "PMODELYR"}


def event_cause(label: str) -> str | None:
    """Most harmful event label -> cause of loss (None = out of scope: pedestrian, rollover...)."""
    t = label.lower()
    if "ridden animal" in t or "animal drawn" in t:
        return None
    if re.search(_VEHICLE, t):
        return "collision_vehicle"
    if "animal" in t or "deer" in t:
        return "animal"
    if "fire" in t and "hydrant" not in t:
        return "fire"
    if re.search(_FIXED, t):
        return "collision_object"
    return None


def damage_extent(label: str) -> str:
    t = label.lower()
    # before 2010 the labels say Moderate / Severe for Functional / Disabling (2003 shows both)
    for key, out in (("disabling", "disabling"), ("severe", "disabling"),
                     ("functional", "functional"), ("moderate", "functional"),
                     ("minor", "minor"), ("no damage", "none"), ("none", "none")):  # fmt: skip
        if key in t:
            return out
    return "unknown"


def towed_flag(label: str) -> float:
    """Towed for any reason = 1 (CRSS 2022+ meaning); unknown -> NaN."""
    t = label.lower()
    if "unknown" in t or "not reported" in t:
        return np.nan
    if "driven away" in t or "not towed" in t:
        return 0.0
    return 1.0 if "towed" in t or "abandoned" in t else np.nan


def light_group(label: str) -> str:
    t = label.lower()
    if "daylight" in t:
        return "daylight"
    if "dawn" in t or "dusk" in t:
        return "dawn_dusk"
    return "dark" if "dark" in t else "unknown"


def area_group(label: str) -> float | str:
    t = label.lower()
    if "population" in t:
        return "urban"
    return "rural" if "other area" in t else np.nan


def _weather_label(label: str) -> str:
    return "Clear" if "no adverse" in label.lower() else label  # pre-2010 wording for clear


@cache
def year_labels(year: int, root: Path | None = None) -> Labels:
    labels = read_labels(root or raw_dir(f"ges_{year}"))
    if labels or root is not None:
        return labels
    # 2006: no label file; borrow from the nearest years where both neighbours agree (checked
    # per format by `borrowed_labels`)
    return borrowed_labels(year)


def borrowed_labels(year: int) -> Labels:
    """The earlier year's labels for every format whose CODES both neighbours share (wording may
    differ slightly, e.g. 'Towed Due To Damage' vs '... Disabling Damage'; meaning does not)."""
    before, after = year_labels(year - 1), year_labels(year + 1)
    return {k: v for k, v in before.items() if k in after and set(after[k]) == set(v)}


def _labelled(path: Path, cols: Sequence[str], labels: Labels) -> pd.DataFrame:
    """Read `cols` (those present) and add `<col>_L` label columns for every formatted one."""
    have = column_formats(path)
    df = pd.read_sas(path, encoding="latin-1")
    alias = ALIASES_PARKED if "parkwork" in path.name else ALIASES
    renames = {a: b for a, b in alias.items() if a in df.columns and b not in df.columns}
    df = df.rename(columns=renames)
    have = {renames.get(c, c): f for c, f in have.items()}
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        fmt = have.get(c)
        if fmt and fmt in labels:
            df[f"{c}_L"] = df[c].map(labels[fmt])
    return df


def load_year(year: int, root: Path | None = None, seed: int = 0) -> pd.DataFrame:
    base = root if root is not None else raw_dir(f"ges_{year}")
    labels = year_labels(year, root)
    acc_cols = ["CASENUM", "WEIGHT", "REGION", "LAND_USE", "VEH_INVL", "YEAR", "MONTH",
                "WEEKDAY", "DAY_WEEK", "HOUR", "LGHT_CON", "LGT_COND", "WEATHER"]  # fmt: skip
    acc = _labelled(base / "accident.sas7bdat", acc_cols, labels)
    veh = _labelled(base / "vehicle.sas7bdat",
                    ["CASENUM", "VEHNO", "HIT_RUN", "MODEL_YR", "MAKE", "BODY_TYP", "V_EVENT",
                     "VEH_SEV", "DEFORMED", "TOWED", "NUM_INJV"], labels)  # fmt: skip
    viol = _labelled(base / "violatn.sas7bdat", ["CASENUM", "VEHNO", "MVIOLATN"], labels)
    park_file = next((base / f for f in ("parkwork.sas7bdat", "parked.sas7bdat")
                      if (base / f).exists()), None)  # fmt: skip

    fled = set(veh.loc[veh["HIT_RUN"] == 1, "CASENUM"])
    vlab = viol.get("MVIOLATN_L", pd.Series("", index=viol.index)).fillna("").astype(str)
    charged = viol.loc[~vlab.str.lower().str.contains(NON_FAULT, regex=True) & (vlab != ""),
                       ["CASENUM", "VEHNO"]].drop_duplicates()  # fmt: skip
    charged_keys = set(zip(charged["CASENUM"], charged["VEHNO"], strict=True))

    sev = "DEFORMED_L" if "DEFORMED_L" in veh else "VEH_SEV_L"
    moving = veh[veh["HIT_RUN"] != 1].copy()
    moving["cause"] = moving["V_EVENT_L"].fillna("").map(event_cause)
    moving = moving[moving["cause"].notna()]
    moving = moving.assign(role="in_transport", damage=moving[sev], towed=moving["TOWED_L"],
                           injuries=moving["NUM_INJV"], make=moving.get("MAKE_L"))  # fmt: skip
    cols = ["CASENUM", "VEHNO", "cause", "role", "MODEL_YR", "make", "BODY_TYP", "damage",
            "towed", "injuries"]  # fmt: skip
    parts = [moving[cols]]
    if park_file is not None:
        park = _labelled(park_file, ["CASENUM", "PVEHNO", "PMODELYR", "PMAKE", "PBODYTYP",
                                     "PVEH_SEV", "PTOWED", "PHARM_EV"], labels)  # fmt: skip
        if "PHARM_EV_L" in park:  # 2011+: keep parked cars struck by a vehicle (as CRSS)
            hit = park["PHARM_EV_L"].fillna("").map(event_cause) == "collision_vehicle"
            park = park[hit]
        parts.append(pd.DataFrame({
            "CASENUM": park["CASENUM"], "VEHNO": park["PVEHNO"], "cause": "parked_hit",
            "role": "parked", "MODEL_YR": park["PMODELYR"], "make": park.get("PMAKE_L"),
            "BODY_TYP": park["PBODYTYP"], "damage": park.get("PVEH_SEV_L"),
            "towed": park.get("PTOWED_L"), "injuries": 0.0,
        }))  # fmt: skip
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[rows["BODY_TYP"].isin(GES_BODY_CLASS)].merge(acc, on="CASENUM", how="inner")
    rows.loc[rows["CASENUM"].isin(fled) & rows["cause"].isin(["collision_vehicle", "parked_hit"]),
             "cause"] = "hit_and_run"  # fmt: skip

    key_fault = np.array([(c, v) in charged_keys for c, v in zip(rows["CASENUM"], rows["VEHNO"],
                                                                   strict=True)])  # fmt: skip
    single = rows["VEH_INVL"] == 1
    at_fault = np.select(
        [rows["cause"].isin(["animal", "fire", "hit_and_run", "parked_hit"]),
         (rows["cause"] == "collision_object") & single],
        [0.0, 1.0], default=key_fault.astype(float),
    )  # fmt: skip
    rng = np.random.default_rng(seed + year)
    dow = rows["DAY_WEEK"] if "DAY_WEEK" in rows else rows["WEEKDAY"]
    light = rows["LGT_COND_L"] if "LGT_COND_L" in rows else rows["LGHT_CON_L"]
    model_year = rows["MODEL_YR"].where(rows["MODEL_YR"] < 9000)
    return pd.DataFrame({
        "source_record": f"ges:{year}:" + rows["CASENUM"].astype(int).astype(str) + ":"
                         + rows["VEHNO"].astype(int).astype(str),
        "weight": rows["WEIGHT"].astype(float),
        "loss_date": assign_day(rows["YEAR"].astype(int), rows["MONTH"].astype(int),
                                dow.astype(int), rng),
        "loss_hour": rows["HOUR"].where(rows["HOUR"].between(0, 23)),
        "region": rows["REGION"].map(REGION),
        "area": rows.get("LAND_USE_L", pd.Series("", index=rows.index)).fillna("").map(area_group),
        "cause": rows["cause"],
        "vehicle_role": rows["role"],
        "n_vehicles": rows["VEH_INVL"].astype(int),
        "injury_count": rows["injuries"].where(rows["injuries"] < 90),
        "damage_extent": rows["damage"].fillna("").map(damage_extent),
        "towed": rows["towed"].fillna("").map(towed_flag),
        "light": light.fillna("").map(light_group),
        "weather": weather_group(rows["WEATHER_L"].fillna("Unknown").map(_weather_label)),
        "vehicle_make": normalize_make(rows["make"].astype("string").str.title()),
        "vehicle_model": pd.Series(pd.NA, index=rows.index, dtype="string"),
        "vehicle_model_year": model_year,
        "body_class": rows["BODY_TYP"].map(GES_BODY_CLASS),
        "at_fault": at_fault,
    }).reset_index(drop=True)  # fmt: skip
