"""The two auto fraud datasets as specs: the grounded-hybrid simulator (production) and the real
1994-96 Kaggle labels (reality check). Both map onto one red-flag rule vocabulary, so the same
rules can be checked against real labels.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from autoclaim.config import DatasetSplit, FraudModelConfig
from autoclaim.datasets import vehicle_fraud
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.lines.auto.simulator.columns import assert_fnol_only, assert_stage_features
from autoclaim.ml import features as legacy
from autoclaim.ml.frame import AMOUNT, T, Y
from autoclaim.ml.spec import DatasetSpec, Frames
from autoclaim.ml.stages import ExpectedAmountStage
from autoclaim.paths import data_dir

EPOCH = pd.Timestamp("1970-01-01")

# ---------------------------------------------------------------- simulator (production)

SIM_RAW_FEATURES = (
    "loss_hour", "region", "area", "weather", "light", "vehicle_make", "body_class", "cause",
    "n_vehicles", "injury_count", "damage_extent", "towed", "at_fault", "vehicle_role",
    "loss_state", "policy_state", "coverage", "collision_deductible", "comprehensive_deductible",
    "rideshare_endorsement", "driver_role", "driver_age", "use_at_loss", "address_change_days",
    "prior_claims_3y", "vehicle_age", "vehicle_acv", "financed", "adas", "police_report",
    "police_report_hours", "witness_count", "notice_days", "attorney_involved", "claimed_amount",
)  # fmt: skip
SIM_BOOLS = ("rideshare_endorsement", "financed", "adas", "police_report", "attorney_involved")


def _days(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (pd.to_datetime(later) - pd.to_datetime(earlier)).dt.days.astype("float64")


def sim_features(raw: pd.DataFrame) -> pd.DataFrame:
    """FNOL features. Missing input columns become NaN (partial runtime records are fine)."""
    out = pd.DataFrame(index=raw.index)
    for col in SIM_RAW_FEATURES:
        out[col] = raw.get(col, np.nan)
    for col in SIM_BOOLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")  # True -> 1.0
    if "address_change_days" in raw:  # in the data, NaN means "no move in the past 2 years"
        out["address_change_days"] = out["address_change_days"].fillna(NO_RECENT_MOVE_DAYS)
    has = set(raw.columns)
    nan = pd.Series(np.nan, index=raw.index)
    loss = pd.to_datetime(raw["loss_date"]) if "loss_date" in has else None
    start = raw["policy_start_date"] if "policy_start_date" in has else None
    report = raw["report_date"] if "report_date" in has else None
    out["days_policy_to_loss"] = (
        _days(loss, start) if loss is not None and start is not None else nan
    )
    out["days_policy_to_claim"] = (
        _days(report, start) if report is not None and start is not None else nan
    )
    out["loss_month"] = loss.dt.month.astype("float64") if loss is not None else nan
    out["loss_dow"] = loss.dt.dayofweek.astype("float64") if loss is not None else nan
    out["claim_to_acv"] = pd.to_numeric(out["claimed_amount"], errors="coerce") / pd.to_numeric(
        out["vehicle_acv"], errors="coerce"
    )
    out["out_of_state"] = (
        (raw["loss_state"] != raw["policy_state"]).astype("float64")
        if {"loss_state", "policy_state"} <= has
        else nan
    )
    return out


def sim_prepare(raw: pd.DataFrame) -> pd.DataFrame:
    frame = sim_features(raw)
    frame[Y] = raw["fraud_confirmed"].astype(int)
    frame[T] = (pd.to_datetime(raw["loss_date"]) - EPOCH).dt.days
    frame[AMOUNT] = raw["claimed_amount"].astype("float64")
    return frame


def split_by_date(split: DatasetSplit) -> Callable[[pd.DataFrame], Frames]:
    assert split.val_start is not None and split.test_start is not None
    val_t = (pd.Timestamp(split.val_start) - EPOCH).days
    test_t = (pd.Timestamp(split.test_start) - EPOCH).days

    def _split(frame: pd.DataFrame) -> Frames:
        return (
            frame[frame[T] < val_t],
            frame[(frame[T] >= val_t) & (frame[T] < test_t)],
            frame[frame[T] >= test_t],
        )

    return _split


# ---------------------------------------------------------------- stage 2: after the appraisal


def sim_appraisal_features(raw: pd.DataFrame) -> pd.DataFrame:
    """First-notice features + what the independent appraisal adds. The estimate audit signal is
    log(claimed / appraised): > 0 means the shop asks more than the appraiser found."""
    out = sim_features(raw)
    nan = pd.Series(np.nan, index=raw.index)
    appraised = pd.to_numeric(raw.get("appraised_amount", nan), errors="coerce")
    claimed = pd.to_numeric(out["claimed_amount"], errors="coerce")
    out["appraised_amount"] = appraised
    flag = raw.get("appraiser_prior_damage", nan)
    out["appraiser_prior_damage"] = pd.to_numeric(flag, errors="coerce").astype("float64")
    out["log_claim_to_appraisal"] = np.log(claimed / appraised)
    return out


def sim_appraisal_prepare(raw: pd.DataFrame) -> pd.DataFrame:
    frame = sim_appraisal_features(raw)
    frame[Y] = raw["fraud_confirmed"].astype(int)
    frame[T] = (pd.to_datetime(raw["loss_date"]) - EPOCH).dt.days
    frame[AMOUNT] = raw["claimed_amount"].astype("float64")
    return frame


def _check_appraisal_stage(features: Sequence[str]) -> None:
    assert_stage_features(features, "appraisal")


# What a repair usually costs depends on these first-notice facts (not on who is claiming).
REPAIR_COST_INPUTS = (
    "cause", "damage_extent", "towed", "vehicle_acv", "body_class", "vehicle_age",
    "n_vehicles", "injury_count", "adas", "vehicle_role",
)  # fmt: skip


def repair_cost_stage(seed: int) -> ExpectedAmountStage:
    return ExpectedAmountStage(REPAIR_COST_INPUTS, amount_col="claimed_amount", seed=seed)


def _sim_holdouts() -> dict[str, pd.DataFrame]:
    return {name: sim_build.load(name) for name in ("holdout_fresh", "holdout_shift")}


# ---------------------------------------------------------------- real 1994-96 (reality check)

DAYS_BAND = {"none": 0.0, "1 to 7": 4.0, "8 to 15": 11.0, "15 to 30": 22.0, "more than 30": 60.0}
# "No move" is a known fact, not missing. Encoded as a large number so it sorts as "long ago":
# a NaN would sort as the *smallest* value in CatBoost (nan_mode=Min), i.e. like a recent move.
NO_RECENT_MOVE_DAYS = 10_000.0
ADDRESS_BAND = {
    "no change": NO_RECENT_MOVE_DAYS,
    "under 6 months": 90.0,
    "1 year": 365.0,
    "2 to 3 years": 900.0,
    "4 to 8 years": 2000.0,
}


def legacy_features(raw: pd.DataFrame) -> pd.DataFrame:
    clean = legacy.clean(raw)
    return clean[legacy.feature_columns(clean)]


def legacy_prepare(raw: pd.DataFrame) -> pd.DataFrame:
    frame = legacy_features(raw)
    frame[Y] = raw[legacy.TARGET].astype(int)
    frame[T] = legacy.time_order(raw)
    return frame


def legacy_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """The real data's banded columns, translated into the shared rule vocabulary (approximate
    band midpoints). Fields the 1990s table lacks stay absent, so those rules are unchecked."""
    clean = legacy.clean(raw)
    return pd.DataFrame(
        {
            "days_policy_to_loss": raw["Days_Policy_Accident"].map(DAYS_BAND),
            "days_policy_to_claim": raw["Days_Policy_Claim"].map(DAYS_BAND),
            "address_change_days": raw["AddressChange_Claim"].map(ADDRESS_BAND),
            "at_fault": (raw["Fault"] == "Policy Holder").astype("float64"),
            "police_report": (raw["PoliceReportFiled"] == "Yes").astype("float64"),
            "witness_count": (raw["WitnessPresent"] == "Yes").astype("float64"),
            "area": raw["AccidentArea"].str.lower(),
            "notice_days": clean["claim_lag_weeks"] * 7,
        },
        index=raw.index,
    )


def split_by_years(split: DatasetSplit) -> Callable[[pd.DataFrame], Frames]:
    assert split.train_years and split.test_years and split.val_fraction
    train_years, test_years, fraction = split.train_years, split.test_years, split.val_fraction
    first_test = min(test_years) * 100

    def _split(frame: pd.DataFrame) -> Frames:
        train_all = frame[(frame[T] < first_test) & (frame[T] // 100).isin(train_years)]
        cutoff = np.quantile(train_all[T], 1 - fraction)
        return (
            train_all[train_all[T] <= cutoff],
            train_all[train_all[T] > cutoff],
            frame[(frame[T] // 100).isin(test_years)],
        )

    return _split


def _legacy_raw() -> pd.DataFrame:
    return vehicle_fraud.load_validated()[0]


# ---------------------------------------------------------------- registry


def get_spec(name: str, cfg: FraudModelConfig) -> DatasetSpec:
    split = cfg.datasets[name]
    if name == "sim_us":
        return DatasetSpec(
            name=name,
            description="Grounded hybrid: real NHTSA CRSS 2022-24 crashes + simulated policy/fraud",
            real_labels=False,
            load_raw=lambda: sim_build.load("claims"),
            features_frame=sim_features,
            prepare=sim_prepare,
            canonical=sim_features,  # the simulator's feature names ARE the rule vocabulary
            split=split_by_date(split),
            sensitive=tuple(split.sensitive_features),
            check_features=assert_fnol_only,
            load_holdouts=_sim_holdouts,
            stage_factories={"repair_cost": repair_cost_stage},
        )
    if name in ("sim_decades", "sim_decades_appraisal"):
        stage2 = name.endswith("_appraisal")
        return DatasetSpec(
            name=name,
            description="Multi-decade world 2002-2024: real GES/CRSS crashes, BLS-priced, "
            "simulated policy/fraud" + (" + appraisal (stage 2)" if stage2 else ""),
            real_labels=False,
            load_raw=lambda: sim_build.load("claims", data_dir() / "sim_decades"),
            features_frame=sim_appraisal_features if stage2 else sim_features,
            prepare=sim_appraisal_prepare if stage2 else sim_prepare,
            canonical=sim_features,
            split=split_by_date(split),
            sensitive=tuple(split.sensitive_features),
            check_features=_check_appraisal_stage if stage2 else assert_fnol_only,
        )
    if name == "sim_us_appraisal":
        return DatasetSpec(
            name=name,
            description="Stage 2 (after the independent appraisal): sim_us + appraisal features",
            real_labels=False,
            load_raw=lambda: sim_build.load("claims"),
            features_frame=sim_appraisal_features,
            prepare=sim_appraisal_prepare,
            canonical=sim_features,
            split=split_by_date(split),
            sensitive=tuple(split.sensitive_features),
            check_features=_check_appraisal_stage,
            load_holdouts=_sim_holdouts,
        )
    if name == "legacy_1990s":
        return DatasetSpec(
            name=name,
            description="Real Kaggle vehicle-claim fraud labels, 1994-96 (reality check)",
            real_labels=True,
            load_raw=_legacy_raw,
            features_frame=legacy_features,
            prepare=legacy_prepare,
            canonical=legacy_canonical,
            split=split_by_years(split),
            sensitive=tuple(split.sensitive_features),
        )
    raise KeyError(f"unknown dataset {name!r}; choose from {sorted(cfg.datasets)}")
