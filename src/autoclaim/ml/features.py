"""Cleaning and feature preparation for the fraud table.

Decisions are explained in docs/data_notes.md.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

TARGET = "FraudFound_P"

MONTH_INDEX = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
}

# Not features: unique ID, redundant with BasePolicy (+ inconsistent), carrier-internal rep ID,
# and calendar year (drifts; future years are never seen in training).
NON_FEATURES = ("PolicyNumber", "PolicyType", "RepNumber", "Year")

NUMERIC_FEATURES = (
    "Age",
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "DriverRating",
    "Deductible",
    "claim_lag_weeks",
)


def claim_lag_weeks(df: pd.DataFrame) -> pd.Series:
    """Approximate weeks from accident to claim, from month + week-of-month only (no exact dates).

    A claimed month earlier than the accident month is read as the following year. NaN if the
    claimed month is unknown. Coarse by design: good enough to flag "reported months later".
    """
    acc_month = df["Month"].map(MONTH_INDEX)
    claim_month = df["MonthClaimed"].map(MONTH_INDEX)
    month_diff = (claim_month - acc_month) % 12
    lag = month_diff * 4 + (df["WeekOfMonthClaimed"] - df["WeekOfMonth"])
    return lag.clip(lower=0).astype("float64")


def time_order(df: pd.DataFrame) -> pd.Series:
    """Monotone accident-time key (year, month, week) for time-aware splits."""
    return df["Year"] * 100 + df["Month"].map(MONTH_INDEX) * 5 + df["WeekOfMonth"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: sentinel values to NaN, plus derived features."""
    out = df.copy()
    out["Age"] = out["Age"].astype("float64").where(out["Age"] != 0, np.nan)
    for col in ("DayOfWeekClaimed", "MonthClaimed"):
        out[col] = out[col].where(out[col].astype(str) != "0", np.nan)
    out["claim_lag_weeks"] = claim_lag_weeks(out)
    return out


def feature_columns(df: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    drop = {TARGET, *NON_FEATURES, *exclude}
    return [c for c in df.columns if c not in drop]


def categorical_columns(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if c not in NUMERIC_FEATURES]


def for_catboost(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """CatBoost needs string categoricals without NaN."""
    out = df[list(columns)].copy()
    for c in categorical_columns(columns):
        out[c] = out[c].astype("string").fillna("missing").astype(str)
    return out


class CategoryVocab:
    """Fixed category levels learned on training data, so train/test/runtime encode identically."""

    def __init__(self, levels: dict[str, list[str]]) -> None:
        self.levels = levels

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Sequence[str]) -> "CategoryVocab":
        cats = categorical_columns(columns)
        return cls({c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cats})

    def transform(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """pandas `category` dtype with fixed levels; unseen values become NaN."""
        out = df[list(columns)].copy()
        for c, levels in self.levels.items():
            if c in out:
                out[c] = pd.Categorical(out[c].astype("string"), categories=levels)
        return out
