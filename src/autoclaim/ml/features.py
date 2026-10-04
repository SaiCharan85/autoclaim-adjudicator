"""Cleaning for the real 1994-96 Kaggle fraud table (the reality-check dataset).

Decisions are explained in docs/data_notes.md. Generic modeling helpers live in ml/frame.py.
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
