"""The standard modeling frame shared by every dataset.

A dataset spec turns its raw table into: feature columns + `y` (label) + `t` (sortable time key)
+ optionally `amount` (claimed amount, for the cost-sensitive savings metric). Everything in
`autoclaim.ml` works on this contract, so the same pipeline serves real and simulated data.
"""

from collections.abc import Sequence

import pandas as pd

Y = "y"
T = "t"
AMOUNT = "amount"
RESERVED = (Y, T, AMOUNT)


def feature_list(frame: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    drop = {*RESERVED, *exclude}
    return [c for c in frame.columns if c not in drop]


def categorical_columns(frame: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """Non-numeric columns are categorical (bools are numeric). Inferred once at fit time and
    stored, so runtime rows with missing values can't flip a column's type."""
    return [c for c in features if not pd.api.types.is_numeric_dtype(frame[c])]


def for_catboost(frame: pd.DataFrame, features: Sequence[str], cats: Sequence[str]) -> pd.DataFrame:
    """CatBoost needs string categoricals without NaN and numeric (not bool/object) numbers."""
    out = frame[list(features)].copy()
    for c in features:
        if c in cats:
            out[c] = out[c].astype("string").fillna("missing").astype(str)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


class CategoryVocab:
    """Fixed category levels learned on training data, so train/test/runtime encode identically."""

    def __init__(self, levels: dict[str, list[str]]) -> None:
        self.levels = levels

    @classmethod
    def fit(cls, frame: pd.DataFrame, cats: Sequence[str]) -> "CategoryVocab":
        return cls({c: sorted(frame[c].dropna().astype(str).unique().tolist()) for c in cats})

    def transform(self, frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        """pandas `category` dtype with fixed levels (unseen -> NaN); numerics as float."""
        out = frame[list(features)].copy()
        for c in features:
            if c in self.levels:
                out[c] = pd.Categorical(out[c].astype("string"), categories=self.levels[c])
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        return out
