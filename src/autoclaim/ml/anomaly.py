"""Isolation Forest: an unsupervised second opinion that needs no fraud labels.

Score = percentile of the claim's anomaly among training claims (0 = most typical,
1 = most unusual), so it is comparable across model versions and easy to explain
("rarer than 97% of claims").
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from autoclaim.ml.frame import categorical_columns


class FrequencyEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Replace each category with its training frequency; unseen values become 0.

    For anomaly detection this is the signal we want: rare or never-seen values map to small
    numbers the forest isolates quickly. (One-hot with handle_unknown="ignore" maps unseen values
    to all zeros, which is the *most common* one-hot pattern, so they would look typical.)
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FrequencyEncoder":  # noqa: N803
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.freqs_ = {
            c: frame[c].astype("string").fillna("missing").value_counts(normalize=True).to_dict()
            for c in self.columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        frame = pd.DataFrame(X, columns=self.columns_)
        cols = [
            frame[c]
            .astype("string")
            .fillna("missing")
            .map(self.freqs_[c])
            .astype("Float64")
            .fillna(0.0)
            .to_numpy(dtype=float)
            for c in self.columns_
        ]
        return np.column_stack(cols) if cols else np.empty((len(frame), 0))


class AnomalyScorer:
    def __init__(self, features: Sequence[str], seed: int, n_estimators: int = 300) -> None:
        self.features = list(features)
        self.seed = seed
        self.n_estimators = n_estimators
        self.cats: list[str] = []

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.features].copy()
        for c in self.features:
            if c in self.cats:
                out[c] = out[c].astype("object")
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        return out

    def fit(self, train: pd.DataFrame) -> "AnomalyScorer":
        self.cats = categorical_columns(train, self.features)
        cats = self.cats
        numeric = [c for c in self.features if c not in cats]
        pre = ColumnTransformer(
            [
                ("cat", FrequencyEncoder(), cats),
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("pre", pre),
                (
                    "iforest",
                    IsolationForest(n_estimators=self.n_estimators, random_state=self.seed),
                ),
            ]
        )
        self.pipeline.fit(self._frame(train))
        # Sorted raw anomaly values of the training set, for percentile normalization.
        self.reference = np.sort(self._raw(train))
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        # score_samples: higher = more normal, so negate to get "higher = more anomalous".
        raw: np.ndarray = -self.pipeline.score_samples(self._frame(df))
        return raw

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Percentile in [0, 1] of each row's anomaly relative to the training claims."""
        ranks = np.searchsorted(self.reference, self._raw(df), side="right")
        out: np.ndarray = ranks / len(self.reference)
        return out
