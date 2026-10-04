"""Fitted feature stages: features learned from training data (unlike stateless spec features).

Leakage rule for every stage: training rows get OUT-OF-FOLD values (each row is scored by a model
that never saw it), and validation/test/runtime rows are scored by a model fit on all training
rows. In-sample values would be unrealistically good on training rows, so the fraud model would
learn from a feature that looks nothing like what it sees later.
"""

from collections.abc import Sequence
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from autoclaim.ml.frame import CategoryVocab, Y, categorical_columns


class FeatureStage(Protocol):
    name: str
    outputs: tuple[str, ...]

    def fit_oof(self, train: pd.DataFrame) -> pd.DataFrame: ...

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...


class ExpectedAmountStage:
    """How much bigger is this claim than a legitimate claim with the same characteristics?

    A regressor learns log(claimed amount) from damage descriptors (cause, damage extent, towing,
    vehicle value, ...) on training claims NOT confirmed as fraud. Outputs:
    - expected_log_amount: what such a loss usually costs
    - amount_residual: log(claimed) - expected; > 0 means "pricier than such damage usually is"
    Insurers do the same when they audit a repair estimate against typical costs.
    """

    outputs = ("expected_log_amount", "amount_residual")

    def __init__(
        self,
        inputs: Sequence[str],
        amount_col: str,
        seed: int,
        n_folds: int = 5,
        name: str = "repair_cost",
    ) -> None:
        self.inputs = list(inputs)
        self.amount_col = amount_col
        self.seed = seed
        self.n_folds = n_folds
        self.name = name

    def _log_amount(self, frame: pd.DataFrame) -> np.ndarray:
        amount = pd.to_numeric(frame[self.amount_col], errors="coerce").to_numpy(dtype=float)
        out: np.ndarray = np.log1p(np.clip(amount, 0, None))
        return out

    def _fit(self, rows: pd.DataFrame) -> tuple[lgb.LGBMRegressor, CategoryVocab]:
        cats = categorical_columns(rows, self.inputs)
        vocab = CategoryVocab.fit(rows, cats)
        target = self._log_amount(rows)
        ok = np.isfinite(target)
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            random_state=self.seed,
            verbose=-1,
        )
        model.fit(vocab.transform(rows, self.inputs)[ok], target[ok])
        return model, vocab

    @staticmethod
    def _predict(
        model: lgb.LGBMRegressor, vocab: CategoryVocab, frame: pd.DataFrame, inputs: list[str]
    ) -> np.ndarray:
        return np.asarray(model.predict(vocab.transform(frame, inputs)), dtype=float)

    def _attach(self, frame: pd.DataFrame, expected: np.ndarray) -> pd.DataFrame:
        out = frame.copy()
        out["expected_log_amount"] = expected
        out["amount_residual"] = self._log_amount(frame) - expected
        return out

    def fit_oof(self, train: pd.DataFrame) -> pd.DataFrame:
        legit = train[Y].to_numpy() == 0
        expected = np.full(len(train), np.nan)
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for fit_idx, score_idx in folds.split(train):
            rows = fit_idx[legit[fit_idx]]  # never learns from confirmed fraud or the scored fold
            model, vocab = self._fit(train.iloc[rows])
            expected[score_idx] = self._predict(model, vocab, train.iloc[score_idx], self.inputs)
        self.model, self.vocab = self._fit(train[legit])  # for validation / test / runtime
        return self._attach(train, expected)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._attach(frame, self._predict(self.model, self.vocab, frame, self.inputs))


def fit_stages(stages: Sequence[FeatureStage], train: pd.DataFrame) -> pd.DataFrame:
    for stage in stages:
        train = stage.fit_oof(train)
    return train


def apply_stages(stages: Sequence[FeatureStage], frame: pd.DataFrame) -> pd.DataFrame:
    for stage in stages:
        frame = stage.transform(frame)
    return frame


def stage_outputs(stages: Sequence[FeatureStage]) -> list[str]:
    return [col for stage in stages for col in stage.outputs]
