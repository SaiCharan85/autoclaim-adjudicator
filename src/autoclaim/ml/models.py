"""The five benchmark classifiers behind one interface: fit(train, val) -> predict_proba(frame).

All boosted models early-stop on ROC-AUC of the validation slice, the same for every model.
Why not the alternatives (see docs/model_card.md, "Protocol history"):
- PR-AUC is too noisy on ~160 validation frauds to pick a tree count.
- Log-loss also scores calibration; the validation slice has a lower fraud rate than training
  (drift), so log-loss worsens almost immediately and stopped models at ~12 trees.
ROC-AUC is rank-based (how the model is used: ranking claims for review) and base-rate invariant.

Each model receives cleaned frames plus the feature list, and does its own encoding:
CatBoost uses native string categoricals, LightGBM/XGBoost use pandas `category` with a fixed
vocabulary, logistic regression one-hot encodes, and EBM handles strings itself.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autoclaim.ml.features import TARGET, CategoryVocab, categorical_columns, for_catboost

EARLY_STOPPING_ROUNDS = 100
MAX_TREES = 2000


class FraudClassifier(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "FraudClassifier": ...

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


class _Base:
    name = "base"
    needs_val = False

    def __init__(self, features: Sequence[str], seed: int) -> None:
        self.features = list(features)
        self.cats = categorical_columns(self.features)
        self.seed = seed

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "_Base":
        raise NotImplementedError

    @property
    def n_trees(self) -> int | None:
        """Trees actually used after early stopping (None for non-tree models)."""
        return None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class CatBoostModel(_Base):
    name = "catboost"
    needs_val = True

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "CatBoostModel":
        self.model = CatBoostClassifier(
            iterations=MAX_TREES,
            learning_rate=0.05,
            depth=6,
            eval_metric="AUC",
            random_seed=self.seed,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )
        eval_set = None if val is None else (for_catboost(val, self.features), val[TARGET])
        self.model.fit(
            for_catboost(train, self.features),
            train[TARGET],
            cat_features=self.cats,
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if val is not None else None,
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        proba: np.ndarray = self.model.predict_proba(for_catboost(frame, self.features))[:, 1]
        return proba

    @property
    def n_trees(self) -> int | None:
        return int(self.model.tree_count_)


class LightGBMModel(_Base):
    name = "lightgbm"
    needs_val = True

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "LightGBMModel":
        self.vocab = CategoryVocab.fit(train, self.features)
        self.model = lgb.LGBMClassifier(
            n_estimators=MAX_TREES,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            metric="auc",  # replaces the default logloss, so early stopping tracks AUC only
            verbose=-1,
        )
        kwargs: dict[str, Any] = {}
        if val is not None:
            kwargs = {
                "eval_set": [(self.vocab.transform(val, self.features), val[TARGET])],
                "callbacks": [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            }
        self.model.fit(self.vocab.transform(train, self.features), train[TARGET], **kwargs)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(self.vocab.transform(frame, self.features))
        return np.asarray(raw)[:, 1]

    @property
    def n_trees(self) -> int | None:
        best = self.model.best_iteration_
        return int(best) if best else int(self.model.n_estimators)


class XGBoostModel(_Base):
    name = "xgboost"
    needs_val = True

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "XGBoostModel":
        self.vocab = CategoryVocab.fit(train, self.features)
        self.model = xgb.XGBClassifier(
            n_estimators=MAX_TREES,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if val is not None else None,
            random_state=self.seed,
        )
        eval_set = None
        if val is not None:
            eval_set = [(self.vocab.transform(val, self.features), val[TARGET])]
        self.model.fit(
            self.vocab.transform(train, self.features),
            train[TARGET],
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(self.vocab.transform(frame, self.features))
        return np.asarray(raw)[:, 1]

    @property
    def n_trees(self) -> int | None:
        best = getattr(self.model, "best_iteration", None)  # 0-based; absent without early stop
        return int(best) + 1 if best is not None else int(self.model.n_estimators or MAX_TREES)


class LogisticModel(_Base):
    name = "logistic"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "LogisticModel":
        numeric = [c for c in self.features if c not in self.cats]
        pre = ColumnTransformer(
            [
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value="missing"),
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                    self.cats,
                ),
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ]
        )
        self.model = Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
            ]
        )
        self.model.fit(self._as_object(train), train[TARGET])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        proba: np.ndarray = self.model.predict_proba(self._as_object(frame))[:, 1]
        return proba

    def _as_object(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.features].astype({c: "object" for c in self.cats})


class EBMModel(_Base):
    name = "ebm"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "EBMModel":
        types = ["nominal" if c in self.cats else "continuous" for c in self.features]
        self.model = ExplainableBoostingClassifier(
            feature_types=types, outer_bags=8, random_state=self.seed, n_jobs=-1
        )
        self.model.fit(for_catboost(train, self.features), train[TARGET])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        proba: np.ndarray = self.model.predict_proba(for_catboost(frame, self.features))[:, 1]
        return proba


ModelFactory = Callable[[Sequence[str], int], _Base]

MODELS: dict[str, ModelFactory] = {
    "logistic": LogisticModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "ebm": EBMModel,
}


def make_model(name: str, features: Sequence[str], seed: int) -> _Base:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](features, seed)
