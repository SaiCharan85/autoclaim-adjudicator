"""The five benchmark classifiers behind one interface: fit(train, val) -> predict_proba(frame).

All boosted models early-stop on ROC-AUC of the validation slice, the same for every model.
Why not the alternatives (see docs/model_card.md, "Protocol history"):
- PR-AUC is too noisy on a few hundred validation frauds to pick a tree count.
- Log-loss also scores calibration; under base-rate drift it worsens almost immediately and
  stopped models at ~12 trees.
ROC-AUC is rank-based (how the model is used: ranking claims for review) and base-rate invariant.

Frames follow ml/frame.py (features + `y`). Categorical columns are inferred from dtypes at fit
time and stored. CatBoost uses native string categoricals, LightGBM/XGBoost use pandas `category`
with a fixed vocabulary, logistic regression one-hot encodes, and EBM handles strings itself.
"""

from collections.abc import Callable, Sequence
from typing import Any

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

from autoclaim.ml.frame import CategoryVocab, Y, categorical_columns, for_catboost

EARLY_STOPPING_ROUNDS = 100
MAX_TREES = 2000
# Chosen on validation (docs/model_card.md): depth 4 + stronger L2 matched depth 6 on recall
# (+0.012, CI [-0.002, +0.023]) while cutting the train-validation PR-AUC gap 0.20 -> 0.13.
CATBOOST_PARAMS: dict[str, Any] = {"learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 10}


class _Base:
    name = "base"
    needs_val = False

    def __init__(self, features: Sequence[str], seed: int) -> None:
        self.features = list(features)
        self.seed = seed
        self.cats: list[str] = []

    def _learn_types(self, train: pd.DataFrame) -> None:
        self.cats = categorical_columns(train, self.features)

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
        self._learn_types(train)
        self.model = CatBoostClassifier(
            iterations=MAX_TREES,
            **CATBOOST_PARAMS,
            eval_metric="AUC",
            random_seed=self.seed,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )
        eval_set = None
        if val is not None:
            eval_set = (for_catboost(val, self.features, self.cats), val[Y])
        self.model.fit(
            for_catboost(train, self.features, self.cats),
            train[Y],
            cat_features=self.cats,
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if val is not None else None,
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = for_catboost(frame, self.features, self.cats)
        proba: np.ndarray = self.model.predict_proba(x)[:, 1]
        return proba

    @property
    def n_trees(self) -> int | None:
        return int(self.model.tree_count_)


class LightGBMModel(_Base):
    name = "lightgbm"
    needs_val = True

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "LightGBMModel":
        self._learn_types(train)
        self.vocab = CategoryVocab.fit(train, self.cats)
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
                "eval_set": [(self.vocab.transform(val, self.features), val[Y])],
                "callbacks": [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            }
        self.model.fit(self.vocab.transform(train, self.features), train[Y], **kwargs)
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
        self._learn_types(train)
        self.vocab = CategoryVocab.fit(train, self.cats)
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
            eval_set = [(self.vocab.transform(val, self.features), val[Y])]
        self.model.fit(
            self.vocab.transform(train, self.features), train[Y], eval_set=eval_set, verbose=False
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
        self._learn_types(train)
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
        self.model.fit(self._as_typed(train), train[Y])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        proba: np.ndarray = self.model.predict_proba(self._as_typed(frame))[:, 1]
        return proba

    def _as_typed(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[self.features].copy()
        for c in self.features:
            if c in self.cats:
                out[c] = out[c].astype("object")
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        return out


class EBMModel(_Base):
    name = "ebm"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None) -> "EBMModel":
        self._learn_types(train)
        types = ["nominal" if c in self.cats else "continuous" for c in self.features]
        self.model = ExplainableBoostingClassifier(
            feature_types=types, outer_bags=8, random_state=self.seed, n_jobs=-1
        )
        self.model.fit(for_catboost(train, self.features, self.cats), train[Y])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = for_catboost(frame, self.features, self.cats)
        proba: np.ndarray = self.model.predict_proba(x)[:, 1]
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
