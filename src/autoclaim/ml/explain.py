"""Per-claim fraud reasons from CatBoost's built-in TreeSHAP values (no extra `shap` dependency).

Contributions are in log-odds: positive pushes towards fraud, negative away from it.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pydantic import BaseModel

from autoclaim.ml.features import categorical_columns, for_catboost


class ShapReason(BaseModel):
    feature: str
    value: str
    contribution: float  # log-odds

    @property
    def direction(self) -> str:
        return "raises risk" if self.contribution > 0 else "lowers risk"


def shap_matrix(
    model: CatBoostClassifier, frame: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    """(n_rows, n_features) SHAP values; CatBoost's trailing bias column is dropped."""
    pool = Pool(for_catboost(frame, features), cat_features=categorical_columns(features))
    values: np.ndarray = model.get_feature_importance(pool, type="ShapValues")
    return values[:, :-1]


def top_reasons(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    features: Sequence[str],
    k: int = 5,
    min_abs: float = 1e-6,
) -> list[list[ShapReason]]:
    """Top-k features by |SHAP| for each row, largest first."""
    shap = shap_matrix(model, frame, features)
    cols = frame[list(features)]
    raw_values = cols.astype(object).where(cols.notna(), "missing")
    order = np.argsort(-np.abs(shap), axis=1, kind="stable")[:, :k]
    results = []
    for i, idx in enumerate(order):
        results.append(
            [
                ShapReason(
                    feature=features[j],
                    value=str(raw_values.iat[i, j]),
                    contribution=round(float(shap[i, j]), 4),
                )
                for j in idx
                if abs(shap[i, j]) >= min_abs
            ]
        )
    return results
