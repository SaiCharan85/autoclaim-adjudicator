import numpy as np
import pandas as pd
from catboost import Pool

from autoclaim.ml.explain import ShapReason, shap_matrix, top_reasons
from autoclaim.ml.features import categorical_columns, clean, for_catboost


def test_shap_shape_and_additivity(trained_artifacts, fraud_frame: pd.DataFrame) -> None:
    art = trained_artifacts
    frame = clean(fraud_frame).head(10)
    shap = shap_matrix(art.model, frame, art.features)
    assert shap.shape == (10, len(art.features))
    # TreeSHAP is exact: contributions + bias == raw log-odds prediction.
    full = art.model.get_feature_importance(
        Pool(for_catboost(frame, art.features), cat_features=categorical_columns(art.features)),
        type="ShapValues",
    )
    raw = art.model.predict(for_catboost(frame, art.features), prediction_type="RawFormulaVal")
    np.testing.assert_allclose(shap.sum(axis=1) + full[:, -1], raw, atol=1e-6)


def test_top_reasons_sorted_and_capped(trained_artifacts, fraud_frame: pd.DataFrame) -> None:
    art = trained_artifacts
    frame = clean(fraud_frame).head(5)
    reasons = top_reasons(art.model, frame, art.features, k=3)
    assert len(reasons) == 5
    for row in reasons:
        assert len(row) <= 3
        magnitudes = [abs(r.contribution) for r in row]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(r.feature in art.features for r in row)


def test_missing_values_shown_as_missing(trained_artifacts, fraud_frame: pd.DataFrame) -> None:
    art = trained_artifacts
    frame = clean(fraud_frame).head(1).copy()
    frame[art.features] = frame[art.features].astype(object)
    frame.loc[:, art.features] = np.nan
    reasons = top_reasons(art.model, frame, art.features, k=len(art.features))[0]
    assert all(r.value == "missing" for r in reasons)


def test_direction_property() -> None:
    assert ShapReason(feature="f", value="v", contribution=0.2).direction == "raises risk"
    assert ShapReason(feature="f", value="v", contribution=-0.2).direction == "lowers risk"
