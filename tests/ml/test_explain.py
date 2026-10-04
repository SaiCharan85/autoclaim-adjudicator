import numpy as np
import pandas as pd
from catboost import Pool

from autoclaim.ml.explain import ShapReason, shap_matrix, top_reasons
from autoclaim.ml.frame import for_catboost


def test_shap_shape_and_additivity(trained_artifacts, fraud_std: pd.DataFrame) -> None:
    art = trained_artifacts
    frame = fraud_std.head(10)
    shap = shap_matrix(art.model, frame, art.features, art.categorical)
    assert shap.shape == (10, len(art.features))
    x = for_catboost(frame, art.features, art.categorical)
    full = art.model.get_feature_importance(
        Pool(x, cat_features=art.categorical), type="ShapValues"
    )
    raw = art.model.predict(x, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(shap.sum(axis=1) + full[:, -1], raw, atol=1e-6)  # exact TreeSHAP


def test_top_reasons_sorted_and_capped(trained_artifacts, fraud_std: pd.DataFrame) -> None:
    art = trained_artifacts
    reasons = top_reasons(art.model, fraud_std.head(5), art.features, art.categorical, k=3)
    assert len(reasons) == 5
    for row in reasons:
        assert len(row) <= 3
        magnitudes = [abs(r.contribution) for r in row]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(r.feature in art.features for r in row)


def test_missing_values_shown_as_missing(trained_artifacts, fraud_std: pd.DataFrame) -> None:
    art = trained_artifacts
    frame = fraud_std.head(1).copy()
    frame[art.features] = frame[art.features].astype(object)
    frame.loc[:, art.features] = np.nan
    reasons = top_reasons(art.model, frame, art.features, art.categorical, k=len(art.features))[0]
    assert all(r.value == "missing" for r in reasons)


def test_direction_property() -> None:
    assert ShapReason(feature="f", value="v", contribution=0.2).direction == "raises risk"
    assert ShapReason(feature="f", value="v", contribution=-0.2).direction == "lowers risk"
