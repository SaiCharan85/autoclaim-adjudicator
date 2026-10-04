import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from autoclaim.ml import models
from autoclaim.ml.frame import Y, feature_list
from autoclaim.ml.split import stratified_holdout


@pytest.fixture(autouse=True)
def _few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 80)


@pytest.fixture
def prepared(fraud_std: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train, test = stratified_holdout(fraud_std, 0.3, seed=0)
    return train, test, feature_list(fraud_std, exclude=["Sex", "MaritalStatus"])


@pytest.mark.parametrize("name", list(models.MODELS))
def test_each_model_learns_the_planted_signal(name: str, prepared) -> None:
    train, test, features = prepared
    model = models.make_model(name, features, seed=0)
    inner, val = stratified_holdout(train, 0.2, seed=0) if model.needs_val else (train, None)
    proba = model.fit(inner, val).predict_proba(test)
    assert proba.shape == (len(test),)
    assert np.all((proba >= 0) & (proba <= 1))
    assert roc_auc_score(test[Y], proba) > 0.65


@pytest.mark.parametrize("name", list(models.MODELS))
def test_each_model_tolerates_unseen_category_and_missing_value(name: str, prepared) -> None:
    train, test, features = prepared
    model = models.make_model(name, features, seed=0)
    inner, val = stratified_holdout(train, 0.2, seed=0) if model.needs_val else (train, None)
    model.fit(inner, val)
    odd = test.head(2).copy()
    odd["Make"] = "NeverSeenMake"
    odd["Age"] = np.nan
    assert np.isfinite(model.predict_proba(odd)).all()


def test_same_seed_same_predictions(prepared) -> None:
    train, test, features = prepared
    a = models.make_model("catboost", features, 3).fit(train, None).predict_proba(test)
    b = models.make_model("catboost", features, 3).fit(train, None).predict_proba(test)
    np.testing.assert_allclose(a, b)


def test_unknown_model_name() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        models.make_model("random_forest", ["Age"], 0)


def test_base_class_is_abstract() -> None:
    base = models._Base(["Age"], 0)
    with pytest.raises(NotImplementedError):
        base.fit(pd.DataFrame(), None)
    with pytest.raises(NotImplementedError):
        base.predict_proba(pd.DataFrame())


@pytest.mark.parametrize("name", ["catboost", "lightgbm", "xgboost"])
def test_boosted_models_report_trees_used(name: str, prepared) -> None:
    train, _, features = prepared
    inner, val = stratified_holdout(train, 0.2, seed=0)
    with_val = models.make_model(name, features, 0).fit(inner, val)
    assert 1 <= with_val.n_trees <= models.MAX_TREES
    no_val = models.make_model(name, features, 0).fit(train, None)
    assert no_val.n_trees == models.MAX_TREES  # no early stopping -> all trees


@pytest.mark.parametrize("name", ["logistic", "ebm"])
def test_non_boosted_models_have_no_tree_count(name: str, prepared) -> None:
    train, _, features = prepared
    assert models.make_model(name, features, 0).fit(train, None).n_trees is None
