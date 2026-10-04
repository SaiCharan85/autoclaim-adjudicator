import pandas as pd
import pytest

from autoclaim.config import FraudModelConfig
from autoclaim.ml import models
from autoclaim.ml.benchmark import evaluate_cv, evaluate_time_split, run_benchmark
from autoclaim.ml.features import clean, feature_columns


@pytest.fixture(autouse=True)
def _few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 60)


def test_time_split_result(fraud_frame: pd.DataFrame, small_fraud_cfg: FraudModelConfig) -> None:
    df = clean(fraud_frame)
    res = evaluate_time_split(df, "logistic", feature_columns(df), small_fraud_cfg, seed=0)
    assert res.variant == "time"
    assert 0 <= res.metrics["pr_auc"] <= 1
    assert res.metrics["base_rate"] == pytest.approx(df[df["Year"] == 1996]["FraudFound_P"].mean())
    assert res.fit_seconds >= 0


def test_cv_result_has_mean_and_std(
    fraud_frame: pd.DataFrame, small_fraud_cfg: FraudModelConfig
) -> None:
    df = clean(fraud_frame)
    res = evaluate_cv(df, "catboost", feature_columns(df), small_fraud_cfg, seed=0)
    assert {"pr_auc_mean", "pr_auc_std", "recall_at_budget_mean"} <= set(res.metrics)


def test_run_benchmark_table_with_ablation(
    fraud_frame: pd.DataFrame, small_fraud_cfg: FraudModelConfig
) -> None:
    table = run_benchmark(
        clean(fraud_frame), small_fraud_cfg, seed=0, models=["logistic", "catboost"]
    )
    assert list(table.index) == ["logistic", "catboost", "catboost + Sex/MaritalStatus"]
    assert {"pr_auc_1996", "recall@budget_1996", "pr_auc_cv", "n_trees", "fit_s"} <= set(
        table.columns
    )
    assert table.loc["logistic", "n_trees"] == "n/a"
    assert table["pr_auc_1996"].between(0, 1).all()
    assert table.loc["catboost", "delta_recall_vs_catboost"] == "n/a"
    assert table.loc["logistic", "delta_recall_vs_catboost"].startswith(("+", "-"))
    assert table.loc["logistic", "pr_auc_1996_ci"].startswith("[")
    # Windows consoles (cp1252) can't print symbols like a Greek delta; keep the table ASCII.
    assert table.to_string().isascii()


def test_no_ablation_without_sensitive_features(
    fraud_frame: pd.DataFrame, small_fraud_cfg: FraudModelConfig
) -> None:
    cfg = small_fraud_cfg.model_copy(update={"sensitive_features": []})
    table = run_benchmark(clean(fraud_frame), cfg, seed=0, models=["logistic"])
    assert list(table.index) == ["logistic"]


def test_time_split_keeps_test_predictions(
    fraud_frame: pd.DataFrame, small_fraud_cfg: FraudModelConfig
) -> None:
    df = clean(fraud_frame)
    res = evaluate_time_split(df, "logistic", feature_columns(df), small_fraud_cfg, seed=0)
    assert len(res.scores) == len(res.y_test) == int((df["Year"] == 1996).sum())


def test_no_reference_model_in_run(
    fraud_frame: pd.DataFrame, small_fraud_cfg: FraudModelConfig
) -> None:
    cfg = small_fraud_cfg.model_copy(update={"sensitive_features": []})
    table = run_benchmark(clean(fraud_frame), cfg, seed=0, models=["logistic", "xgboost"])
    assert (table["delta_recall_vs_catboost"] == "n/a").all()
