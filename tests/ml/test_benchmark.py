from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autoclaim.config import FraudModelConfig
from autoclaim.lines.auto.simulator.columns import LeakageError, assert_fnol_only
from autoclaim.ml import models
from autoclaim.ml.benchmark import evaluate, run_benchmark
from autoclaim.ml.frame import AMOUNT, Y, feature_list


@pytest.fixture(autouse=True)
def _few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 60)


def test_evaluate_reports_gap_and_trees(
    legacy_spec, fraud_std: pd.DataFrame, fraud_cfg: FraudModelConfig
) -> None:
    train, val, _ = legacy_spec.split(fraud_std)
    run = evaluate("catboost", feature_list(fraud_std), train, val, 0, fraud_cfg)
    assert 0 <= run.metrics["pr_auc"] <= 1
    assert "train_pr_auc" in run.metrics
    assert run.n_trees is not None
    assert len(run.scores) == len(val)
    assert "net_savings_per_1k" not in run.metrics  # no amount column in this dataset


def test_evaluate_adds_savings_when_amount_present(
    legacy_spec, fraud_std: pd.DataFrame, fraud_cfg: FraudModelConfig
) -> None:
    frame = fraud_std.assign(**{AMOUNT: 5000.0})
    train, val, _ = legacy_spec.split(frame)
    run = evaluate("logistic", feature_list(frame), train, val, 0, fraud_cfg)
    assert "net_savings_per_1k" in run.metrics


def test_validation_only_by_default(
    legacy_spec, fraud_std: pd.DataFrame, fraud_cfg: FraudModelConfig
) -> None:
    bench = run_benchmark(fraud_std, legacy_spec, fraud_cfg, models=["logistic", "catboost"])
    assert bench.test is None  # the locked test set is untouched without final=True
    assert list(bench.validation.index) == ["logistic", "catboost", "catboost + Sex/MaritalStatus"]
    assert "Sex" not in bench.features
    row = bench.validation.loc["logistic"]
    assert row["pr_auc"].count("+/-") == 1  # mean +/- std over seeds
    assert row["delta_recall_vs_catboost"].startswith(("+", "-"))
    assert bench.validation.loc["catboost", "delta_recall_vs_catboost"] == "n/a"
    assert bench.validation.to_string().isascii()  # Windows consoles can't print every symbol


def test_final_scores_the_test_set(
    legacy_spec, fraud_std: pd.DataFrame, fraud_cfg: FraudModelConfig
) -> None:
    bench = run_benchmark(
        fraud_std, legacy_spec, fraud_cfg, models=["logistic"], final=True, ablation_model=None
    )
    assert bench.test is not None
    assert list(bench.test.index) == ["logistic"]


def test_leaky_feature_is_rejected(
    legacy_spec, fraud_std: pd.DataFrame, fraud_cfg: FraudModelConfig
) -> None:
    strict = replace(legacy_spec, check_features=assert_fnol_only)  # 1990s names aren't tagged
    with pytest.raises(LeakageError):
        run_benchmark(fraud_std, strict, fraud_cfg, models=["logistic"])


def test_train_eval_gap_reported(
    legacy_spec, fraud_std: pd.DataFrame, fraud_cfg: FraudModelConfig
) -> None:
    rng = np.random.default_rng(0)
    noisy = fraud_std.assign(**{Y: rng.integers(0, 2, len(fraud_std))})
    bench = run_benchmark(noisy, legacy_spec, fraud_cfg, models=["catboost"], ablation_model=None)
    gap = float(bench.validation.loc["catboost", "train_minus_eval_pr_auc"].split()[0])
    assert gap >= 0  # random labels: a model can only look better on its own training rows
