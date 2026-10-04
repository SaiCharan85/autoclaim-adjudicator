from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autoclaim.config import FraudModelConfig
from autoclaim.ml import models
from autoclaim.ml.experiment import LEVERS, Variant, compare, run_variant
from autoclaim.ml.frame import AMOUNT, WEIGHT, feature_list
from autoclaim.ml.stages import ExpectedAmountStage


@pytest.fixture(autouse=True)
def _few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 40)


@pytest.fixture
def frame(fraud_std: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    amount = rng.lognormal(8, 0.6, len(fraud_std)) * np.where(fraud_std["y"] == 1, 2.0, 1.0)
    return fraud_std.assign(claimed_amount=amount, **{AMOUNT: amount})


@pytest.fixture
def spec(legacy_spec):
    def factory(seed: int) -> ExpectedAmountStage:
        return ExpectedAmountStage(["Make", "VehiclePrice"], "claimed_amount", seed=seed)

    return replace(legacy_spec, stage_factories={"cost": factory})


def test_baseline_runs_every_seed(frame, spec, fraud_cfg: FraudModelConfig) -> None:
    res = run_variant(Variant("baseline"), frame, spec, fraud_cfg)
    _, val, _ = spec.split(frame)
    assert len(res.metrics) == len(fraud_cfg.seeds)
    assert len(res.scores) == len(val)
    assert {"pr_auc", "recall_at_budget", "net_savings_per_1k"} <= set(res.metrics.columns)


def test_stage_variant_adds_features(frame, spec, fraud_cfg: FraudModelConfig) -> None:
    res = run_variant(Variant("cost", stages=("cost",)), frame, spec, fraud_cfg)
    assert res.metrics["pr_auc"].between(0, 1).all()


def test_ensemble_and_weights(frame, spec, fraud_cfg: FraudModelConfig) -> None:
    ens = run_variant(Variant("ens", models=("catboost", "lightgbm")), frame, spec, fraud_cfg)
    assert ens.metrics["train_minus_val_pr_auc"].isna().all()  # gap is per single model only
    assert ens.scores.min() >= 0 and ens.scores.max() <= 1  # rank-averaged
    weighted = run_variant(Variant("w", weight_by_amount=True), frame, spec, fraud_cfg)
    assert len(weighted.metrics) == len(fraud_cfg.seeds)


def test_compare_pairs_against_reference(frame, spec, fraud_cfg: FraudModelConfig) -> None:
    results = [
        run_variant(Variant("baseline"), frame, spec, fraud_cfg),
        run_variant(Variant("deeper", params={"depth": 6}), frame, spec, fraud_cfg),
    ]
    _, val, _ = spec.split(frame)
    table = compare(results, val, fraud_cfg, "baseline")
    assert table.loc["baseline", "delta_recall_vs_ref"] == "reference"
    assert table.loc["deeper", "delta_recall_vs_ref"].startswith(("+", "-"))
    assert table.loc["deeper", "delta_savings_vs_ref"].startswith(("+", "-"))
    assert table.to_string().isascii()


def test_lever_registry() -> None:
    assert {"repair_cost", "weighting", "ensemble", "grid"} == set(LEVERS)
    assert all(isinstance(v, Variant) for vs in LEVERS.values() for v in vs)


def test_catboost_params_override_and_weights(fraud_std: pd.DataFrame) -> None:
    feats = feature_list(fraud_std)
    m = models.make_model("catboost", feats, 0, {"depth": 2}).fit(fraud_std, None)
    assert m.model.get_params()["depth"] == 2
    weighted = fraud_std.assign(**{WEIGHT: 1.0})
    assert models.make_model("lightgbm", feats, 0).fit(weighted, None).n_trees is not None


@pytest.mark.parametrize("name", ["logistic", "ebm"])
def test_unsupported_weights_raise(fraud_std: pd.DataFrame, name: str) -> None:
    weighted = fraud_std.assign(**{WEIGHT: 1.0})
    with pytest.raises(NotImplementedError, match="sample weights"):
        models.make_model(name, feature_list(fraud_std), 0).fit(weighted, None)


def test_xgboost_accepts_weights(fraud_std: pd.DataFrame) -> None:
    weighted = fraud_std.assign(**{WEIGHT: 1.0})
    assert models.make_model("xgboost", feature_list(fraud_std), 0).fit(weighted, None).n_trees
