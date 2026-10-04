import numpy as np
import pandas as pd
import pytest

from autoclaim.ml.canaries import (
    CanaryError,
    chance_tolerance,
    run_canaries,
    shuffled_label_auc,
    single_feature_aucs,
)
from autoclaim.ml.frame import Y, feature_list


@pytest.fixture
def splits(legacy_spec, fraud_std: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val, _ = legacy_spec.split(fraud_std)
    return train, val


def test_clean_data_passes(splits, fraud_std: pd.DataFrame) -> None:
    train, val = splits
    report = run_canaries(train, val, feature_list(fraud_std), seed=0)
    assert report.ok
    assert len(report.shuffled_aucs) == 5
    assert report.flagged == []


def test_label_column_used_as_feature_is_caught(splits, fraud_std: pd.DataFrame) -> None:
    """The classic pipeline bug: the label itself ends up in the feature list. Shuffling the
    label column in the frame shuffles that feature too, so the model maps it perfectly and the
    real validation labels score about 1.0 on every permutation."""
    train, val = splits
    with pytest.raises(CanaryError, match="mean shuffled AUC"):
        run_canaries(train, val, [*feature_list(fraud_std), Y], seed=0)


def test_one_sided_below_chance_is_not_a_leak() -> None:
    from autoclaim.ml.canaries import CanaryReport

    report = CanaryReport(shuffled_auc=0.40, shuffled_tolerance=0.05)
    assert report.shuffled_ok  # scoring below chance cannot come from leaking the label


def test_planted_leak_is_caught(splits, fraud_std: pd.DataFrame) -> None:
    train, val = splits
    leak = "fraud_risk_score"
    train = train.assign(**{leak: train[Y] * 80 + np.random.default_rng(0).random(len(train))})
    val = val.assign(**{leak: val[Y] * 80 + np.random.default_rng(1).random(len(val))})
    with pytest.raises(CanaryError, match=leak):
        run_canaries(train, val, [*feature_list(fraud_std), leak], seed=0)


def test_leaky_categorical_is_caught(splits, fraud_std: pd.DataFrame) -> None:
    train, val = splits
    train = train.assign(status=np.where(train[Y] == 1, "Under Review", "Approved"))
    val = val.assign(status=np.where(val[Y] == 1, "Under Review", "Approved"))
    report = run_canaries(train, val, [*feature_list(fraud_std), "status"], seed=0, strict=False)
    assert report.flagged == ["status"]
    assert not report.ok


def test_shuffled_auc_near_chance(splits, fraud_std: pd.DataFrame) -> None:
    train, val = splits
    assert abs(shuffled_label_auc(train, val, feature_list(fraud_std), seed=3) - 0.5) < 0.15


def test_single_feature_aucs_sorted_and_direction_free(splits) -> None:
    train, val = splits
    train = train.assign(neg=-train[Y].astype(float))
    val = val.assign(neg=-val[Y].astype(float))
    aucs = single_feature_aucs(train, val, ["neg", "Make"])
    assert next(iter(aucs)) == "neg"
    assert aucs["neg"] == 1.0  # a perfectly inverted feature is still perfect
    assert list(aucs.values()) == sorted(aucs.values(), reverse=True)


def test_chance_tolerance_shrinks_with_data() -> None:
    assert chance_tolerance(5000, 50000) == 0.05  # floor
    assert chance_tolerance(20, 200) > chance_tolerance(200, 2000)
