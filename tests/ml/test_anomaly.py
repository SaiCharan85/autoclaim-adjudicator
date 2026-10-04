import numpy as np
import pandas as pd
import pytest

from autoclaim.ml.anomaly import AnomalyScorer, FrequencyEncoder
from autoclaim.ml.frame import feature_list

SKEWED = ["Make", "VehiclePrice", "AgeOfVehicle", "PastNumberOfClaims", "NumberOfCars"]


def _fit(df: pd.DataFrame, seed: int = 0) -> AnomalyScorer:
    return AnomalyScorer(feature_list(df), seed, n_estimators=200).fit(df)


@pytest.fixture
def skewed(fraud_std: pd.DataFrame) -> pd.DataFrame:
    """Real claims are skewed (e.g. 96% share one deductible); make a few columns 95/5."""
    df = fraud_std.copy()
    rng = np.random.default_rng(0)
    for col in SKEWED:
        df[col] = np.where(rng.random(len(df)) < 0.95, "common", "uncommon")
    return df


def test_frequency_encoder_maps_to_training_frequency() -> None:
    enc = FrequencyEncoder().fit(pd.DataFrame({"c": ["a", "a", "b", None]}))
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "never_seen", None]}))
    assert out[:, 0].tolist() == [0.5, 0.25, 0.0, 0.25]  # unseen -> 0, missing is its own value


def test_frequency_encoder_handles_no_columns() -> None:
    enc = FrequencyEncoder().fit(pd.DataFrame(index=range(3)))
    assert enc.transform(pd.DataFrame(index=range(3))).shape == (3, 0)


def test_scores_are_percentiles(fraud_std: pd.DataFrame) -> None:
    scores = _fit(fraud_std).score(fraud_std)
    assert scores.shape == (len(fraud_std),)
    assert scores.min() >= 0
    assert scores.max() <= 1
    assert 0.4 < np.median(scores) < 0.6


def test_categorical_types_are_learned_at_fit(fraud_std: pd.DataFrame) -> None:
    scorer = _fit(fraud_std)
    assert "Make" in scorer.cats
    assert "Age" not in scorer.cats
    runtime = fraud_std.head(3).copy()
    runtime["Age"] = np.nan  # an all-missing numeric column must not flip to categorical
    assert np.isfinite(scorer.score(runtime)).all()


def test_unseen_categories_look_anomalous(skewed: pd.DataFrame) -> None:
    scorer = _fit(skewed)
    typical = skewed[skewed[SKEWED].eq("common").all(axis=1)].head(50)
    outlier = typical.head(1).copy()
    outlier[SKEWED] = "never_seen"
    assert scorer.score(outlier)[0] > 0.9
    assert np.median(scorer.score(typical)) < 0.6


def test_rare_seen_values_rank_above_common_ones(skewed: pd.DataFrame) -> None:
    scorer = _fit(skewed)
    common = skewed[skewed[SKEWED].eq("common").all(axis=1)]
    rare = skewed[skewed[SKEWED].eq("uncommon").sum(axis=1) >= 2]
    assert np.median(scorer.score(rare)) > np.median(scorer.score(common))


def test_deterministic_given_seed(fraud_std: pd.DataFrame) -> None:
    a, b = _fit(fraud_std, 5), _fit(fraud_std, 5)
    np.testing.assert_array_equal(a.score(fraud_std.head(20)), b.score(fraud_std.head(20)))
