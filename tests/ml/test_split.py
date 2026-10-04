import pandas as pd
import pytest

from autoclaim.ml.features import TARGET, time_order
from autoclaim.ml.split import latest_slice, stratified_folds, stratified_holdout, time_split


def test_time_split_partitions_by_year(fraud_frame: pd.DataFrame) -> None:
    train, test = time_split(fraud_frame, [1994, 1995], [1996])
    assert set(train["Year"]) == {1994, 1995}
    assert set(test["Year"]) == {1996}
    assert len(train) + len(test) == len(fraud_frame)


def test_time_split_rejects_overlap(fraud_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="overlap"):
        time_split(fraud_frame, [1994, 1995], [1995])


def test_time_split_rejects_empty_side(fraud_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        time_split(fraud_frame, [1994], [2001])


def test_latest_slice_validation_is_strictly_later(fraud_frame: pd.DataFrame) -> None:
    train, val = latest_slice(fraud_frame, 0.2)
    assert time_order(val).min() > time_order(train).max()
    assert 0.1 < len(val) / len(fraud_frame) < 0.3


def test_latest_slice_rejects_degenerate_input() -> None:
    df = pd.DataFrame({"Year": [1994] * 4, "Month": ["Jan"] * 4, "WeekOfMonth": [1] * 4})
    with pytest.raises(ValueError, match="empty"):
        latest_slice(df, 0.2)


def test_stratified_folds_cover_each_row_once(fraud_frame: pd.DataFrame) -> None:
    seen = []
    for train, test in stratified_folds(fraud_frame, 3, seed=0):
        assert set(train.index).isdisjoint(test.index)
        seen.extend(test.index)
    assert sorted(seen) == sorted(fraud_frame.index)


def test_stratified_holdout_keeps_fraud_rate(fraud_frame: pd.DataFrame) -> None:
    _, val = stratified_holdout(fraud_frame, 0.25, seed=0)
    assert len(val) == pytest.approx(0.25 * len(fraud_frame), abs=1)
    assert val[TARGET].mean() == pytest.approx(fraud_frame[TARGET].mean(), abs=0.02)
