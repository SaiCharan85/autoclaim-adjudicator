import numpy as np
import pandas as pd
import pytest

from autoclaim.ml.frame import T, Y
from autoclaim.ml.split import (
    assert_time_ordered,
    latest_slice,
    stratified_folds,
    stratified_holdout,
)


def test_latest_slice_is_strictly_later(fraud_std: pd.DataFrame) -> None:
    train, val = latest_slice(fraud_std, 0.2)
    assert val[T].min() > train[T].max()
    assert 0.1 < len(val) / len(fraud_std) < 0.3


def test_latest_slice_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        latest_slice(pd.DataFrame({T: [1, 1, 1, 1], Y: [0, 1, 0, 1]}), 0.2)


def test_stratified_folds_cover_each_row_once(fraud_std: pd.DataFrame) -> None:
    seen = []
    for train, test in stratified_folds(fraud_std, 3, seed=0):
        assert set(train.index).isdisjoint(test.index)
        seen.extend(test.index)
    assert sorted(seen) == sorted(fraud_std.index)


def test_stratified_holdout_keeps_fraud_rate(fraud_std: pd.DataFrame) -> None:
    _, val = stratified_holdout(fraud_std, 0.25, seed=0)
    assert len(val) == pytest.approx(0.25 * len(fraud_std), abs=1)
    assert val[Y].mean() == pytest.approx(fraud_std[Y].mean(), abs=0.02)


def test_assert_time_ordered() -> None:
    f = pd.DataFrame({T: [1, 2, 3, 4, 5, 6]})
    assert_time_ordered(f.iloc[:2], f.iloc[2:4], f.iloc[4:])
    with pytest.raises(ValueError, match="train overlaps validation"):
        assert_time_ordered(f.iloc[:3], f.iloc[2:4], f.iloc[4:])
    with pytest.raises(ValueError, match="validation overlaps test"):
        assert_time_ordered(f.iloc[:2], f.iloc[2:5], f.iloc[4:])


def test_rolling_origin_never_looks_ahead() -> None:
    import pandas as pd

    from autoclaim.ml.frame import T
    from autoclaim.ml.split import rolling_origin

    frame = pd.DataFrame({T: np.arange(100), Y: np.arange(100) % 2})
    folds = list(rolling_origin(frame, [50, 70], horizon=10, calib=10))
    assert len(folds) == 2
    for start, (fit, cal, test) in zip([50, 70], folds, strict=True):
        assert fit[T].max() < cal[T].min() and cal[T].max() < test[T].min()
        assert cal[T].min() == start - 10 and test[T].min() == start
        assert test[T].max() == start + 9
    with pytest.raises(ValueError, match="empty"):
        list(rolling_origin(frame, [5], horizon=10, calib=10))


def test_recency_weights() -> None:
    from autoclaim.ml.split import recency_weights

    t = np.array([0.0, 365.0, 730.0])
    w = recency_weights(t, half_life=365)
    assert w.mean() == pytest.approx(1) and w[2] == pytest.approx(2 * w[1]) == pytest.approx(
        4 * w[0]
    )
    assert (recency_weights(t, None) == 1).all()
