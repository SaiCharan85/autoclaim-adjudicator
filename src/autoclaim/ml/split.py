"""Splits on the standard frame (ml/frame.py). Dataset-level train/val/test splits live in each
DatasetSpec; these helpers carve early-stopping slices and stratified folds out of a training set.
"""

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from autoclaim.ml.frame import T, Y


def latest_slice(frame: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the most recent `fraction` of rows (by time key) as an early-stopping slice."""
    cutoff = np.quantile(frame[T].to_numpy(), 1 - fraction)
    is_val = frame[T].to_numpy() > cutoff
    if not is_val.any() or is_val.all():
        raise ValueError("latest_slice produced an empty side")
    return frame[~is_val], frame[is_val]


def stratified_folds(
    frame: pd.DataFrame, n_splits: int, seed: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(frame, frame[Y]):
        yield frame.iloc[tr], frame.iloc[te]


def stratified_holdout(
    frame: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, val = train_test_split(frame, test_size=fraction, stratify=frame[Y], random_state=seed)
    return tr, val


def assert_time_ordered(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Leakage guard: every training row is strictly older than validation, and validation older
    than test. Raises if a split ever overlaps in time."""
    if len(train) and len(val) and train[T].max() >= val[T].min():
        raise ValueError("train overlaps validation in time")
    if len(val) and len(test) and val[T].max() >= test[T].min():
        raise ValueError("validation overlaps test in time")


def rolling_origin(
    frame: pd.DataFrame, fold_starts: Sequence[int], horizon: int, calib: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Time-series cross-validation that mimics deployment, one fold per start (time-key units):
    fit = everything before `start - calib`; calib = the `calib` units just before `start` (for
    thresholds / early decisions); test = [start, start + horizon). Never random, never future."""
    t = frame[T].to_numpy()
    for start in fold_starts:
        fit = frame[t < start - calib]
        cal = frame[(t >= start - calib) & (t < start)]
        test = frame[(t >= start) & (t < start + horizon)]
        if fit.empty or cal.empty or test.empty:
            raise ValueError(f"fold starting at {start} has an empty side")
        yield fit, cal, test


def recency_weights(t: np.ndarray, half_life: float | None) -> np.ndarray:
    """Training weights that halve every `half_life` time units back from the newest row
    (mean 1). None = all rows equal."""
    t = np.asarray(t, dtype=float)
    if half_life is None:
        return np.ones(len(t))
    w = 0.5 ** ((t.max() - t) / half_life)
    out: np.ndarray = w / w.mean()
    return out
