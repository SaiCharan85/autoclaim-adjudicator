"""Train/validation/test splits.

Time split: train on earlier years, test on later ones (how the model is used in production).
Early-stopping validation is the *latest* slice of the training period, never the test years.
"""

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from autoclaim.ml.features import TARGET, time_order


def time_split(
    df: pd.DataFrame, train_years: Sequence[int], test_years: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(train_years) & set(test_years):
        raise ValueError("train and test years overlap")
    train = df[df["Year"].isin(train_years)]
    test = df[df["Year"].isin(test_years)]
    if train.empty or test.empty:
        raise ValueError("time split produced an empty train or test set")
    return train, test


def latest_slice(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the most recent `fraction` of rows (by accident time) as validation."""
    order = time_order(df).to_numpy()
    cutoff = np.quantile(order, 1 - fraction)
    is_val = order > cutoff
    if not is_val.any() or is_val.all():
        raise ValueError("latest_slice produced an empty side")
    return df[~is_val], df[is_val]


def stratified_folds(
    df: pd.DataFrame, n_splits: int, seed: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(df, df[TARGET]):
        yield df.iloc[tr], df.iloc[te]


def stratified_holdout(
    df: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, val = train_test_split(df, test_size=fraction, stratify=df[TARGET], random_state=seed)
    return tr, val
