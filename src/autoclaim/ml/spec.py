"""A dataset spec: how one raw table becomes the standard modeling frame (see ml/frame.py)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

Frames = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _no_check(features: Sequence[str]) -> None:
    return None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    real_labels: bool  # True only for real fraud labels (the 1990s data)
    load_raw: Callable[[], pd.DataFrame]
    # raw -> first-notice-of-loss features only (works on unlabeled runtime rows too)
    features_frame: Callable[[pd.DataFrame], pd.DataFrame]
    # raw -> features + y + t (+ amount)
    prepare: Callable[[pd.DataFrame], pd.DataFrame]
    # raw -> columns named in the shared red-flag rule vocabulary
    canonical: Callable[[pd.DataFrame], pd.DataFrame]
    # standard frame -> (train, validation, test), strictly ordered in time
    split: Callable[[pd.DataFrame], Frames]
    sensitive: tuple[str, ...] = ()
    check_features: Callable[[Sequence[str]], None] = _no_check
    load_holdouts: Callable[[], dict[str, pd.DataFrame]] | None = field(default=None)
    # name -> factory(seed) for fitted feature stages this dataset can use (ml/stages.py)
    stage_factories: dict[str, Callable[[int], Any]] = field(default_factory=dict)
