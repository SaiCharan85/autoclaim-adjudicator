import math

import numpy as np
import pytest

from autoclaim.ml.metrics import (
    bootstrap_ci,
    paired_bootstrap_diff,
    precision_at_budget,
    recall_at_budget,
    score_report,
)

Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
S = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])


def test_recall_at_budget() -> None:
    # top 30% = 3 claims (0.9, 0.8, 0.7) contain 2 of 3 frauds
    assert recall_at_budget(Y, S, 0.3) == pytest.approx(2 / 3)


def test_precision_at_budget() -> None:
    assert precision_at_budget(Y, S, 0.3) == pytest.approx(2 / 3)


def test_budget_rounds_up_to_at_least_one_claim() -> None:
    assert recall_at_budget(Y, S, 0.01) == pytest.approx(1 / 3)  # k = 1


def test_full_budget_catches_everything() -> None:
    assert recall_at_budget(Y, S, 1.0) == 1.0


@pytest.mark.parametrize("budget", [0, -0.1, 1.5])
def test_invalid_budget(budget: float) -> None:
    with pytest.raises(ValueError, match="budget"):
        recall_at_budget(Y, S, budget)


def test_no_positives_gives_nan() -> None:
    assert math.isnan(recall_at_budget(np.zeros(4), np.arange(4), 0.5))


def test_ties_are_broken_stably_by_input_order() -> None:
    y = np.array([0, 1])
    s = np.array([0.5, 0.5])
    assert recall_at_budget(y, s, 0.5) == 0.0  # first of the tied rows is taken


def test_score_report_perfect_ranking() -> None:
    y = np.array([1, 1, 0, 0])
    report = score_report(y, np.array([0.9, 0.8, 0.2, 0.1]), 0.5)
    assert report["pr_auc"] == 1.0
    assert report["roc_auc"] == 1.0
    assert report["recall_at_budget"] == 1.0
    assert report["base_rate"] == 0.5


def _rng_data(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    good = y + rng.normal(0, 0.5, n)  # informative scores
    noise = rng.random(n)  # uninformative scores
    return y, good, noise


def _recall10(y: np.ndarray, s: np.ndarray) -> float:
    return recall_at_budget(y, s, 0.1)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    y, good, _ = _rng_data()
    lo, hi = bootstrap_ci(y, good, _recall10, n_boot=300, seed=1)
    assert lo <= _recall10(y, good) <= hi
    assert 0 < hi - lo < 0.3


def test_bootstrap_ci_is_seeded() -> None:
    y, good, _ = _rng_data()
    assert bootstrap_ci(y, good, _recall10, n_boot=100, seed=3) == bootstrap_ci(
        y, good, _recall10, n_boot=100, seed=3
    )


def test_paired_diff_detects_real_gap() -> None:
    y, good, noise = _rng_data()
    d, lo, _ = paired_bootstrap_diff(y, good, noise, _recall10, n_boot=300)
    assert d > 0
    assert lo > 0  # CI excludes zero: "good" is genuinely better


def test_paired_diff_of_identical_scores_is_zero() -> None:
    y, good, _ = _rng_data()
    assert paired_bootstrap_diff(y, good, good, _recall10, n_boot=50) == (0.0, 0.0, 0.0)
