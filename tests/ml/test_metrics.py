import math

import numpy as np
import pytest

from autoclaim.ml.metrics import (
    bootstrap_ci,
    net_savings_per_1k,
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


def test_net_savings_per_1k() -> None:
    # top 30% = 3 reviewed claims; 2 are fraud worth 1000 + 3000; reviews cost 3 x 100
    amount = np.array([1000, 50, 3000, 0, 0, 0, 0, 0, 0, 9999])
    assert net_savings_per_1k(Y, S, amount, 0.3, 100) == pytest.approx((4000 - 300) / 10 * 1000)


def test_net_savings_can_be_negative() -> None:
    y = np.zeros(10)
    assert net_savings_per_1k(y, np.arange(10), np.ones(10), 0.5, 100) < 0


def test_class_weights_restore_population_rate() -> None:
    from autoclaim.ml.metrics import class_weights

    y = np.array([1] * 25 + [0] * 75)
    w = class_weights(y, 0.05)
    assert np.isclose((w * y).sum() / w.sum(), 0.05)
    with pytest.raises(ValueError):
        class_weights(np.zeros(5), 0.05)


def test_weighted_recall_matches_unweighted_with_unit_weights() -> None:
    from autoclaim.ml.metrics import recall_at_budget, weighted_recall_at_budget

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    s = y + rng.normal(0, 1, 400)
    assert weighted_recall_at_budget(y, s, np.ones(400), 0.1) == pytest.approx(
        recall_at_budget(y, s, 0.1)
    )
    with pytest.raises(ValueError):
        weighted_recall_at_budget(y, s, np.ones(400), 0)


def test_weighted_report_perfect_and_upweighted_legit() -> None:
    from autoclaim.ml.metrics import class_weights, weighted_report

    y = np.array([1] * 20 + [0] * 80)
    perfect = weighted_report(y, y.astype(float), class_weights(y, 0.05), 0.05)
    assert perfect["roc_auc"] == 1 and perfect["pr_auc"] == pytest.approx(1)
    assert perfect["recall_at_budget"] == pytest.approx(1, abs=0.01)
    # random scores: weighted recall@5% should sit near 5%, not near the enriched sample's rate
    rng = np.random.default_rng(1)
    y = (np.arange(4000) < 1000).astype(int)
    rand = weighted_report(y, rng.random(4000), class_weights(y, 0.05), 0.05)
    assert rand["recall_at_budget"] == pytest.approx(0.05, abs=0.03)
