"""Ranking metrics for imbalanced fraud detection.

Accuracy is useless at a ~6% base rate. A fraud team can only review a fixed share of claims,
so we measure how many frauds land inside that review budget.
"""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score


def _top_k_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    k = max(1, math.ceil(budget * len(scores)))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def recall_at_budget(y_true: ArrayLike, scores: ArrayLike, budget: float) -> float:
    """Share of all frauds that fall in the top `budget` fraction of claims by score."""
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    positives = y.sum()
    if positives == 0:
        return float("nan")
    return float(y[_top_k_mask(s, budget)].sum() / positives)


def precision_at_budget(y_true: ArrayLike, scores: ArrayLike, budget: float) -> float:
    """Share of reviewed claims (top `budget` fraction) that are fraud."""
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    return float(y[_top_k_mask(s, budget)].mean())


def net_savings_per_1k(
    y_true: ArrayLike, scores: ArrayLike, amount: ArrayLike, budget: float, review_cost: float
) -> float:
    """Cost-sensitive view (cf. Yankol-Schalck 2025): dollars of fraudulent claims caught in the
    review budget minus the cost of the reviews, per 1,000 claims. Higher is better."""
    y, s, a = np.asarray(y_true), np.asarray(scores, dtype=float), np.asarray(amount, dtype=float)
    reviewed = _top_k_mask(s, budget)
    caught = float(np.nansum(a[reviewed & (y == 1)]))
    return float((caught - reviewed.sum() * review_cost) / len(y) * 1000)


Metric = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(
    y_true: ArrayLike,
    scores: ArrayLike,
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `metric` over resampled claims."""
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    stats = np.array([metric(y[i], s[i]) for i in idx if y[i].any()])
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    y_true: ArrayLike,
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """metric(a) - metric(b) on the same resampled claims: (point estimate, CI low, CI high).

    Pairing removes the shared "which claims landed in the test set" noise, so this is much
    tighter than comparing two separate CIs.
    """
    y = np.asarray(y_true)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = np.array([metric(y[i], a[i]) - metric(y[i], b[i]) for i in idx if y[i].any()])
    lo, hi = np.nanquantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, a) - metric(y, b)), float(lo), float(hi)


def score_report(y_true: ArrayLike, scores: ArrayLike, budget: float) -> dict[str, float]:
    y = np.asarray(y_true)
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_budget": recall_at_budget(y, scores, budget),
        "precision_at_budget": precision_at_budget(y, scores, budget),
        "base_rate": float(y.mean()),
    }
