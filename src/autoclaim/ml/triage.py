"""Fraud triage: how much fraud gets intercepted for how much review work.

Two-stage policy (what insurers do):
1. Stage 1 at first notice: the riskiest `stage1_budget` share of claims goes straight to the
   fraud team (SIU).
2. Stage 2 after the independent appraisal: every other claim is re-scored with the appraisal
   (claimed vs appraised amount, prior-damage note); claims scoring >= `threshold` are referred.

Interception recall = share of fraud referred by either stage. The threshold is fitted on
validation for a target recall at the smallest review rate, then frozen for the locked test.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _top_mask(scores: np.ndarray, share: float) -> np.ndarray:
    k = int(np.ceil(share * len(scores)))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def recall_curve(
    y: ArrayLike, scores: ArrayLike, budgets: Sequence[float], amounts: ArrayLike | None = None
) -> pd.DataFrame:
    """Recall, precision and (optionally) dollar recall when reviewing the top `budget` share."""
    yy, s = np.asarray(y, dtype=bool), np.asarray(scores, dtype=float)
    a = None if amounts is None else np.asarray(amounts, dtype=float)
    rows = []
    for b in budgets:
        top = _top_mask(s, b)
        row = {"budget": b, "recall": yy[top].sum() / yy.sum(), "precision": yy[top].mean()}
        if a is not None:
            row["dollar_recall"] = a[top & yy].sum() / a[yy].sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("budget")


def budget_for_recall(y: ArrayLike, scores: ArrayLike, target: float) -> float:
    """Smallest review share (top by score) that reaches `target` recall."""
    yy, s = np.asarray(y, dtype=bool), np.asarray(scores, dtype=float)
    cum = np.cumsum(yy[np.argsort(-s, kind="stable")]) / yy.sum()
    return float((np.argmax(cum >= target - 1e-12) + 1) / len(yy))


@dataclass(frozen=True)
class TwoStagePolicy:
    stage1_budget: float
    threshold: float  # stage-2 score at or above which a claim is referred

    def referrals(self, s1: ArrayLike, s2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        tier1 = _top_mask(np.asarray(s1, dtype=float), self.stage1_budget)
        tier2 = ~tier1 & (np.asarray(s2, dtype=float) >= self.threshold)
        return tier1, tier2

    def evaluate(
        self, y: ArrayLike, s1: ArrayLike, s2: ArrayLike, amounts: ArrayLike | None = None
    ) -> dict[str, float]:
        yy = np.asarray(y, dtype=bool)
        tier1, tier2 = self.referrals(s1, s2)
        referred = tier1 | tier2
        out = {
            "stage1_review_rate": float(tier1.mean()),
            "stage2_review_rate": float(tier2.mean()),
            "review_rate": float(referred.mean()),
            "recall": float(yy[referred].sum() / yy.sum()),
            "recall_stage1": float(yy[tier1].sum() / yy.sum()),
            "precision": float(yy[referred].mean()) if referred.any() else float("nan"),
        }
        if amounts is not None:
            a = np.asarray(amounts, dtype=float)
            out["dollar_recall"] = float(a[referred & yy].sum() / a[yy].sum())
        return out


def fit_policy(
    y: ArrayLike, s1: ArrayLike, s2: ArrayLike, stage1_budget: float, target_recall: float
) -> TwoStagePolicy:
    """Stage-2 threshold reaching `target_recall` with the fewest referrals (fit on validation)."""
    yy, b = np.asarray(y, dtype=bool), np.asarray(s2, dtype=float)
    tier1 = _top_mask(np.asarray(s1, dtype=float), stage1_budget)
    need = target_recall * yy.sum() - yy[tier1].sum()
    if need <= 0:
        return TwoStagePolicy(stage1_budget, float("inf"))
    rest = np.flatnonzero(~tier1)
    order = rest[np.argsort(-b[rest], kind="stable")]
    cum = np.cumsum(yy[order])
    if cum[-1] < need:  # even referring everyone is not enough
        return TwoStagePolicy(stage1_budget, float(b[order[-1]]))
    k = int(np.argmax(cum >= need - 1e-12))
    return TwoStagePolicy(stage1_budget, float(b[order[k]]))


def bootstrap_policy(
    policy: TwoStagePolicy,
    y: ArrayLike,
    s1: ArrayLike,
    s2: ArrayLike,
    amounts: ArrayLike | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile 95% CI of every policy metric over resampled claims (stage 1's top share is
    re-ranked inside each resample, as it would be on a different month of claims)."""
    yy, a1, a2 = np.asarray(y, dtype=bool), np.asarray(s1, float), np.asarray(s2, float)
    am = None if amounts is None else np.asarray(amounts, float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        i = rng.integers(0, len(yy), len(yy))
        if not yy[i].any():
            continue
        m = policy.evaluate(yy[i], a1[i], a2[i], None if am is None else am[i])
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    return {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))
            for k, v in draws.items()}  # fmt: skip
