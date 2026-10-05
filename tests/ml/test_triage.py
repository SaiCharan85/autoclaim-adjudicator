import numpy as np
import pytest

from autoclaim.ml.triage import TwoStagePolicy, budget_for_recall, fit_policy, recall_curve

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
S1 = np.array([0.9, 0.2, 0.1, 0.05, 0.8, 0.3, 0.25, 0.15, 0.12, 0.01])
S2 = np.array([0.5, 0.9, 0.7, 0.2, 0.1, 0.3, 0.05, 0.6, 0.02, 0.01])
AMT = np.array([1000, 3000, 500, 500, 10, 10, 10, 10, 10, 10], dtype=float)


def test_recall_curve_and_dollars() -> None:
    curve = recall_curve(Y, S1, [0.1, 0.2], AMT)
    assert curve.loc[0.1, "recall"] == 0.25 and curve.loc[0.1, "precision"] == 1.0
    assert curve.loc[0.1, "dollar_recall"] == pytest.approx(1000 / 5000)
    assert curve.loc[0.2, "recall"] == 0.25 and curve.loc[0.2, "precision"] == 0.5


def test_budget_for_recall() -> None:
    perfect = Y.astype(float)
    assert budget_for_recall(Y, perfect, 0.8) == 0.4  # 4 frauds need 4 of 10 claims
    assert budget_for_recall(Y, perfect, 0.25) == 0.1


def test_policy_counts_each_referral_once() -> None:
    pol = TwoStagePolicy(stage1_budget=0.1, threshold=0.6)
    tier1, tier2 = pol.referrals(S1, S2)
    assert tier1.sum() == 1 and not (tier1 & tier2).any()
    m = pol.evaluate(Y, S1, S2, AMT)
    # tier1 = claim 0 (fraud); tier2 = claims 1, 2 (fraud) and 7 (legit)
    assert m["recall"] == 0.75 and m["review_rate"] == 0.4 and m["recall_stage1"] == 0.25
    assert m["precision"] == 0.75 and m["dollar_recall"] == pytest.approx(4500 / 5000)


def test_fit_policy_reaches_target_with_fewest_referrals() -> None:
    pol = fit_policy(Y, S1, S2, 0.1, target_recall=0.75)
    assert pol.threshold == 0.7  # claims 1 and 2 suffice; 0.6 would add a legit claim
    assert pol.evaluate(Y, S1, S2)["recall"] >= 0.75


def test_fit_policy_edges() -> None:
    assert fit_policy(Y, S1, S2, 1.0, 0.8).threshold == float("inf")  # stage 1 already enough
    hard = fit_policy(Y, S1, np.zeros(10), 0.1, 1.0)
    assert hard.evaluate(Y, S1, np.zeros(10))["recall"] == 1.0  # refer everyone


def test_bootstrap_policy_brackets_point_estimate() -> None:
    from autoclaim.ml.triage import bootstrap_policy

    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.1
    s1, s2 = y * 0.3 + rng.random(2000) * 0.7, y * 0.5 + rng.random(2000) * 0.6
    pol = TwoStagePolicy(0.05, 0.6)
    point = pol.evaluate(y, s1, s2)
    ci = bootstrap_policy(pol, y, s1, s2, n_boot=200, seed=1)
    lo, hi = ci["recall"]
    assert lo <= point["recall"] <= hi and hi - lo < 0.2
    assert ci["stage1_review_rate"] == pytest.approx((0.05, 0.05), abs=1e-3)
