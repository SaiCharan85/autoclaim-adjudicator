import pandas as pd
import pytest

from autoclaim.lines.flood import evaluate as fe


def s(auto: bool, outcome: str, payout: float | None, paid: float, rc: bool | None = False,
      reasons: tuple[str, ...] = ()) -> fe.FloodScore:  # fmt: skip
    return fe.FloodScore(claim_id="x", auto=auto, outcome=outcome, payout=payout,
                         actual_paid=paid, replacement_cost=rc, route_reasons=reasons)  # fmt: skip


SCORES = [
    s(True, "approve", 1000.0, 1000.0),  # exact
    s(True, "approve", 1000.0, 1200.0, rc=True),  # RC: ACV first
    s(True, "deny", None, 0.0),  # right denial
    s(True, "deny", None, 500.0),  # wrongly denied
    s(True, "approve", 800.0, 0.0),  # wrongly paid
    s(False, "human", 70000.0, 70000.0, reasons=("over_authority_limit",)),
]


def test_metrics() -> None:
    assert fe.metric(SCORES, "auto_rate") == pytest.approx(5 / 6)
    assert fe.metric(SCORES, "auto_agreement") == pytest.approx(3 / 5)
    assert fe.metric(SCORES, "wrongly_denied_rate") == pytest.approx(1 / 6)
    assert fe.metric(SCORES, "wrongly_paid_rate") == pytest.approx(1 / 6)
    assert fe.metric(SCORES, "acv_within_5pct") == pytest.approx(1 / 2)  # exact one, paid-0 one
    assert fe.metric(SCORES, "rc_first_payment_ratio") == pytest.approx(1000 / 1200)


def test_metric_edge_cases() -> None:
    assert fe.metric([], "auto_rate") is None
    with pytest.raises(ValueError):
        fe.metric(SCORES, "vibes")


def test_route_counts_and_summary() -> None:
    assert fe.route_counts(SCORES) == {"over_authority_limit": 1}
    est = fe.summarize(SCORES * 4, n_boot=50)
    assert set(est) == set(fe.HEADLINE) and est["auto_rate"].value == pytest.approx(5 / 6)


def test_actual_paid_sums_building_and_contents() -> None:
    row = pd.Series({"amountPaidOnBuildingClaim": 100.0, "amountPaidOnContentsClaim": float("nan")})
    assert fe.actual_paid(row) == 100.0
