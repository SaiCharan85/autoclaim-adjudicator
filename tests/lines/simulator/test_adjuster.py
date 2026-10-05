import pandas as pd
import pytest

from autoclaim.lines.auto.simulator.adjuster import OracleAdjuster

TRUTH = pd.DataFrame(
    {
        "claim_id": [f"C{i}" for i in range(200)],
        "gt_decision": ["approve", "deny", "escalate", "approve"] * 50,
        "gt_reasons": [float("nan"), "excluded_driver", "late_notice_prejudice_review",
                       float("nan")] * 50,
        "gt_payout": [1650.0, 0.0, 0.0, 900.5] * 50,
    }
)  # fmt: skip


def req(cid: str, proposed: dict | None = None) -> dict:
    return {"claim_id": cid, "route_reasons": ["low_confidence"], "proposed_decision": proposed}


def test_perfect_adjuster_returns_ground_truth() -> None:
    adj = OracleAdjuster(TRUTH, error_rate=0.0)
    approve = adj.decide(req("C0"))
    assert (approve.outcome, approve.payout, approve.reason) == (
        "approve",
        1650.0,
        "reasons: covered_loss",
    )
    deny = adj.decide(req("C1"))
    assert (deny.outcome, deny.payout, deny.reason) == ("deny", None, "reasons: excluded_driver")
    assert adj.decide(req("C2")).outcome == "escalate"
    assert approve.adjuster == "oracle" and not adj.errors


def test_noise_rate_and_recorded_mistakes() -> None:
    adj = OracleAdjuster(TRUTH, error_rate=0.2, seed=1)
    wrong = 0
    for cid, truth in zip(TRUTH["claim_id"], TRUTH["gt_decision"], strict=True):
        d = adj.decide(req(cid))
        wrong += d.outcome != truth
        assert (d.outcome != truth) == (cid in adj.errors)
    assert 25 <= wrong <= 55  # ~20% of 200
    assert all(adj.decide(req(c)).reason == "reasons: judgment_call" for c in adj.errors)


def test_seeded_per_claim() -> None:
    a = OracleAdjuster(TRUTH, error_rate=0.3, seed=5)
    b = OracleAdjuster(TRUTH, error_rate=0.3, seed=5)
    assert [a.decide(req(c)) for c in TRUTH["claim_id"][:50]] == [
        b.decide(req(c)) for c in TRUTH["claim_id"][:50]
    ]


def test_mistaken_approval_pays_the_proposal_or_truth() -> None:
    adj = OracleAdjuster(TRUTH, error_rate=1.0, seed=0)
    for cid in TRUTH["claim_id"][:40]:
        d = adj.decide(req(cid, {"outcome": "deny", "payout": 321.0}))
        if d.outcome == "approve":
            assert d.payout == 321.0
        else:
            assert d.payout is None


def test_unknown_claim_and_bad_rate() -> None:
    with pytest.raises(ValueError, match="no ground truth"):
        OracleAdjuster(TRUTH).decide(req("NOPE"))
    with pytest.raises(ValueError):
        OracleAdjuster(TRUTH, error_rate=1.5)
