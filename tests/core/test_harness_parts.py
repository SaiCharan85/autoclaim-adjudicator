import pytest

from autoclaim.core.audit import AuditEvent, AuditSink
from autoclaim.core.budget import BudgetConfig, BudgetTrippedError, check_budget
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.router import RouterConfig, route_claim
from autoclaim.core.state import keep_first

RCFG = RouterConfig(min_confidence=0.7, fraud_review_score=0.5, authority_limit_usd=10_000)
OK = {"outcome": "approve", "payout": 100.0, "confidence": 0.9}


def test_router_auto_when_clean() -> None:
    rd = route_claim(OK, [], 0.1, True, None, RCFG)
    assert rd.route == "auto_decide" and rd.reasons == []


def test_router_deny_with_high_fraud_is_not_escalated_for_fraud() -> None:
    deny = {"outcome": "deny", "payout": None, "confidence": 0.9}
    assert route_claim(deny, [], 0.9, True, None, RCFG).route == "auto_decide"


def test_router_collects_every_reason() -> None:
    rd = route_claim(None, ["late"], None, False, "intake: boom", RCFG)
    assert rd.route == "human_review"
    assert rd.reasons == ["failsafe:intake: boom", "no_decision", "checks_failed_after_retries",
                          "hard_rule:late"]  # fmt: skip


def test_budget() -> None:
    cfg = BudgetConfig(max_llm_calls=2, max_tokens=100)
    check_budget(2, 100, cfg)
    with pytest.raises(BudgetTrippedError, match="llm_calls"):
        check_budget(3, 0, cfg)
    with pytest.raises(BudgetTrippedError, match="tokens"):
        check_budget(0, 101, cfg)


def test_ledger_first_write_wins(tmp_path) -> None:
    ledger = FinalizationLedger(tmp_path / "f.sqlite3")
    stored, created = ledger.finalize("c", {"outcome": "approve", "payout": 10})
    again, created2 = FinalizationLedger(tmp_path / "f.sqlite3").finalize("c", {"outcome": "deny"})
    assert created and not created2
    assert stored == again == {"outcome": "approve", "payout": 10}
    assert ledger.get("other") is None


def test_audit_sink_appends(tmp_path) -> None:
    sink = AuditSink(tmp_path)
    sink.write(AuditEvent(claim_id="c", node="a"))
    sink.write(AuditEvent(claim_id="c", node="b", prompt_tokens=5))
    events = sink.read("c")
    assert [e.node for e in events] == ["a", "b"] and events[1].prompt_tokens == 5
    assert events[0].cost_usd == 0.0
    assert sink.read("missing") == [] and AuditSink(None).read("c") == []


def test_keep_first() -> None:
    assert keep_first(None, "b") == "b" and keep_first("a", "b") == "a"
    assert keep_first(None, None) is None
