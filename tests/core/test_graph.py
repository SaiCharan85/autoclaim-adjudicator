from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from autoclaim.core.audit import AuditSink
from autoclaim.core.budget import BudgetConfig
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.graph import Harness, HarnessConfig
from autoclaim.core.lob import NodeResult
from autoclaim.core.router import RouterConfig
from autoclaim.llm.client import CallMeta

DECISION = {
    "outcome": "approve",
    "payout": 900.0,
    "reasons": [],
    "cited_clauses": ["INS-COLLISION"],
    "explanation": "Covered collision.",
    "confidence": 0.9,
    "self_check": {},
}


def _call(tokens: int = 100) -> CallMeta:
    return CallMeta(role="x", model="m", prompt_tokens=tokens, completion_tokens=10)


class FakeLine:
    name = "fake"

    def __init__(self, critic_passes: list[bool] | None = None, **kw: Any) -> None:
        self.critic_passes = list(critic_passes or [True])
        self.kw = kw
        self.adjudications = 0
        self.seen_issues: list[list] = []

    def guardrails(self, state):
        return NodeResult({"guardrails": {"rejected": self.kw.get("reject", False),
                                          "errors": ["bad"]}})  # fmt: skip

    def intake(self, state):
        return NodeResult({"facts": {"cause": "collision"}}, [_call()])

    def coverage(self, state):
        return NodeResult({"coverage": {"covered": True}}, [_call()])

    def fraud(self, state):
        if self.kw.get("fraud_raises"):
            raise RuntimeError("model file missing")
        return NodeResult({"fraud": {"score": self.kw.get("fraud_score", 0.05)}})

    def adjudicate(self, state):
        self.adjudications += 1
        self.seen_issues.append(state.get("issues", []))
        return NodeResult({"decision": DECISION | self.kw.get("decision", {})},
                          [_call(self.kw.get("adj_tokens", 100))])  # fmt: skip

    def critic(self, state):
        ok = self.critic_passes.pop(0) if self.critic_passes else False
        issues = [] if ok else [{"source": "critic", "code": "no_clause", "detail": "x"}]
        return NodeResult({"critic": {"passed": ok, "issues": issues}})

    def judge(self, state):
        return NodeResult({"judge": {"passed": True, "issues": []}}, [_call()])

    def hard_escalations(self, state):
        return self.kw.get("hard", [])

    def fraud_score(self, state):
        return state.get("fraud", {}).get("score")

    def case_summary(self, state):
        return {"facts": state.get("facts")}


class Memory:
    def __init__(self) -> None:
        self.states: list = []

    def remember(self, state) -> None:
        self.states.append(state)


CFG = HarnessConfig(
    max_retries=2,
    router=RouterConfig(min_confidence=0.7, fraud_review_score=0.5, authority_limit_usd=10_000),
    budget=BudgetConfig(max_llm_calls=20, max_tokens=5_000),
)


def _run(line: FakeLine, tmp_path, claim_id: str = "C1", memory: Memory | None = None):
    harness = Harness(line, CFG, AuditSink(tmp_path / "audit"), FinalizationLedger(":memory:"),
                      memory)  # fmt: skip
    graph = harness.build(InMemorySaver())
    config = {"configurable": {"thread_id": claim_id}}
    out = graph.invoke({"claim_id": claim_id, "claim": {}}, config)
    return harness, graph, config, out


def test_happy_path_auto_decides_with_audit(tmp_path) -> None:
    harness, _, _, out = _run(FakeLine(), tmp_path)
    assert out["final"]["decided_by"] == "auto" and out["final"]["payout"] == 900.0
    assert not out["final"]["duplicate"]
    nodes = [e["node"] for e in out["audit"]]
    assert nodes[0] == "guardrails" and nodes[-1] == "auto_decide"
    assert {"coverage", "fraud", "critic", "judge", "router"} <= set(nodes)
    assert out["llm_calls"] == 4 and out["tokens"] == 4 * 110
    on_disk = harness.audit.read("C1")
    # append-only file mirrors the state; parallel coverage/fraud may hit the disk in either order
    assert sorted(e.node for e in on_disk) == sorted(nodes)
    assert on_disk[0].node == "guardrails" and on_disk[-1].node == "auto_decide"


def test_critic_failure_retries_with_issues(tmp_path) -> None:
    line = FakeLine(critic_passes=[False, True])
    _, _, _, out = _run(line, tmp_path)
    assert line.adjudications == 2 and out["retries"] == 1
    assert line.seen_issues[0] == [] and line.seen_issues[1][0]["code"] == "no_clause"
    assert out["final"]["decided_by"] == "auto"


def test_exhausted_retries_interrupt_then_resume_once(tmp_path) -> None:
    line, memory = FakeLine(critic_passes=[False, False, False]), Memory()
    harness, graph, config, out = _run(line, tmp_path, memory=memory)
    assert line.adjudications == 3  # first try + 2 retries
    assert "final" not in out and out["__interrupt__"]
    request = out["__interrupt__"][0].value
    assert "checks_failed_after_retries" in request["route_reasons"]
    assert request["proposed_decision"]["outcome"] == "approve"
    human = {"outcome": "deny", "payout": None, "reason": "excluded driver", "adjuster": "a1"}
    done = graph.invoke(Command(resume=human), config)
    assert done["final"]["decided_by"] == "human" and done["final"]["outcome"] == "deny"
    assert len(memory.states) == 1 and memory.states[0]["human"]["adjuster"] == "a1"
    assert harness.ledger.get("C1")["outcome"] == "deny"


def test_failsafe_skips_downstream_and_goes_to_human(tmp_path) -> None:
    *_, out = _run(FakeLine(fraud_raises=True), tmp_path)
    req = out["__interrupt__"][0].value
    assert any(r.startswith("failsafe:fraud: RuntimeError") for r in req["route_reasons"])
    statuses = {e["node"]: e["status"] for e in out["audit"]}
    assert statuses["fraud"] == "failsafe" and statuses["adjudicator"] == "skipped"
    assert "judge" not in statuses  # critic -> router directly: no LLM call wasted


def test_budget_trip_goes_to_human(tmp_path) -> None:
    _, _, _, out = _run(FakeLine(adj_tokens=10_000), tmp_path)
    assert any("budget" in r for r in out["__interrupt__"][0].value["route_reasons"])


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"hard": ["late_notice"]}, "hard_rule:late_notice"),
        ({"fraud_score": 0.8}, "approve_with_high_fraud_score"),
        ({"decision": {"payout": 50_000.0}}, "over_authority_limit"),
        ({"decision": {"confidence": 0.3}}, "low_confidence"),
        ({"decision": {"outcome": "escalate", "payout": None}}, "adjudicator_escalated"),
    ],
)
def test_router_escalations(tmp_path, kw, reason) -> None:
    _, _, _, out = _run(FakeLine(**kw), tmp_path)
    assert reason in out["__interrupt__"][0].value["route_reasons"]


def test_guardrails_reject_ends_early(tmp_path) -> None:
    _, _, _, out = _run(FakeLine(reject=True), tmp_path)
    assert out["final"]["outcome"] == "rejected_input"
    assert "intake" not in {e["node"] for e in out["audit"]}


def test_same_claim_never_finalized_twice(tmp_path) -> None:
    line = FakeLine()
    harness = Harness(line, CFG, AuditSink(None), FinalizationLedger(":memory:"))
    graph = harness.build(InMemorySaver())
    first = graph.invoke({"claim_id": "C9", "claim": {}}, {"configurable": {"thread_id": "t1"}})
    second = graph.invoke({"claim_id": "C9", "claim": {}}, {"configurable": {"thread_id": "t2"}})
    assert not first["final"]["duplicate"] and second["final"]["duplicate"]


def test_llm_fallback_errors_are_audited(tmp_path) -> None:
    class Flaky(FakeLine):
        def judge(self, state):
            meta = _call()
            meta.errors.append("g:m: rate limited")
            return NodeResult({"judge": {"passed": True, "issues": []}}, [meta])

    *_, out = _run(Flaky(), tmp_path)
    judge = next(e for e in out["audit"] if e["node"] == "judge")
    assert judge["outputs"]["llm_errors"] == ["g:m: rate limited"]


def test_failsafe_reason_keeps_every_model_in_a_long_fallback_chain(tmp_path) -> None:
    class ChainFails(FakeLine):
        def coverage(self, state):
            raise RuntimeError("every model failed: ['a: " + "x" * 500 + "', 'b: rate limited']")

    *_, out = _run(ChainFails(), tmp_path)
    assert out["failsafe"].endswith("'b: rate limited']")
