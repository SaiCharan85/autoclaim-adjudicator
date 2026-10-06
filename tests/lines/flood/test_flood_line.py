"""The flood line plugged into the unchanged core graph (no LLM)."""

import pandas as pd
from test_flood_claim import ROW

from autoclaim.lines.flood import evaluate as fe
from autoclaim.lines.flood.build import build_flood
from autoclaim.lines.flood.claim import from_nfip
from autoclaim.lines.flood.line import FloodLine, template_explanation
from autoclaim.lines.flood.policy import POLICY_PATH
from autoclaim.retrieval.corpus import load_policy


def run(**over: object) -> tuple[dict, pd.Series]:
    row = pd.Series({**ROW, **over})
    claim = from_nfip(row)
    _, graph = build_flood()
    out = graph.invoke({"claim_id": claim.claim_id, "claim": claim.model_dump(mode="json")},
                       {"configurable": {"thread_id": claim.claim_id}})  # fmt: skip
    return out, row


def test_clear_claim_is_auto_approved_by_the_shared_graph_with_no_llm_calls() -> None:
    out, row = run()
    assert out["final"]["decided_by"] == "auto" and out["final"]["payout"] == 4948.0
    assert out["llm_calls"] == 0
    nodes = [e["node"] for e in out["audit"]]
    assert nodes[:3] == ["guardrails", "intake", "coverage"] and "auto_decide" in nodes
    s = fe.score(out, row, False)
    assert s.auto and s.agrees  # FEMA paid this claim


def test_over_authority_goes_to_a_human() -> None:
    out, _ = run(buildingDamageAmount=90000.0)
    req = out["__interrupt__"][0].value
    assert "over_authority_limit" in req["route_reasons"]
    assert req["proposed_decision"]["payout"] == 88000.0


def test_judgment_case_goes_to_a_human() -> None:
    out, _ = run(causeOfDamage="0")
    assert "adjudicator_escalated" in out["__interrupt__"][0].value["route_reasons"]


def test_invalid_record_is_rejected_by_guardrails() -> None:
    _, graph = build_flood()
    out = graph.invoke({"claim_id": "bad", "claim": {"claim_id": "bad"}},
                       {"configurable": {"thread_id": "bad"}})  # fmt: skip
    assert out["final"]["outcome"] == "rejected_input"


def test_template_explanation_states_the_code_numbers() -> None:
    from autoclaim.lines.flood.claim import assess, decide

    c = from_nfip(pd.Series(ROW))
    cov = assess(c)
    outcome, _, payout = decide(cov)
    text = template_explanation(c, cov, outcome, payout)
    assert "$6,948.00" in text and "$2,000.00" in text and "$4,948.00" in text
    assert "[FLD-DEDUCTIBLE]" in text and "[FLD-LIMITS]" in text


def test_critic_catches_a_payout_that_is_not_the_code_number() -> None:
    line = FloodLine(load_policy(POLICY_PATH))
    out, _ = run()
    state = {**out, "decision": {**out["decision"], "payout": 9999.0}}
    res = line.critic(state).update["critic"]
    assert not res["passed"] and res["issues"][0]["code"] == "payout_not_code_number"


def test_memory_text_and_no_fraud_model() -> None:
    line = FloodLine(load_policy(POLICY_PATH))
    out, _ = run()
    assert line.fraud_score(out) is None
    desc = line.memory_text(out)
    assert "zone AE" in desc.text and desc.when == pd.Timestamp("2023-09-23").date()


def test_core_has_no_flood_specific_code() -> None:
    from pathlib import Path

    core = Path(__file__).resolve().parents[3] / "src" / "autoclaim" / "core"
    hits = [p.name for p in core.glob("*.py") if "flood" in p.read_text(encoding="utf-8").lower()]
    assert hits == [] or all(h == "lob.py" for h in hits)  # lob.py only names it in a comment
