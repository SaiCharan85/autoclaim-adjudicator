import json
from pathlib import Path

import pytest

from autoclaim.ui.review_view import (
    FormError,
    audit_rows,
    case_markdown,
    decision_from_form,
    queue_rows,
)

REQ = {
    "claim_id": "CLM-1",
    "route_reasons": ["hard_rule:late_notice", "low_confidence"],
    "proposed_decision": {"outcome": "approve", "payout": 1650.0, "reasons": ["covered_loss"],
                          "cited_clauses": ["INS-COMPREHENSIVE", "LIM-DEDUCTIBLE"],
                          "explanation": "Deer strike is comprehensive.", "confidence": 0.62},
    "case_summary": {
        "estimate": 1900.0, "statement": "I hit a deer.",
        "facts": {"derived": {"coverage_part": "comprehensive", "part_on_policy": True,
                              "deductible": 250.0, "payout": 1650.0, "driver_role": "named_insured",
                              "notice_days": 41, "late_notice": True}},
        "fraud": {"score": 0.31, "red_flags": ["early_inception"]},
        "judge": {"passed": False, "issues": [{"code": "facts_supported", "detail": "date"}]},
    },
}  # fmt: skip


def test_queue_rows() -> None:
    (row,) = queue_rows([("CLM-1", REQ)])
    assert row == {
        "claim": "CLM-1",
        "why escalated": "hard_rule:late_notice, low_confidence",
        "proposed": "approve",
        "proposed payout": 1650.0,
        "confidence": 0.62,
    }


def test_queue_row_for_a_failsafe_claim_without_proposal() -> None:
    (row,) = queue_rows(
        [("C2", {"claim_id": "C2", "route_reasons": [], "proposed_decision": None})]
    )
    assert row["proposed"] == "none (failsafe)" and row["why escalated"] == "-"


def test_case_markdown_shows_what_an_adjuster_needs() -> None:
    md = case_markdown(REQ)
    for part in ("### CLM-1", "hard_rule:late_notice", "approve $1,650.00", "INS-COMPREHENSIVE",
                 "Deer strike is comprehensive.", "Repair estimate:** $1,900.00",
                 "comprehensive coverage on policy", "deductible $250.00", "notice 41 days (late)",
                 "score 0.31; red flags: early_inception", "facts_supported: date",
                 "> I hit a deer."):  # fmt: skip
        assert part in md, part


def test_case_markdown_minimal_and_failsafe() -> None:
    md = case_markdown({"claim_id": "C3", "case_summary": {"failsafe": "coverage: down"}})
    assert "### C3" in md and "**Failsafe:** coverage: down" in md
    assert "Harness proposal" not in md


# ---------------------------------------------------------------- the adjuster's form


def test_valid_decisions() -> None:
    d = decision_from_form("approve", 1500.0, "  late but no prejudice ", " Ana ")
    assert (d.outcome, d.payout, d.reason, d.adjuster) == (
        "approve",
        1500.0,
        "late but no prejudice",
        "Ana",
    )
    assert decision_from_form("deny", 900.0, "excluded driver", "Ana").payout is None


@pytest.mark.parametrize(
    ("args", "msg"),
    [(("pay", 1.0, "r", "a"), "outcome"), (("deny", None, "  ", "a"), "reason"),
     (("deny", None, "r", ""), "name"), (("approve", None, "r", "a"), "payout"),
     (("approve", -5.0, "r", "a"), "greater")],
)  # fmt: skip
def test_invalid_forms_never_resume(args, msg) -> None:
    with pytest.raises(FormError, match=msg):
        decision_from_form(*args)


# ---------------------------------------------------------------- audit trail


def test_audit_rows(tmp_path: Path) -> None:
    events = [{"claim_id": "C1", "node": "intake", "ts": "2026-10-05T12:00:01.000+00:00",
               "status": "ok", "model": "m", "prompt_tokens": 100, "completion_tokens": 20,
               "outputs": {"cause": "animal"}},
              {"claim_id": "C1", "node": "router", "ts": "2026-10-05T12:00:05.000+00:00",
               "status": "ok", "model": None, "outputs": {"route": "human_review"}}]  # fmt: skip
    (tmp_path / "C1.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events) + "\n")
    rows = audit_rows(tmp_path, "C1")
    assert [r["node"] for r in rows] == ["intake", "router"]
    assert rows[0]["time"] == "12:00:01" and rows[0]["tokens"] == 120 and rows[1]["model"] == "-"
    assert audit_rows(tmp_path, "missing") == []
