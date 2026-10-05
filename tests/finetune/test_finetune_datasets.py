import json
from datetime import date

import judgekit
import pandas as pd
import pytest

from autoclaim.config import load_carrier_config
from autoclaim.core.judge import load_rubric
from autoclaim.finetune import datasets as ft
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH
from autoclaim.lines.auto.claim import policy_record
from autoclaim.lines.auto.facts import ClaimFacts
from autoclaim.lines.auto.line import INTAKE
from autoclaim.lines.auto.policy import POLICY_PATH
from autoclaim.llm.client import system_prompt
from autoclaim.retrieval.corpus import load_policy

JUR = load_carrier_config().active_jurisdiction
POLICY = load_policy(POLICY_PATH)
VAL, TEST = date(2024, 1, 1), date(2024, 7, 1)
ROW = {
    "claim_id": "CLM-000001", "policy_id": "POL-000001", "loss_date": "2023-05-15",
    "report_date": "2023-05-16", "cause": "hit_and_run", "use_at_loss": "personal",
    "vehicle_role": "in_transport", "n_vehicles": 2, "injury_count": 0.0,
    "damage_extent": "minor", "towed": 0.0, "at_fault": 0.0, "police_report": True,
    "police_report_hours": 14.6, "witness_count": 1, "attorney_involved": False,
    "claimed_amount": 1161.91, "policy_state": "OH", "policy_start_date": "2023-01-01",
    "coverage": "collision_comprehensive", "collision_deductible": 500.0,
    "comprehensive_deductible": 250.0, "rideshare_endorsement": False,
    "driver_role": "named_insured", "vehicle_model_year": 2018, "vehicle_make": "Honda",
    "vehicle_model": "Civic", "body_class": "car", "vehicle_acv": 15000.0, "adas": False,
    "financed": False, "prior_claims_3y": 0, "address_change_days": float("nan"),
    "gt_decision": "approve", "gt_reasons": float("nan"), "gt_payout": 661.91,
    "source_record": "crss:2022:1:1",
}  # fmt: skip


def row(**over: object) -> pd.Series:
    return pd.Series({**ROW, **over})


# ---------------------------------------------------------------- intake targets


def test_intake_target_is_the_truth_with_rounded_hours() -> None:
    t = ft.intake_target(row(), None)
    assert t.cause == "hit_and_run" and t.witnesses == 1
    assert t.police_report_hours == 15  # stories say "that afternoon", not 14.6 hours
    assert t.missing_info == []


@pytest.mark.parametrize(
    ("leave_out", "field", "missing"),
    [("witness_count", "witnesses", "witnesses"),
     ("police_report_hours", "police_report_hours", "police_report_timing")],
)  # fmt: skip
def test_left_out_fact_is_null_and_listed_missing(leave_out: str, field: str, missing: str) -> None:
    t = ft.intake_target(row(), leave_out)
    assert getattr(t, field) is None and t.missing_info == [missing]


def test_non_extracted_omissions_change_nothing() -> None:
    assert ft.intake_target(row(), "weather") == ft.intake_target(row(), None)


def test_intake_examples_use_the_production_prompt() -> None:
    r = row()
    pkg = {"claim_id": r["claim_id"], "channel": "web", "report_date": "2023-05-16",
           "estimate_amount": 1161.91, "narrative": "Someone hit my car and drove off, "
           "I called the police that afternoon and a neighbor saw it happen.",
           "policy": policy_record(r).model_dump(mode="json")}  # fmt: skip
    claims = pd.DataFrame([dict(r)])
    (ex,) = ft.intake_examples([{"package": pkg, "style": {"leave_out": None}}], claims)
    system, user, assistant = (m["content"] for m in ex["messages"])
    assert system == system_prompt(INTAKE, ClaimFacts)
    assert user.startswith("channel: web\nreport_date: 2023-05-16\nstatement:\n")
    assert ClaimFacts.model_validate_json(assistant).cause == "hit_and_run"
    assert ex["meta"] == {"claim_id": "CLM-000001", "source_record": "crss:2022:1:1",
                          "loss_date": "2023-05-15"}  # fmt: skip


# ---------------------------------------------------------------- judge examples


def test_judge_examples_label_planted_items_no_and_clean_yes() -> None:
    rubric = load_rubric(RUBRIC_PATH)
    glass = row(cause="glass", police_report=False, police_report_hours=float("nan"),
                gt_payout=911.91)  # fmt: skip
    s = je.reference_sample(glass, JUR, POLICY)
    assert s is not None
    examples = ft.judge_examples([s], je.ERROR_TYPES, 0, {s.id: "crss:x"}, {s.id: "2023-05-15"})
    clean = next(e for e in examples if e["meta"]["error_type"] is None)
    assert all(a["answer"] == "yes" for a in json.loads(clean["messages"][2]["content"])["answers"])
    expected = {e.name: e.expected_items for e in je.ERROR_TYPES}
    for ex in examples:
        if ex["meta"]["error_type"] is None:
            continue
        answers = json.loads(ex["messages"][2]["content"])["answers"]
        noes = {a["id"] for a in answers if a["answer"] == "no"}
        assert noes == expected[ex["meta"]["error_type"]]
        assert all(a["note"] for a in answers if a["answer"] == "no")
    system = system_prompt(judgekit.render_system(rubric), judgekit.JudgeResponse)
    assert all(ex["messages"][0]["content"] == system for ex in examples)


def test_judge_answer_covers_every_rubric_item() -> None:
    rubric = load_rubric(RUBRIC_PATH)
    out = json.loads(ft.judge_answer(rubric, frozenset({"facts_supported"}), "invented"))
    assert [a["id"] for a in out["answers"]] == list(rubric.ids)


# ---------------------------------------------------------------- leakage-safe split


def ex(cid: str, when: str, src: str) -> dict:
    return {"messages": [], "meta": {"claim_id": cid, "loss_date": when, "source_record": src}}


def test_split_by_period_and_source_record() -> None:
    out = ft.split([ex("a", "2023-01-01", "r1"), ex("b", "2024-02-01", "r2"),
                    ex("c", "2024-03-01", "r1")], VAL, TEST)  # fmt: skip
    assert [e["meta"]["claim_id"] for e in out["train"]] == ["a"]
    assert [e["meta"]["claim_id"] for e in out["val"]] == ["b"]  # c shares r1 with training


def test_split_refuses_the_locked_test_period() -> None:
    with pytest.raises(ValueError, match="test-period"):
        ft.split([ex("t", "2024-07-01", "r9")], VAL, TEST)


def test_period_of() -> None:
    assert ft.period_of("2023-12-31", VAL, TEST) == "train"
    assert ft.period_of("2024-01-01", VAL, TEST) == "val"
    assert ft.period_of(date(2024, 7, 1), VAL, TEST) == "test"


def test_jsonl_round_trip() -> None:
    examples = [ft.chat("s", "u", "a", claim_id="x")]
    assert json.loads(ft.to_jsonl(examples)) == examples[0]
