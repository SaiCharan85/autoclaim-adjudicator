import json
from collections import Counter
from datetime import date
from typing import cast

import judgekit
import pandas as pd
import pytest

from autoclaim.config import load_carrier_config
from autoclaim.core.decision import Decision
from autoclaim.core.judge import load_rubric
from autoclaim.finetune import datasets as ft
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH
from autoclaim.lines.auto.claim import policy_record
from autoclaim.lines.auto.facts import ClaimFacts
from autoclaim.lines.auto.fraud_tools import FraudSignals
from autoclaim.lines.auto.line import ADJUDICATE, INTAKE, AutoLine
from autoclaim.lines.auto.policy import POLICY_PATH
from autoclaim.llm.client import LLMClient, system_prompt
from autoclaim.retrieval.corpus import load_policy
from autoclaim.retrieval.retriever import Hit

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


# ---------------------------------------------------------------- adjudicator


class FakeRetriever:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def search(self, query: str, k: int, hops: int, max_expanded: int = 6) -> list[Hit]:
        return [Hit(POLICY.by_id[i], 1.0, 0, "test") for i in self.ids]


class FakeToolkit:
    def __init__(self, score: float) -> None:
        self.score = score

    def assess(self, record: dict) -> FraudSignals:
        return FraudSignals(model_score=self.score, top_reasons=[], anomaly_score=0.5,
                            rules_fired=[], rule_score=0.0, rules_unchecked=[],
                            model_version="test")  # fmt: skip


def line(score: float = 0.05, ids: tuple[str, ...] = ("INS-COLLISION", "LIM-DEDUCTIBLE",
         "COND-HIT-AND-RUN-POLICE-REPORT", "EXC-EXCLUDED-DRIVER")) -> AutoLine:  # fmt: skip
    client = cast(LLMClient, ft.GoldClient())
    return AutoLine(client=client, policy=POLICY, retriever=FakeRetriever(list(ids)),  # type: ignore[arg-type]
                    toolkit=FakeToolkit(score), judge_impl=None, jurisdiction=JUR,  # type: ignore[arg-type]
                    fraud_review_score=0.24)  # fmt: skip


def target(ex: dict) -> Decision:
    return Decision.model_validate_json(ex["messages"][2]["content"])


def test_adjudicator_example_uses_the_production_prompt_and_true_decision() -> None:
    ex = ft.adjudicator_example(row(), line(), JUR, 0.24)
    assert ex is not None
    system, user = ex["messages"][0]["content"], ex["messages"][1]["content"]
    assert system == system_prompt(ADJUDICATE, Decision)
    for part in ("declarations:", "facts:", "deterministic:", "coverage_analysis:", "fraud:",
                 "clauses:\n", "[INS-COLLISION]"):  # fmt: skip
        assert part in user
    d = target(ex)
    assert d.outcome == "approve" and d.payout == 661.91 and d.reasons == ["covered_loss"]
    assert "COND-HIT-AND-RUN-POLICE-REPORT" in d.cited_clauses
    assert (
        ex["meta"]["primary_reason"] == "covered_loss" and ex["meta"]["loss_date"] == "2023-05-15"
    )


def test_denial_cites_the_exclusion_and_coverage_agrees() -> None:
    late = row(police_report_hours=60.0, gt_decision="deny",
               gt_reasons="hit_and_run_report_condition", gt_payout=0.0)  # fmt: skip
    ex = ft.adjudicator_example(late, line(), JUR, 0.24)
    assert ex is not None
    d = target(ex)
    assert d.outcome == "deny" and d.payout is None
    assert d.cited_clauses == ["COND-HIT-AND-RUN-POLICE-REPORT"]
    assert '"covered":"no"' in ex["messages"][1]["content"]
    assert '"conditions_unmet":["COND-HIT-AND-RUN-POLICE-REPORT"]' in ex["messages"][1]["content"]


def test_fraud_escalation_only_when_the_score_shows_it() -> None:
    fraud = row(gt_decision="escalate", gt_reasons="suspected_fraud_siu", gt_payout=0.0)
    assert ft.adjudicator_example(fraud, line(score=0.05), JUR, 0.24) is None  # unlearnable
    ex = ft.adjudicator_example(fraud, line(score=0.6), JUR, 0.24)
    assert ex is not None
    d = target(ex)
    assert d.outcome == "escalate" and d.reasons == ["suspected_fraud_siu"]
    assert "0.60" in d.explanation and d.explanation.endswith("We refer the claim to an adjuster.")


def test_low_score_fraud_with_late_notice_keeps_only_the_visible_reason() -> None:
    r = row(report_date="2023-07-20", gt_decision="escalate", gt_payout=0.0,
            gt_reasons="late_notice_prejudice_review;suspected_fraud_siu")  # fmt: skip
    ex = ft.adjudicator_example(r, line(score=0.05), JUR, 0.24)
    assert ex is not None and target(ex).reasons == ["late_notice_prejudice_review"]


@pytest.mark.parametrize(
    "over",
    [{"gt_reasons": "missing_information", "gt_decision": "escalate"},  # not supported
     {"gt_reasons": "excluded_driver", "gt_decision": "deny"},  # the facts don't support it
     {"gt_payout": 900.0}],  # inflated estimate: derived payout is not the true one
)  # fmt: skip
def test_unfaithful_or_unsupported_truth_is_skipped(over: dict) -> None:
    assert ft.adjudicator_example(row(**over), line(), JUR, 0.24) is None


def test_gold_coverage_cites_only_retrieved_clauses() -> None:
    x = je.derive(je.true_facts(row()), je.package(row()), JUR)
    cov = ft.gold_coverage(
        x, ["covered_loss"], ["INS-COLLISION", "LIM-DEDUCTIBLE"], ["DEF-INSURED"]
    )
    assert [s.clause_id for s in cov.reasoning] == ["DEF-INSURED"] and cov.covered == "yes"


def test_gold_client_refuses_roles_it_has_no_answer_for() -> None:
    with pytest.raises(ValueError, match="intake"):
        ft.GoldClient().structured("intake", "", "", ClaimFacts)


def test_adjudicator_examples_balance_reasons_and_respect_the_window() -> None:
    base = [row(claim_id=f"CLM-{i:06d}", source_record=f"r{i}") for i in range(30)]
    deny = [row(claim_id=f"CLM-{i:06d}", source_record=f"r{i}", police_report_hours=60.0,
                gt_decision="deny", gt_reasons="hit_and_run_report_condition", gt_payout=0.0)
            for i in range(30, 60)]  # fmt: skip
    late = [row(claim_id="CLM-999999", loss_date="2024-02-01", report_date="2024-02-02")]
    claims = pd.DataFrame(base + deny + late)
    out = ft.adjudicator_examples(claims, "2000-01-01", "2024-01-01", 10, line(), JUR, 0.24, 1,
                                  approve_share=0.5)  # fmt: skip
    kinds = Counter(e["meta"]["primary_reason"] for e in out)
    assert kinds["covered_loss"] == 5 and kinds["hit_and_run_report_condition"] == 1
    assert all(e["meta"]["claim_id"] != "CLM-999999" for e in out)
