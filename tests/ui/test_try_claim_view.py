from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from autoclaim.datasets.nhtsa_complaints import Complaint
from autoclaim.lines.auto.claim import ClaimPackage
from autoclaim.ui import try_claim_view as tv

STORY = "I was stopped at a red light when the car behind me failed to stop and hit my bumper."
PAGE = Path(__file__).resolve().parents[2] / "src" / "autoclaim" / "ui" / "pages" / "try_a_claim.py"
TODAY = date(2026, 10, 8)


def complaint(**over: object) -> Complaint:
    base = {"odi_number": 11600001, "make": "Honda", "model": "Civic", "year": 2021,
            "body_class": "car", "date_of_incident": date(2024, 9, 18),
            "date_filed": date(2024, 12, 30), "injuries": 0, "components": "SERVICE BRAKES",
            "summary": STORY}  # fmt: skip
    return Complaint.model_validate(base | over)


def form(**over: object) -> tv.ClaimForm:
    base = tv.form_from_complaint(complaint(), TODAY).model_copy(
        update={"policyholder": "Jordan Lee"}
    )
    return base.model_copy(update=over)


def test_form_defaults_follow_the_complaint() -> None:
    f = form()
    assert (f.vehicle_year, f.vehicle_make, f.vehicle_model) == (2021, "Honda", "Civic")
    assert f.loss_date == date(2024, 9, 18) and f.report_date == date(2024, 9, 20)  # promptly
    assert f.policy_start_date < f.loss_date and f.vehicle_acv > 0


def test_report_date_never_in_the_future() -> None:
    f = tv.form_from_complaint(complaint(date_of_incident=TODAY), TODAY)
    assert f.report_date == TODAY


def test_package_keeps_the_story_verbatim_after_a_form_header() -> None:
    pkg = ClaimPackage.model_validate(tv.build_package(STORY, form()))
    head, _, body = pkg.narrative.partition("\nStatement:\n")
    assert body == STORY
    assert "policyholder, Jordan Lee" in head and "2024-09-18" in head and "Honda Civic" in head
    assert pkg.policy.named_insured == "Jordan Lee" and pkg.claim_id.startswith("TRY-")


def test_deductibles_follow_the_coverage() -> None:
    liab = ClaimPackage.model_validate(tv.build_package(STORY, form(coverage="liability_only")))
    assert liab.policy.collision_deductible is None and liab.policy.comprehensive_deductible is None
    coll = ClaimPackage.model_validate(tv.build_package(STORY, form(coverage="collision")))
    assert coll.policy.collision_deductible == 500.0
    assert coll.policy.comprehensive_deductible is None


def test_claim_id_is_stable_and_changes_with_any_input() -> None:
    a = tv.claim_id(STORY, form())
    assert a == tv.claim_id(STORY + "  ", form())  # same story, same form: same claim
    assert a != tv.claim_id(STORY, form(estimate_amount=4600.0))
    assert a != tv.claim_id(STORY + " Also the trunk.", form())


def test_form_validation() -> None:
    with pytest.raises(ValidationError):
        tv.ClaimForm.model_validate(form().model_dump() | {"estimate_amount": 0})
    assert tv.missing_fields(form().model_dump() | {"policyholder": "  "}) == ["policyholder name"]


def test_acv_default_depreciates_with_a_floor() -> None:
    assert tv.acv_default(2024, "suv", TODAY) > tv.acv_default(2018, "suv", TODAY)
    assert tv.acv_default(1995, "car", TODAY) == 4500.0  # 15% of a $30k new car
    assert tv.acv_default(2026, "boat", TODAY) == 30000.0  # unknown body: car prices


def test_summarize_decided_and_paused_runs() -> None:
    decided = {"final": {"decided_by": "auto", "outcome": "approve", "payout": 4000.0,
                         "reasons": ["covered_loss"]},
               "decision": {"outcome": "approve", "explanation": "Covered."},
               "facts": {"derived": {"coverage_part": "collision", "deductible": 500.0}},
               "llm_calls": 4, "tokens": 9000}  # fmt: skip
    r = tv.summarize(decided)
    assert (r["status"], r["outcome"], r["payout"], r["coverage_part"]) == (
        "decided", "approve", 4000.0, "collision")  # fmt: skip

    proposal = {"outcome": "deny", "reasons": ["excluded_driver"]}
    paused = SimpleNamespace(
        value={"route_reasons": ["low_confidence"], "proposed_decision": proposal}
    )
    r = tv.summarize({"__interrupt__": [paused], "decision": {"outcome": "deny"}})
    assert r["status"] == "human_review" and r["route_reasons"] == ["low_confidence"]
    assert r["outcome"] is None and r["proposed"] == "deny" and r["reasons"] == ["excluded_driver"]


def test_page_without_downloaded_stories_explains_what_to_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOCLAIM_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(str(PAGE), default_timeout=60).run()
    assert not at.exception
    assert any("nhtsa_complaints" in w.value for w in at.warning)  # no harness was built


RUN = {"facts": {"extracted": {"cause": "collision_vehicle", "driver_name": "Jordan Lee"},
                 "derived": {"driver_role": "named_insured", "coverage_part": "collision",
                             "part_on_policy": True, "deductible": 500.0, "gross_loss": 4500.0,
                             "payout": 4000.0, "total_loss": False}},
       "fraud": {"model_score": 0.01}, "critic": {"passed": True}, "judge": {"passed": True},
       "final": {"decided_by": "auto", "outcome": "approve", "payout": 4000.0}}  # fmt: skip


def test_trace_tells_the_story_of_a_clean_approval() -> None:
    steps = tv.trace(RUN, 0.24)
    assert [s["title"] for s in steps] == [
        "Read the story", "Checked the driver", "Matched the coverage", "Screened for fraud",
        "Worked out the payout", "Double-checked the decision",
        "Decided automatically"]  # fmt: skip
    assert all(s["status"] == "ok" for s in steps)
    assert "$4,500.00 repair estimate - $500.00 deductible = $4,000.00" in steps[4]["detail"]


def test_trace_flags_problems() -> None:
    bad = {**RUN, "facts": {"extracted": {"cause": "unknown"},
                            "derived": {**RUN["facts"]["derived"], "driver_role": "excluded_driver",
                                        "part_on_policy": False, "total_loss": True}},
           "fraud": {"model_score": 0.4}, "judge": {"passed": False}, "retries": 2}  # fmt: skip
    del bad["final"]
    bad["__interrupt__"] = [SimpleNamespace(value={"route_reasons": ["checks_failed"]})]
    s = {x["title"]: x for x in tv.trace(bad, 0.24)}
    assert s["Read the story"]["status"] == "warn" and s["Checked the driver"]["status"] == "bad"
    assert s["Matched the coverage"]["status"] == "bad"
    assert (
        s["Screened for fraud"]["status"] == "warn"
        and "at or above" in s["Screened for fraud"]["detail"]
    )
    assert s["Worked out the payout"]["detail"].startswith("Nothing is payable")
    assert "after 2 revisions" in s["Double-checked the decision"]["detail"]
    assert "checks failed" in s["Sent to a human adjuster"]["detail"]


def test_verdict_banner() -> None:
    assert tv.verdict({"status": "decided", "outcome": "approve", "payout": 4000.0}) == (
        "Approved: insurer pays $4,000.00",
        "Paid to the policyholder or the repair shop.",
        "ok",
    )
    assert tv.verdict({"status": "decided", "outcome": "deny", "payout": None})[2] == "bad"
    assert tv.verdict({"status": "human_review", "outcome": None, "payout": None})[2] == "warn"


LONG = (
    "On May 20th, 2023 at about 9:50 P.M. we struck a deer head on. The impact was severe "
    "enough that the passenger airbag should have deployed, but it failed to deploy at all."
)


def test_pasted_nhtsa_story_is_recognized_whole_or_in_part() -> None:
    c = complaint(summary=LONG, odi_number=42)
    others = [complaint(odi_number=1), c]
    assert tv.match_complaint(LONG, others) is c
    assert tv.match_complaint("  " + LONG.upper().replace(" ", "\n "), others) is c  # spacing/case
    assert tv.match_complaint(LONG[:130], others) is c  # a long excerpt
    assert tv.match_complaint(LONG + " Extra words I typed.", others) is c


def test_own_story_or_short_text_is_not_matched() -> None:
    others = [complaint(summary=LONG)]
    assert tv.match_complaint("A deer ran out on the highway.", others) is None
    assert tv.match_complaint("x" * 200, others) is None and tv.match_complaint(LONG, []) is None


def test_missing_fields_names_every_empty_required_detail() -> None:
    full = tv.generic_form(TODAY).model_dump() | {"policyholder": "Jordan Lee"}
    assert tv.missing_fields(full) == []
    empty = full | {"estimate_amount": None, "loss_date": None, "vehicle_make": ""}
    assert tv.missing_fields(empty) == ["repair estimate", "date of loss", "make"]


def test_no_made_up_policyholder_name() -> None:
    assert tv.form_from_complaint(complaint(), TODAY).policyholder == ""
    assert tv.generic_form(TODAY).policyholder == ""
