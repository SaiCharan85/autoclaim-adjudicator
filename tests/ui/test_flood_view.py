from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from autoclaim.lines.flood.build import build_flood
from autoclaim.ui import flood_view as fv

PAGE = Path(__file__).resolve().parents[2] / "src" / "autoclaim" / "ui" / "views" / "flood_claim.py"


def run(form: fv.FloodForm) -> dict:
    _, graph = build_flood()  # in-memory state, template explanations: no LLM calls
    c = fv.to_claim(form)
    return graph.invoke({"claim_id": c.claim_id, "claim": c.model_dump(mode="json")},
                        {"configurable": {"thread_id": c.claim_id}})  # fmt: skip


def test_form_maps_dollar_deductibles_to_fema_codes() -> None:
    c = fv.to_claim(fv.SAMPLE)
    assert c.building_deductible_code == c.contents_deductible_code == "2"  # $2,000
    assert c.cause_codes == ["4"] and c.building_damage == 38_500.0
    assert c.claim_id == fv.claim_id(fv.SAMPLE) and c.claim_id.startswith("FLD-")
    assert fv.claim_id(fv.SAMPLE.model_copy(update={"building_damage": 38_600.0})) != c.claim_id


def test_missing_fields() -> None:
    assert fv.missing_fields(fv.SAMPLE) == []
    gaps = fv.missing_fields(fv.FloodForm())
    assert gaps == ["policyholder name", "date of loss", "policy start",
                    "building or contents damage"]  # fmt: skip
    with pytest.raises(ValueError, match="date of loss"):
        fv.to_claim(fv.FloodForm())


def test_sample_flash_flood_is_approved_for_46700() -> None:
    out = run(fv.SAMPLE)
    r = fv.flood_summary(out)
    assert (r["status"], r["outcome"], r["payout"]) == ("decided", "approve", 46_700.0)
    steps = {s["title"]: s for s in fv.flood_trace(out)}
    assert "$38,500.00 damage - $2,000.00 deductible = $36,500.00" in (
        steps["Building coverage"]["detail"])  # fmt: skip
    assert steps["Total"]["detail"] == "Building + contents = $46,700.00."
    assert "Decided automatically" in steps


def test_over_the_50k_authority_goes_to_an_adjuster() -> None:
    r = fv.flood_summary(run(fv.SAMPLE.model_copy(update={"building_damage": 45_000.0})))
    assert r["status"] == "human_review" and r["payout"] == 53_200.0


@pytest.mark.parametrize(
    ("update", "status", "outcome"),
    [({"date_of_loss": date(2024, 1, 5)}, "human_review", None),  # before the policy start
     ({"building_damage": 1_500.0, "contents_damage": None}, "decided", "deny"),  # deductible
     ({"cause_code": "9"}, "human_review", None)],  # earth movement: not clearly a flood
)  # fmt: skip
def test_judgment_and_deductible_cases(update: dict, status: str, outcome: str | None) -> None:
    out = run(fv.SAMPLE.model_copy(update=update))
    r = fv.flood_summary(out)
    assert (r["status"], r["outcome"]) == (status, outcome)
    assert any(s["status"] != "ok" for s in fv.flood_trace(out)) or outcome == "deny"


def test_page_story_to_decision_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    import autoclaim.ui.shared as shared

    monkeypatch.setattr(shared, "llm_client", lambda: None)  # code-only story reading
    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception
    at.radio[0].set_value("Write or paste my own").run()
    at.text_area[0].set_value(STORY).run()
    next(b for b in at.button if b.label == "Use this story").click().run()
    assert not at.exception
    by_label = {w.label.split(" :")[0]: w for w in list(at.text_input) + list(at.number_input)}
    assert by_label["Building damage ($)"].value == 38_500.0  # read from the story
    by_label["Policyholder"].set_value("Priya Raman")
    at.date_input[0].set_value(date(2024, 6, 1))  # policy start: never in a story
    next(b for b in at.button if "Decide" in b.label).click().run()
    assert not at.exception
    verdicts = [m.value for m in at.markdown if 'class="verdict' in m.value]
    assert verdicts and "Approved: insurer pays $36,500.00" in verdicts[0]


def test_app_has_an_auto_and_a_flood_section() -> None:
    src = (PAGE.parents[1] / "app.py").read_text(encoding="utf-8")
    assert '"Auto": [' in src and '"Flood": [' in src and "views/flood_claim.py" in src


STORY = (
    "On May 18, 2026 a flash flood after hours of heavy rain put 14 inches of water through "
    "the ground floor of our house. The contractor estimated the repairs at $38,500."
)
TODAY = date(2026, 10, 10)
RAW = pd.DataFrame([{
    "id": "abc", "dateOfLoss": "2024-09-27", "state": "FL", "ratedFloodZone": "AE",
    "causeOfDamage": "1", "floodEvent": "Hurricane Helene", "originalNBDate": "2019-05-01",
    "totalBuildingInsuranceCoverage": 250000, "totalContentsInsuranceCoverage": 50000,
    "buildingDamageAmount": 41250.0, "contentsDamageAmount": 8000.0,
    "buildingDeductibleCode": "2", "contentsDeductibleCode": "0", "waterDepth": 22.0,
    "primaryResidenceIndicator": True,
}, {"id": "zero", "dateOfLoss": "2024-09-27", "causeOfDamage": "1", "floodEvent": "x",
    "buildingDamageAmount": 0.0}])  # fmt: skip


def test_fema_records_keep_claims_with_damage_and_read_like_a_story() -> None:
    recs = fv.records_from_frame(RAW)
    assert [r.id for r in recs] == ["abc"]
    r = recs[0]
    assert (r.building_deductible, r.contents_deductible, r.cause_code) == (2000.0, 500.0, "1")
    text = fv.record_summary(r)
    assert "Hurricane Helene" in text and "tidal water overflow" in text and "$41,250" in text
    form = fv.form_from_record(r)
    assert form.policyholder == "" and form.building_damage == 41_250.0
    assert form.policy_start == date(2019, 5, 1)


def test_flood_story_code_reading() -> None:
    d = fv.flood_details_from_text(STORY, TODAY)
    assert (d.date_of_loss, d.cause_code, d.water_depth_inches, d.building_damage) == (
        date(2026, 5, 18), "4", 14.0, 38_500.0)  # fmt: skip
    assert (
        fv.flood_details_from_text("The river rose 2 feet into the den.", TODAY).cause_code == "2"
    )
    assert fv.flood_details_from_text("The river rose 2 feet.", TODAY).water_depth_inches == 24.0


class FakeClient:
    def __init__(self, value: object) -> None:
        self.value = value

    def structured(self, *args: object, **kw: object) -> object:
        if isinstance(self.value, Exception):
            raise self.value
        return SimpleNamespace(value=self.value)


def test_flood_story_model_reading_with_code_fallback_and_checks() -> None:
    llm = fv.FloodStoryDetails(contents_damage=9000.0, policyholder="Priya Raman",
                               cause_code="7", date_of_loss=date(2030, 1, 1))  # fmt: skip
    d = fv.read_flood_story(STORY, TODAY, FakeClient(llm))
    assert d.contents_damage == 9000.0 and d.policyholder == "Priya Raman"
    assert d.cause_code == "7" and d.date_of_loss == date(2026, 5, 18)  # future date dropped
    assert fv.read_flood_story(STORY, TODAY, FakeClient(RuntimeError())).building_damage == 38_500.0


def test_flood_sources() -> None:
    rec = fv.records_from_frame(RAW)[0]
    src = fv.flood_sources(rec, None)
    assert src["building_damage"] == "fema" and src["policyholder"] == "needed"
    story = fv.flood_sources(None, fv.flood_details_from_text(STORY, TODAY))
    assert story["building_damage"] == "story" and story["contents_damage"] == "needed"
    assert story["building_limit"] == "sample" and story["policy_start"] == "needed"
