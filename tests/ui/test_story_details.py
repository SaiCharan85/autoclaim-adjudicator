from datetime import date
from types import SimpleNamespace

import pytest

from autoclaim.ui import try_claim_view as tv

TODAY = date(2026, 10, 8)
CX5 = ("Was driving up [XXX] when my dash lit up. The 2023 CX 5 with only 18K struck the guard "
       "rail.")  # fmt: skip


class FakeClient:
    def __init__(self, value: tv.StoryDetails | Exception) -> None:
        self.value, self.calls = value, []

    def structured(self, role, instructions, user, schema, **kw):
        self.calls.append((role, user))
        if isinstance(self.value, Exception):
            raise self.value
        return SimpleNamespace(value=self.value)


@pytest.mark.parametrize(
    ("story", "want"),
    [(CX5, (2023, "Mazda", "CX-5", "suv")),
     ("My 2021 Honda Civic was rear-ended.", (2021, "Honda", "Civic", "car")),
     ("I called toyota corporate after the crash.", (None, "Toyota", None, None)),
     ("Our 2022 F-150 slid off the road.", (2022, "Ford", "F-150", "pickup")),
     ("Nothing about the car here at all.", (None, None, None, None))],
)  # fmt: skip
def test_code_reading_finds_the_car(story: str, want: tuple) -> None:
    d = tv.details_from_text(story, TODAY)
    assert (d.vehicle_year, d.vehicle_make, d.vehicle_model, d.body_class) == want


def test_code_reading_finds_dates_and_amounts_but_never_invents() -> None:
    d = tv.details_from_text("On May 20th, 2023 a deer hit us; the shop quoted $3,850.50. "
                             "Earlier, on 01/02/2023, nothing happened.", TODAY)  # fmt: skip
    assert d.loss_date == date(2023, 5, 20) or d.loss_date == date(2023, 1, 2)
    assert d.estimate_amount == 3850.0
    assert (
        tv.details_from_text("It was about 4k short of what we owe.", TODAY).estimate_amount is None
    )
    assert (
        tv.details_from_text("On 12/31/2030 it will be fixed.", TODAY).loss_date is None
    )  # future


def test_model_reading_wins_and_code_fills_its_gaps() -> None:
    llm = tv.StoryDetails(vehicle_make="Mazda", loss_date=date(2024, 8, 9), policyholder="Ana Ruiz")
    client = FakeClient(llm)
    d = tv.read_story(CX5, TODAY, client)
    assert client.calls[0][0] == "intake" and "statement:" in client.calls[0][1]
    assert (d.vehicle_make, d.loss_date, d.policyholder) == ("Mazda", date(2024, 8, 9), "Ana Ruiz")
    assert (d.vehicle_year, d.vehicle_model) == (2023, "CX-5")  # from the code reading


def test_model_outage_falls_back_to_code_reading() -> None:
    d = tv.read_story(CX5, TODAY, FakeClient(RuntimeError("quota")))
    assert (d.vehicle_year, d.vehicle_make, d.vehicle_model) == (2023, "Mazda", "CX-5")
    assert tv.read_story(CX5, TODAY, None).vehicle_make == "Mazda"


def test_impossible_model_answers_are_dropped() -> None:
    bad = tv.StoryDetails(vehicle_year=1850, loss_date=date(2030, 1, 1), estimate_amount=-5)
    d = tv.read_story("A plain story with no details in it at all, really.", TODAY, FakeClient(bad))
    assert d.vehicle_year is None and d.loss_date is None and d.estimate_amount is None


def test_defaults_and_what_is_still_missing() -> None:
    d = tv.StoryDetails(vehicle_year=2023, vehicle_make="Mazda", vehicle_model="CX-5",
                        body_class="suv", loss_date=date(2024, 8, 9))  # fmt: skip
    v = tv.defaults_from_details(d, TODAY)
    assert v["report_date"] == date(2024, 8, 11) and v["vehicle_acv"] > 0
    assert v["estimate_amount"] is None and v["policyholder"] == ""
    found, missing = tv.found_and_missing(d)
    assert found == ["year", "make", "model", "date of loss"]
    assert missing == ["repair estimate", "policyholder name"]
    empty = tv.defaults_from_details(tv.StoryDetails(), TODAY)
    assert (
        empty["vehicle_acv"] is None and empty["loss_date"] is None and empty["body_class"] == "car"
    )


@pytest.mark.parametrize(
    ("name", "kept"), [("toy", None), ("- toy", None), ("Ana Ruiz", "Ana Ruiz")]
)
def test_only_a_full_name_counts_as_the_policyholder(name: str, kept: str | None) -> None:
    d = tv.read_story(CX5, TODAY, FakeClient(tv.StoryDetails(policyholder=name)))
    assert d.policyholder == kept


def test_nhtsa_record_fills_the_car_and_date_never_the_estimate_or_name() -> None:
    src = tv.field_sources(True, None)
    assert {f for f, v in src.items() if v == "record"} == {
        "loss_date", "vehicle_year", "vehicle_make", "vehicle_model", "body_class"}  # fmt: skip
    assert src["report_date"] == src["vehicle_acv"] == "estimated"
    assert src["estimate_amount"] == src["policyholder"] == "needed"


def test_story_sources_follow_what_the_story_said() -> None:
    d = tv.StoryDetails(vehicle_year=2023, vehicle_make="Mazda", vehicle_model="CX-5")
    src = tv.field_sources(False, d)
    assert src["vehicle_make"] == "story" and src["vehicle_acv"] == "estimated"
    assert src["loss_date"] == src["report_date"] == src["estimate_amount"] == "needed"
    assert src["body_class"] == "estimated"  # a default to check, not a fact from the story
    assert set(tv.field_sources(False, None).values()) <= {"needed", "estimated"}


def test_tags_and_progress() -> None:
    assert tv.tagged("Make", "story") == "Make :green[● from the story]"
    assert "please enter" in tv.tagged("Year", "needed")
    src = tv.field_sources(True, None)
    assert tv.autofill_progress(src) == (7, 9)
