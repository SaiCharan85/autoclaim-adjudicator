from datetime import date

import pytest
from auto_fakes import facts, package

from autoclaim.config import load_carrier_config
from autoclaim.lines.auto.facts import coverage_part, derive, driver_role

US = load_carrier_config().jurisdictions["US"]


def test_deer_trace_comprehensive_deductible_payout() -> None:
    d = derive(facts(), package(), US)
    assert d.coverage_part == "comprehensive" and d.part_on_policy
    assert d.deductible == 250 and not d.total_loss and d.payout == 1_650
    assert d.notice_days == 1 and not d.late_notice


def test_deer_on_collision_only_policy_has_no_part() -> None:
    d = derive(facts(), package(coverage="collision", comprehensive_deductible=None), US)
    assert d.coverage_part == "comprehensive" and not d.part_on_policy and d.deductible is None


@pytest.mark.parametrize(
    ("name", "role"),
    [("Maria Garcia", "named_insured"), ("daniel  garcia", "listed_driver"),
     ("Kevin Garcia", "excluded_driver"), ("Kevin", "excluded_driver"),
     ("Jake Moore", "permissive_unlisted"), (None, "unknown")],
)  # fmt: skip
def test_driver_lookup(name, role) -> None:
    assert driver_role(name, package()) == role


def test_total_loss_and_below_deductible() -> None:
    total = derive(facts(cause="collision_object"),
                   package().model_copy(update={"estimate_amount": 12_000.0}), US)  # fmt: skip
    assert total.total_loss and total.gross_loss == 15_000 and total.payout == 14_500
    small = derive(facts(cause="collision_object"),
                   package().model_copy(update={"estimate_amount": 400.0}), US)  # fmt: skip
    assert small.payout == 0


def test_hit_and_run_condition_and_business_use() -> None:
    late = derive(facts(cause="hit_and_run", police_report=True, police_report_hours=30), package(),
                  US)  # fmt: skip
    ok = derive(
        facts(cause="hit_and_run", police_report=True, police_report_hours=2), package(), US
    )
    assert late.hit_and_run_report_ok is False and ok.hit_and_run_report_ok is True
    assert derive(facts(), package(), US).hit_and_run_report_ok is None
    uber = derive(facts(use_at_loss="rideshare_active"), package(), US)
    assert uber.business_use_without_endorsement
    endorsed = derive(
        facts(use_at_loss="rideshare_active"), package(rideshare_endorsement=True), US
    )
    assert not endorsed.business_use_without_endorsement


def test_late_notice_and_unknowns() -> None:
    d = derive(facts(loss_date=date(2024, 6, 1)), package(), US)
    assert d.notice_days == 63 and d.late_notice
    u = derive(facts(loss_date=None, cause="unknown"), package(), US)
    assert u.notice_days is None and u.coverage_part == "unknown" and not u.part_on_policy
    assert coverage_part("mechanical_breakdown", US) == "none"
