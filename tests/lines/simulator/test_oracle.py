"""Coverage oracle on hand-built claims: one rule at a time, then precedence and payouts."""

import numpy as np
import pandas as pd
import pytest

from autoclaim.config import load_carrier_config
from autoclaim.lines.auto.simulator.oracle import adjudicate, trap_families

CFG = load_carrier_config()
US = CFG.jurisdictions["US"]
UK = CFG.jurisdictions["UK_EU"]


def claim(**kw: object) -> dict:
    base = {
        "cause": "collision_vehicle",
        "coverage": "collision_comprehensive",
        "collision_deductible": 500.0,
        "comprehensive_deductible": 250.0,
        "driver_role": "named_insured",
        "use_at_loss": "personal",
        "rideshare_endorsement": False,
        "police_report": True,
        "police_report_hours": 2.0,
        "notice_days": 3,
        "gt_is_fraud": False,
        "vehicle_acv": 20000.0,
        "gt_true_damage": 3000.0,
    }
    return base | kw


def run(*claims: dict, jur=US) -> pd.DataFrame:
    return adjudicate(pd.DataFrame(list(claims)), jur)


def test_clean_collision_is_approved_net_of_deductible() -> None:
    row = run(claim()).iloc[0]
    assert (row.gt_decision, row.gt_coverage_part, row.gt_payout) == (
        "approve",
        "collision",
        2500.0,
    )
    assert row.gt_reasons == ""


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"coverage": "liability_only"}, "no_physical_damage_coverage"),
        ({"driver_role": "excluded_driver"}, "excluded_driver"),
        ({"cause": "mechanical_breakdown"}, "wear_and_tear_mechanical"),
        ({"use_at_loss": "delivery_active"}, "business_use_exclusion"),
        ({"cause": "glass", "coverage": "collision"}, "no_comprehensive_coverage"),
        (
            {"cause": "hit_and_run", "police_report": False, "police_report_hours": np.nan},
            "hit_and_run_report_condition",
        ),
        ({"cause": "hit_and_run", "police_report_hours": 30.0}, "hit_and_run_report_condition"),
    ],
)
def test_each_denial_rule(changes: dict, reason: str) -> None:
    row = run(claim(**changes)).iloc[0]
    assert row.gt_decision == "deny"
    assert reason in row.gt_reasons.split(";")
    assert np.isnan(row.gt_payout)


def test_permissive_unlisted_driver_is_covered() -> None:
    assert run(claim(driver_role="permissive_unlisted")).iloc[0].gt_decision == "approve"


def test_rideshare_with_endorsement_is_covered() -> None:
    row = run(claim(use_at_loss="rideshare_active", rideshare_endorsement=True)).iloc[0]
    assert row.gt_decision == "approve"


def test_timely_hit_and_run_is_covered() -> None:
    row = run(claim(cause="hit_and_run", police_report_hours=20.0)).iloc[0]
    assert (row.gt_decision, row.gt_coverage_part) == ("approve", "collision")


def test_deer_jurisdiction_difference() -> None:
    deer = claim(cause="animal", coverage="collision", comprehensive_deductible=np.nan)
    us, uk = run(deer).iloc[0], run(deer, jur=UK).iloc[0]
    assert (us.gt_decision, us.gt_reasons) == ("deny", "no_comprehensive_coverage")
    assert (uk.gt_decision, uk.gt_coverage_part) == ("approve", "collision")


def test_deer_with_comprehensive_uses_comprehensive_deductible() -> None:
    row = run(claim(cause="animal")).iloc[0]
    assert (row.gt_coverage_part, row.gt_deductible, row.gt_payout) == (
        "comprehensive",
        250.0,
        2750.0,
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"notice_days": 45}, "late_notice_prejudice_review"),
        ({"gt_is_fraud": True}, "suspected_fraud_siu"),
    ],
)
def test_escalations(changes: dict, reason: str) -> None:
    row = run(claim(**changes)).iloc[0]
    assert (row.gt_decision, row.gt_reasons) == ("escalate", reason)


def test_denial_takes_precedence_over_escalation() -> None:
    row = run(claim(coverage="liability_only", gt_is_fraud=True, notice_days=60)).iloc[0]
    assert row.gt_decision == "deny"
    assert row.gt_reasons == "no_physical_damage_coverage"


def test_multiple_denial_reasons_are_all_listed() -> None:
    row = run(claim(driver_role="excluded_driver", use_at_loss="delivery_active")).iloc[0]
    assert set(row.gt_reasons.split(";")) == {"excluded_driver", "business_use_exclusion"}


def test_total_loss_pays_vehicle_value() -> None:
    row = run(claim(gt_true_damage=16000.0)).iloc[0]  # >= 75% of 20,000
    assert row.gt_total_loss
    assert row.gt_payout == 19500.0


def test_total_loss_threshold_is_jurisdictional() -> None:
    c = claim(gt_true_damage=14500.0)  # 72.5% of ACV: total in UK_EU (70%), not in US (75%)
    assert not run(c).iloc[0].gt_total_loss
    assert run(c, jur=UK).iloc[0].gt_total_loss


def test_below_deductible_is_denied() -> None:
    row = run(claim(gt_true_damage=400.0)).iloc[0]
    assert (row.gt_decision, row.gt_reasons) == ("deny", "below_deductible")


def test_trap_families() -> None:
    frame = pd.DataFrame(
        [
            claim(cause="animal"),
            claim(use_at_loss="rideshare_active", driver_role="permissive_unlisted"),
            claim(cause="hit_and_run", notice_days=40),
            claim(),
        ]
    )
    fam = trap_families(frame, US)
    assert fam[0] == "animal_strike"
    assert set(fam[1].split(";")) == {"business_use", "driver_status"}
    assert set(fam[2].split(";")) == {"hit_and_run", "late_notice"}
    assert fam[3] == ""


def test_index_is_preserved() -> None:
    frame = pd.DataFrame([claim(), claim()], index=[10, 20])
    assert list(adjudicate(frame, US).index) == [10, 20]
