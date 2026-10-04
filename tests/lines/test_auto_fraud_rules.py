"""The auto red-flag rule file: valid, uses the shared vocabulary, fires on the intended cases."""

import pandas as pd
import pytest

from autoclaim.core.rules import RuleSet
from autoclaim.lines.auto.datasets import legacy_canonical, sim_features
from autoclaim.lines.auto.fraud_tools import RULES_PATH
from autoclaim.lines.auto.simulator.columns import Availability, availability


@pytest.fixture(scope="module")
def rules() -> RuleSet:
    return RuleSet.from_yaml(RULES_PATH)


def test_rules_file_loads(rules: RuleSet) -> None:
    assert rules.name == "auto_physical_damage_red_flags"
    assert rules.version == "2.0"
    assert len(rules.rules) >= 10


def test_rules_use_only_first_notice_fields(rules: RuleSet) -> None:
    for rule in rules.rules:
        for f in rule.fields:
            assert availability(f) is Availability.FNOL, (rule.id, f)


def test_simulator_features_cover_every_rule(rules: RuleSet) -> None:
    columns = set(sim_features(pd.DataFrame(index=[0])).columns)
    assert rules.missing_fields(columns) == {}


def test_legacy_mapping_covers_the_rules_it_can(rules: RuleSet, valid_frame: pd.DataFrame) -> None:
    covered = set(legacy_canonical(valid_frame).columns)
    unchecked = set(rules.missing_fields(covered))
    assert "loss_soon_after_inception" not in unchecked
    assert "delayed_reporting" not in unchecked
    assert "financed_theft_reported_late" in unchecked  # no theft/financing fields in 1990s data


def _base() -> dict[str, object]:
    return {
        "days_policy_to_loss": 400.0,
        "days_policy_to_claim": 402.0,
        "address_change_days": float("nan"),
        "at_fault": 0.0,
        "police_report": 1.0,
        "witness_count": 1.0,
        "area": "urban",
        "claim_to_acv": 0.1,
        "cause": "collision_vehicle",
        "notice_days": 2.0,
        "loss_hour": 14.0,
        "out_of_state": 0.0,
        "financed": 0.0,
        "police_report_hours": 1.0,
        "attorney_involved": 0.0,
        "injury_count": 0.0,
    }


def test_clean_record_fires_nothing(rules: RuleSet) -> None:
    fired, score = rules.evaluate_record(_base())
    assert fired == []
    assert score == 0.0


@pytest.mark.parametrize(
    ("changes", "rule_id"),
    [
        ({"days_policy_to_loss": 3.0}, "loss_soon_after_inception"),
        ({"days_policy_to_claim": 10.0}, "claim_soon_after_inception"),
        ({"address_change_days": 60.0}, "recent_address_change"),
        ({"at_fault": 1.0, "police_report": 0.0, "witness_count": 0.0}, "unverified_at_fault_loss"),
        ({"area": "rural", "police_report": 0.0, "witness_count": 0.0}, "isolated_unverified_loss"),
        ({"claim_to_acv": 0.8}, "claim_near_vehicle_value"),
        ({"notice_days": 40.0}, "delayed_reporting"),
        (
            {"loss_hour": 2.0, "witness_count": 0.0, "police_report": 0.0},
            "late_night_unwitnessed_loss",
        ),
        ({"out_of_state": 1.0}, "out_of_state_loss"),
        (
            {"cause": "theft", "financed": 1.0, "police_report_hours": 48.0},
            "financed_theft_reported_late",
        ),
        ({"attorney_involved": 1.0, "injury_count": 2.0}, "early_attorney_on_injury_claim"),
    ],
)
def test_each_rule_fires_on_its_case(rules: RuleSet, changes: dict, rule_id: str) -> None:
    fired, score = rules.evaluate_record(_base() | changes)
    assert rule_id in {h.rule_id for h in fired}
    assert 0 < score <= 1


def test_theft_is_not_a_near_value_red_flag(rules: RuleSet) -> None:
    fired, _ = rules.evaluate_record(_base() | {"claim_to_acv": 1.0, "cause": "theft"})
    assert "claim_near_vehicle_value" not in {h.rule_id for h in fired}


def test_partial_conditions_do_not_fire(rules: RuleSet) -> None:
    fired, _ = rules.evaluate_record(_base() | {"at_fault": 1.0, "police_report": 0.0})
    assert "unverified_at_fault_loss" not in {h.rule_id for h in fired}
