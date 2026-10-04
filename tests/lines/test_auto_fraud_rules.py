"""The auto red-flag rule file: valid, uses real feature names, and fires on the intended cases."""

import pandas as pd
import pytest

from autoclaim.core.rules import RuleSet
from autoclaim.lines.auto.fraud_tools import RULES_PATH
from autoclaim.ml.features import clean


@pytest.fixture(scope="module")
def rules() -> RuleSet:
    return RuleSet.from_yaml(RULES_PATH)


def test_rules_file_loads(rules: RuleSet) -> None:
    assert rules.name == "auto_physical_damage_red_flags"
    assert len(rules.rules) >= 5


def test_every_rule_field_exists_after_cleaning(rules: RuleSet, valid_frame: pd.DataFrame) -> None:
    columns = set(clean(valid_frame).columns)
    assert rules.missing_fields(columns) == {}


def _base() -> dict[str, object]:
    return {
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "AddressChange_Claim": "no change",
        "Fault": "Third Party",
        "PoliceReportFiled": "Yes",
        "WitnessPresent": "Yes",
        "AccidentArea": "Urban",
        "VehiclePrice": "20000 to 29000",
        "AgeOfVehicle": "3 years",
        "claim_lag_weeks": 0.0,
    }


def test_clean_record_fires_nothing(rules: RuleSet) -> None:
    fired, score = rules.evaluate_record(_base())
    assert fired == []
    assert score == 0.0


@pytest.mark.parametrize(
    ("changes", "rule_id"),
    [
        ({"Days_Policy_Accident": "none"}, "loss_soon_after_inception"),
        ({"Days_Policy_Claim": "8 to 15"}, "claim_soon_after_inception"),
        ({"AddressChange_Claim": "under 6 months"}, "recent_address_change"),
        (
            {"Fault": "Policy Holder", "PoliceReportFiled": "No", "WitnessPresent": "No"},
            "unverified_at_fault_loss",
        ),
        (
            {"AccidentArea": "Rural", "PoliceReportFiled": "No", "WitnessPresent": "No"},
            "isolated_unverified_loss",
        ),
        ({"VehiclePrice": "more than 69000", "AgeOfVehicle": "7 years"}, "expensive_older_vehicle"),
        ({"claim_lag_weeks": 6.0}, "delayed_reporting"),
    ],
)
def test_each_rule_fires_on_its_case(rules: RuleSet, changes: dict, rule_id: str) -> None:
    fired, score = rules.evaluate_record(_base() | changes)
    assert rule_id in {h.rule_id for h in fired}
    assert 0 < score <= 1


def test_partial_conditions_do_not_fire(rules: RuleSet) -> None:
    fired, _ = rules.evaluate_record(
        _base() | {"Fault": "Policy Holder", "PoliceReportFiled": "No"}
    )
    assert "unverified_at_fault_loss" not in {h.rule_id for h in fired}
