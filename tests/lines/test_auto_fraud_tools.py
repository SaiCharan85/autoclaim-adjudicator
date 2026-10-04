from pathlib import Path

import pandas as pd
import pytest

from autoclaim.core.rules import RuleSet
from autoclaim.lines.auto.fraud_tools import RULES_PATH, FraudSignals, FraudToolkit
from autoclaim.ml import fraud_model
from autoclaim.ml.features import clean


@pytest.fixture
def toolkit(trained_artifacts) -> FraudToolkit:
    return FraudToolkit(trained_artifacts, RuleSet.from_yaml(RULES_PATH), k_reasons=4)


def _record(fraud_frame: pd.DataFrame, **changes: object) -> dict:
    return fraud_frame.iloc[0].to_dict() | changes


def test_assess_returns_typed_signals(toolkit: FraudToolkit, fraud_frame: pd.DataFrame) -> None:
    sig = toolkit.assess(_record(fraud_frame))
    assert isinstance(sig, FraudSignals)
    assert 0 <= sig.model_score <= 1
    assert 0 <= sig.anomaly_score <= 1
    assert len(sig.top_reasons) <= 4
    assert sig.model_version == toolkit.artifacts.version
    assert sig.rules_unchecked == []


def test_model_score_is_the_tool_number(toolkit: FraudToolkit, fraud_frame: pd.DataFrame) -> None:
    rec = _record(fraud_frame)
    expected = toolkit.artifacts.predict_proba(clean(pd.DataFrame([rec]))).iat[0]
    assert toolkit.assess(rec).model_score == pytest.approx(expected, abs=1e-6)


def test_raw_dates_are_cleaned_so_lag_rule_fires(
    toolkit: FraudToolkit, fraud_frame: pd.DataFrame
) -> None:
    rec = _record(fraud_frame, Month="Jan", WeekOfMonth=1, MonthClaimed="Apr", WeekOfMonthClaimed=1)
    fired = {h.rule_id for h in toolkit.assess(rec).rules_fired}
    assert "delayed_reporting" in fired


def test_red_flags_raise_rule_score(toolkit: FraudToolkit, fraud_frame: pd.DataFrame) -> None:
    calm = _record(
        fraud_frame,
        Days_Policy_Accident="more than 30",
        Days_Policy_Claim="more than 30",
        AddressChange_Claim="no change",
        Fault="Third Party",
        AccidentArea="Urban",
        VehiclePrice="20000 to 29000",
        Month="Jan",
        MonthClaimed="Jan",
        WeekOfMonth=1,
        WeekOfMonthClaimed=1,
    )
    flagged = calm | {"Days_Policy_Accident": "none", "AddressChange_Claim": "under 6 months"}
    calm_sig, flagged_sig = toolkit.assess(calm), toolkit.assess(flagged)
    assert calm_sig.rule_score == 0
    assert {h.rule_id for h in flagged_sig.rules_fired} >= {
        "loss_soon_after_inception",
        "recent_address_change",
    }
    assert flagged_sig.rule_score > 0.5


def test_batch_matches_single(toolkit: FraudToolkit, fraud_frame: pd.DataFrame) -> None:
    records = [r for _, r in fraud_frame.head(6).iterrows()]
    batch = toolkit.assess_many([r.to_dict() for r in records])
    singles = [toolkit.assess(r.to_dict()) for r in records]
    assert [b.model_score for b in batch] == pytest.approx([s.model_score for s in singles])
    assert [b.rule_score for b in batch] == [s.rule_score for s in singles]


def test_missing_fields_are_tolerated_and_reported(toolkit: FraudToolkit) -> None:
    sig = toolkit.assess({"Fault": "Policy Holder", "AccidentArea": "Rural"})
    assert 0 <= sig.model_score <= 1
    assert "delayed_reporting" in sig.rules_unchecked
    assert "loss_soon_after_inception" in sig.rules_unchecked


def test_load_from_disk(trained_artifacts, tmp_path: Path, fraud_frame: pd.DataFrame) -> None:
    fraud_model.save(trained_artifacts, tmp_path)
    tk = FraudToolkit.load(model_dir=tmp_path)
    assert tk.assess(_record(fraud_frame)).model_version == trained_artifacts.version
