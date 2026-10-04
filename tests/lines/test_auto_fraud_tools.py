from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from autoclaim.core.rules import RuleSet
from autoclaim.lines.auto import fraud_tools
from autoclaim.lines.auto.fraud_tools import RULES_PATH, FraudSignals, FraudToolkit
from autoclaim.ml import fraud_model


@pytest.fixture
def toolkit(trained_artifacts, legacy_spec) -> FraudToolkit:
    return FraudToolkit(trained_artifacts, RuleSet.from_yaml(RULES_PATH), legacy_spec, k_reasons=4)


def _record(fraud_frame: pd.DataFrame, **changes: object) -> dict:
    return fraud_frame.iloc[0].to_dict() | changes


def test_assess_returns_typed_signals(toolkit: FraudToolkit, fraud_frame: pd.DataFrame) -> None:
    sig = toolkit.assess(_record(fraud_frame))
    assert isinstance(sig, FraudSignals)
    assert 0 <= sig.model_score <= 1
    assert 0 <= sig.anomaly_score <= 1
    assert len(sig.top_reasons) <= 4
    assert sig.model_version == toolkit.artifacts.version


def test_model_score_is_the_tool_number(
    toolkit: FraudToolkit, legacy_spec, fraud_frame: pd.DataFrame
) -> None:
    rec = _record(fraud_frame)
    features = legacy_spec.features_frame(pd.DataFrame([rec]))
    expected = toolkit.artifacts.predict_proba(features).iat[0]
    assert toolkit.assess(rec).model_score == pytest.approx(expected, abs=1e-6)


def test_canonical_rules_run_on_raw_legacy_rows(
    toolkit: FraudToolkit, fraud_frame: pd.DataFrame
) -> None:
    rec = _record(
        fraud_frame,
        Days_Policy_Accident="1 to 7",
        Month="Jan",
        WeekOfMonth=1,
        MonthClaimed="Apr",
        WeekOfMonthClaimed=1,
    )
    fired = {h.rule_id for h in toolkit.assess(rec).rules_fired}
    assert {"loss_soon_after_inception", "delayed_reporting"} <= fired


def test_unavailable_facts_are_reported_not_hidden(
    toolkit: FraudToolkit, fraud_frame: pd.DataFrame
) -> None:
    sig = toolkit.assess(_record(fraud_frame))
    assert "financed_theft_reported_late" in sig.rules_unchecked  # 1990s data has no such field
    assert "financed_theft_reported_late" not in {h.rule_id for h in sig.rules_fired}


def test_batch_matches_single(toolkit: FraudToolkit, fraud_frame: pd.DataFrame) -> None:
    records = [r.to_dict() for _, r in fraud_frame.head(6).iterrows()]
    batch = toolkit.assess_many(records)
    singles = [toolkit.assess(r) for r in records]
    assert [b.model_score for b in batch] == pytest.approx([s.model_score for s in singles])
    assert [b.rule_score for b in batch] == [s.rule_score for s in singles]


def test_simulator_rows_partial_record(trained_artifacts) -> None:
    """Production spec tolerates partial FNOL records: features become NaN, rules unchecked."""
    from autoclaim.config import load_carrier_config
    from autoclaim.lines.auto.datasets import get_spec

    sim = get_spec("sim_us", load_carrier_config().fraud_model)
    frame = sim.features_frame(pd.DataFrame([{"cause": "theft", "financed": True}]))
    assert frame.loc[0, "financed"] == 1.0
    assert pd.isna(frame.loc[0, "days_policy_to_loss"])


def test_load_uses_the_artifact_dataset(
    trained_artifacts, tmp_path: Path, fraud_frame: pd.DataFrame, monkeypatch, legacy_spec
) -> None:
    fraud_model.save(trained_artifacts, tmp_path)
    monkeypatch.setattr(fraud_tools, "get_spec", lambda name, cfg: replace(legacy_spec, name=name))
    tk = FraudToolkit.load(model_dir=tmp_path)
    assert tk.spec.name == "legacy_1990s"
    assert tk.assess(_record(fraud_frame)).model_version == trained_artifacts.version
