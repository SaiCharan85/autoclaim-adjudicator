from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from autoclaim.config import DEFAULT_CONFIG_PATH, FraudModelConfig, load_carrier_config


def test_real_config_loads_typed_sections() -> None:
    cfg = load_carrier_config()
    assert cfg.retrieval.hnsw.M == 32
    assert cfg.harness.max_retries == 2
    assert cfg.fraud_model.train_years == [1994, 1995]
    assert cfg.fraud_model.test_years == [1996]
    assert cfg.fraud_model.review_budget == 0.05
    assert set(cfg.fraud_model.sensitive_features) == {"Sex", "MaritalStatus"}


def test_untyped_sections_pass_through() -> None:
    cfg = load_carrier_config()
    assert cfg.model_extra is not None
    assert "models" in cfg.model_extra


def test_custom_path(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["harness"]["seed"] = 7
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_carrier_config(path).harness.seed == 7


def test_overlapping_years_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        FraudModelConfig(
            train_years=[1994, 1995], test_years=[1995], review_budget=0.05, cv_folds=5
        )


@pytest.mark.parametrize("budget", [0, 1, 1.5, -0.1])
def test_review_budget_must_be_a_fraction(budget: float) -> None:
    with pytest.raises(ValidationError):
        FraudModelConfig(train_years=[1994], test_years=[1996], review_budget=budget, cv_folds=5)


def test_missing_required_section_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["fraud_model"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carrier_config(path)
