from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from autoclaim.config import (
    DEFAULT_CONFIG_PATH,
    DatasetSplit,
    FraudModelConfig,
    load_carrier_config,
)


def test_real_config_loads_typed_sections() -> None:
    cfg = load_carrier_config()
    assert cfg.retrieval.hnsw.M == 32
    assert cfg.harness.max_retries == 2
    fm = cfg.fraud_model
    assert fm.production_dataset == "sim_us"
    assert fm.review_budget == 0.05
    assert len(fm.seeds) == 3
    assert set(fm.datasets) == {"sim_us", "legacy_1990s"}
    assert set(fm.datasets["legacy_1990s"].sensitive_features) == {"Sex", "MaritalStatus"}
    assert cfg.active_jurisdiction.late_notice_days == 30


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


def test_split_needs_exactly_one_scheme() -> None:
    with pytest.raises(ValidationError, match="either"):
        DatasetSplit(val_start=date(2024, 1, 1))
    with pytest.raises(ValidationError, match="either"):
        DatasetSplit(
            val_start=date(2024, 1, 1), test_start=date(2024, 7, 1),
            train_years=[1994], test_years=[1996], val_fraction=0.2,
        )  # fmt: skip


def test_split_rules() -> None:
    with pytest.raises(ValidationError, match="before"):
        DatasetSplit(val_start=date(2024, 7, 1), test_start=date(2024, 1, 1))
    with pytest.raises(ValidationError, match="overlap"):
        DatasetSplit(train_years=[1994, 1995], test_years=[1995], val_fraction=0.2)
    with pytest.raises(ValidationError, match="val_fraction"):
        DatasetSplit(train_years=[1994], test_years=[1996])


def _fm(**kw: object) -> dict:
    base = {
        "production_dataset": "d",
        "review_budget": 0.05,
        "seeds": [1],
        "siu_review_cost_usd": 300,
        "datasets": {"d": {"train_years": [1994], "test_years": [1996], "val_fraction": 0.2}},
    }
    return base | kw


@pytest.mark.parametrize("budget", [0, 1, 1.5, -0.1])
def test_review_budget_must_be_a_fraction(budget: float) -> None:
    with pytest.raises(ValidationError):
        FraudModelConfig.model_validate(_fm(review_budget=budget))


def test_production_dataset_must_exist() -> None:
    with pytest.raises(ValidationError, match="production_dataset"):
        FraudModelConfig.model_validate(_fm(production_dataset="nope"))


def test_needs_a_seed() -> None:
    with pytest.raises(ValidationError):
        FraudModelConfig.model_validate(_fm(seeds=[]))


def test_unknown_jurisdiction_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["jurisdiction"] = "MARS"
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="no profile"):
        load_carrier_config(path)


def test_missing_required_section_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["fraud_model"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_carrier_config(path)
