"""Typed access to config/carrier_config.yaml.

Sections are typed as the steps that use them land.
"""

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoclaim.paths import REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "carrier_config.yaml"


class HNSWConfig(BaseModel):
    M: int = Field(gt=0)
    ef_construction: int = Field(gt=0)
    ef_search: int = Field(gt=0)


class RetrievalConfig(BaseModel):
    hnsw: HNSWConfig


class HarnessConfig(BaseModel):
    max_retries: int = Field(ge=0)
    seed: int


class DatasetSplit(BaseModel):
    """Either date boundaries (simulator) or year lists + validation fraction (1990s data)."""

    val_start: date | None = None
    test_start: date | None = None
    train_years: list[int] | None = None
    test_years: list[int] | None = None
    val_fraction: float | None = Field(default=None, gt=0, lt=1)
    sensitive_features: list[str] = []

    @model_validator(mode="after")
    def _one_scheme(self) -> "DatasetSplit":
        by_date = self.val_start is not None and self.test_start is not None
        by_year = self.train_years is not None and self.test_years is not None
        if by_date == by_year:
            raise ValueError("give either val_start+test_start or train_years+test_years")
        if by_date and self.val_start >= self.test_start:  # type: ignore[operator]
            raise ValueError("val_start must be before test_start")
        if by_year:
            if set(self.train_years or []) & set(self.test_years or []):
                raise ValueError("train_years and test_years must not overlap")
            if self.val_fraction is None:
                raise ValueError("year-based splits need val_fraction")
        return self


class FraudModelConfig(BaseModel):
    production_dataset: str
    review_budget: float = Field(gt=0, lt=1)
    seeds: list[int] = Field(min_length=1)
    siu_review_cost_usd: float = Field(ge=0)
    datasets: dict[str, DatasetSplit]

    @model_validator(mode="after")
    def _checks(self) -> "FraudModelConfig":
        if self.production_dataset not in self.datasets:
            raise ValueError(f"production_dataset {self.production_dataset!r} has no split config")
        return self


class JurisdictionProfile(BaseModel):
    """Rules that differ by jurisdiction; the policy's common core does not."""

    terms: dict[str, str]
    comprehensive_causes: list[str]  # causes of loss paid by comprehensive, not collision
    hit_and_run_police_report_hours: float = Field(gt=0)
    late_notice_days: int = Field(gt=0)
    total_loss_threshold: float = Field(gt=0, le=1)


class CarrierConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # sections typed in later steps pass through

    carrier: dict[str, Any]
    retrieval: RetrievalConfig
    harness: HarnessConfig
    fraud_model: FraudModelConfig
    jurisdiction: str
    jurisdictions: dict[str, JurisdictionProfile]

    @model_validator(mode="after")
    def _known_jurisdiction(self) -> "CarrierConfig":
        if self.jurisdiction not in self.jurisdictions:
            raise ValueError(f"jurisdiction {self.jurisdiction!r} has no profile")
        return self

    @property
    def active_jurisdiction(self) -> JurisdictionProfile:
        return self.jurisdictions[self.jurisdiction]


def load_carrier_config(path: Path | None = None) -> CarrierConfig:
    raw = yaml.safe_load((path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    return CarrierConfig.model_validate(raw)
