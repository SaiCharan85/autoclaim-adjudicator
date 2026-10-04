"""Typed access to config/carrier_config.yaml.

Sections are typed as the steps that use them land.
"""

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


class FraudModelConfig(BaseModel):
    train_years: list[int]
    test_years: list[int]
    review_budget: float = Field(gt=0, lt=1)
    cv_folds: int = Field(ge=2)
    sensitive_features: list[str] = []

    @model_validator(mode="after")
    def _years_disjoint(self) -> "FraudModelConfig":
        if set(self.train_years) & set(self.test_years):
            raise ValueError("train_years and test_years must not overlap")
        return self


class CarrierConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # sections typed in later steps pass through

    carrier: dict[str, Any]
    retrieval: RetrievalConfig
    harness: HarnessConfig
    fraud_model: FraudModelConfig


def load_carrier_config(path: Path | None = None) -> CarrierConfig:
    raw = yaml.safe_load((path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    return CarrierConfig.model_validate(raw)
