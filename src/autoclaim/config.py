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
    embedding_model: str
    query_prefix: str = ""  # bge v1.5 retrieval instruction (optional for that model)
    rrf_k: int = Field(default=60, gt=0)
    candidates: int = Field(default=20, gt=0)  # per ranker, before fusion
    top_k: int = Field(default=5, gt=0)
    graph_hops: int = Field(default=1, ge=0)
    max_expanded: int = Field(default=6, ge=0)


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


class TriageConfig(BaseModel):
    """Two-stage fraud triage (docs/fraud_triage.md). The threshold is fitted on validation and
    frozen here before the locked test is touched."""

    stage2_dataset: str
    stage1_budget: float = Field(gt=0, lt=1)
    target_true_recall: float = Field(gt=0, le=1)
    stage2_threshold: float | None = None  # None until fitted on validation


class FraudModelConfig(BaseModel):
    production_dataset: str
    review_budget: float = Field(gt=0, lt=1)
    seeds: list[int] = Field(min_length=1)
    siu_review_cost_usd: float = Field(ge=0)
    datasets: dict[str, DatasetSplit]
    triage: TriageConfig | None = None

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


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str | None  # None = no key (local Ollama)


class ModelLimits(BaseModel):
    """A model's family (judge must differ from adjudicator) and free-tier limits (None = none)."""

    family: str
    rpm: int | None = Field(default=None, gt=0)
    rpd: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    tpd: int | None = Field(default=None, gt=0)
    efforts: list[str] = []  # reasoning_effort values it accepts; others are not sent


class RoleConfig(BaseModel):
    chain: list[str] = Field(min_length=1)  # "<provider>:<model id>", tried in order
    max_tokens: int = Field(gt=0)
    reasoning_effort: str | None = None


class ModelsConfig(BaseModel):
    safety_margin: float = Field(gt=0, le=1)
    day_reset_utc_offset_hours: int = Field(ge=-12, le=14)
    timeout_s: float = Field(gt=0)
    cache_path: str
    usage_path: str
    providers: dict[str, ProviderConfig]
    catalog: dict[str, ModelLimits]
    roles: dict[str, RoleConfig]

    @model_validator(mode="after")
    def _chains_resolve(self) -> "ModelsConfig":
        for role, rc in self.roles.items():
            for key in rc.chain:
                provider, sep, _ = key.partition(":")
                if not sep or provider not in self.providers:
                    raise ValueError(f"role {role!r}: {key!r} has no known provider prefix")
                if key not in self.catalog:
                    raise ValueError(f"role {role!r}: {key!r} is not in the model catalog")
        return self


class MemoryConfig(BaseModel):
    """Feedback memory (core/memory.py). Few-shot stays off (k = 0) until measured to help."""

    enabled: bool = True
    few_shot_k: int = Field(default=0, ge=0)
    cutoff: date  # never remember cases on or after this date (the evaluation period)
    path: str = ".cache/memory/feedback.sqlite3"


class CarrierConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # sections typed in later steps pass through

    carrier: dict[str, Any]
    retrieval: RetrievalConfig
    harness: HarnessConfig
    fraud_model: FraudModelConfig
    jurisdiction: str
    jurisdictions: dict[str, JurisdictionProfile]
    models: ModelsConfig
    memory: MemoryConfig | None = None

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
