"""Typed world parameters for the grounded-hybrid claims builder (config/simulator_auto.yaml)."""

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from autoclaim.paths import REPO_ROOT

DEFAULT_WORLD_PATH = REPO_ROOT / "config" / "simulator_auto.yaml"

Dist = dict[str, float]

# US Census regions (the only location CRSS publishes) for the states we simulate.
STATE_REGION = {
    "NY": "northeast", "PA": "northeast", "NJ": "northeast",
    "IL": "midwest", "OH": "midwest", "MI": "midwest", "MN": "midwest",
    "TX": "south", "FL": "south", "GA": "south", "NC": "south",
    "CA": "west", "AZ": "west", "WA": "west", "CO": "west",
}  # fmt: skip


def _check_dist(d: dict[str, float] | dict[int, float], name: str) -> None:
    if not d or any(v < 0 for v in d.values()):
        raise ValueError(f"{name}: needs non-negative weights")
    if abs(sum(d.values()) - 1) > 1e-6:
        raise ValueError(f"{name}: weights sum to {sum(d.values()):.4f}, expected 1")


class LogNormal(BaseModel):
    median: float = Field(gt=0)
    sigma: float = Field(ge=0)


class FraudWorld(BaseModel):
    base_rate: float = Field(gt=0, lt=1)
    coverage_multiplier: Dist
    prior_damage_share: float = Field(ge=0, le=1)
    staged_share: float = Field(ge=0, le=1)
    detection_rate: float = Field(gt=0, le=1)
    early_inception_share: float = Field(ge=0, le=1)
    legit_early_inception_share: float = Field(ge=0, le=1)
    profile_multiplier: dict[str, float]
    inflation_factor: tuple[float, float]
    late_notice_share: float = Field(ge=0, le=1)
    address_change_recent_share: float = Field(ge=0, le=1)
    legit_address_change_recent_share: float = Field(ge=0, le=1)


class TheftRecovery(BaseModel):
    rate: float = Field(ge=0, le=1)  # share of stolen vehicles recovered (then repaired)
    damage_fraction: tuple[float, float]  # lognormal median, sigma of a recovered car's damage


class AppraisalWorld(BaseModel):
    log_bias: float
    sigma: float = Field(ge=0)
    shop_sigma: float = Field(default=0.0, ge=0)  # honest shop error, folded into the gap
    prior_damage_detect: float = Field(ge=0, le=1)
    prior_damage_false_flag: float = Field(ge=0, le=1)
    seed_offset: int


class HoldoutSpec(BaseModel):
    seed: int
    n_claims: int = Field(gt=0)
    overrides: dict[str, Any] = {}


class Holdouts(BaseModel):
    test_window_start: date
    fresh: HoldoutSpec
    shift: HoldoutSpec


class World(BaseModel):
    seed: int
    n_claims: int = Field(gt=0)
    loss_date_start: date
    loss_date_end: date
    states: Dist
    out_of_state_loss_rate: float = Field(ge=0, le=1)
    coverage_mix: Dist
    collision_deductibles: dict[int, float]
    comprehensive_deductibles: dict[int, float]
    rideshare_endorsement_rate: float = Field(ge=0, le=1)
    policy_term_days: int = Field(gt=0)
    driver_role: Dist
    driver_age: dict[str, float]
    vehicle_age_years: dict[str, float]
    new_vehicle_price: LogNormal
    depreciation_per_year: float = Field(ge=0, lt=1)
    min_acv: float = Field(gt=0)
    financed_rate_by_age: dict[str, float]
    adas_rate_by_age: dict[str, float]
    kia_hyundai_theft_multiplier: dict[str, float]
    loss_causes: Dist
    use_at_loss: Dist
    damage_fraction: dict[str, tuple[float, float]]
    damage_by_extent: dict[str, tuple[float, float]]
    towed_damage_multiplier: float = Field(ge=1)
    body_new_price: dict[str, float]
    luxury_makes: list[str]
    luxury_multiplier: float = Field(ge=1)
    adas_glass_recalibration_usd: float = Field(ge=0)
    theft_recovery: TheftRecovery
    police_report_rate: dict[str, float]
    police_report_hours: LogNormal
    witness_rate: dict[str, float]
    notice_days: LogNormal
    injury_rate: dict[str, float]
    at_fault_rate: dict[str, float]
    fraud: FraudWorld
    appraisal: AppraisalWorld
    traps: dict[str, float]
    holdouts: Holdouts | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "World":
        unknown = set(self.states) - set(STATE_REGION)
        if unknown:
            raise ValueError(f"states without a census region: {sorted(unknown)}")
        if set(STATE_REGION.values()) - {STATE_REGION[s] for s in self.states}:
            raise ValueError("every census region needs at least one state")
        for name in (
            "coverage_mix",
            "collision_deductibles",
            "comprehensive_deductibles",
            "driver_role",
            "loss_causes",
            "use_at_loss",
        ):
            _check_dist(getattr(self, name), name)
        if self.loss_date_end <= self.loss_date_start:
            raise ValueError("loss_date_end must be after loss_date_start")
        causes = set(self.loss_causes)
        for name in ("damage_fraction", "police_report_rate"):
            missing = causes - set(getattr(self, name))
            if missing:
                raise ValueError(f"{name} lacks causes: {sorted(missing)}")
        if sum(self.traps.values()) >= 0.5:
            raise ValueError("trap rates must leave most claims ordinary")
        return self


def with_overrides(world: World, overrides: dict[str, Any]) -> World:
    """Copy of `world` with dotted-key overrides, e.g. {"fraud.base_rate": 0.1}; re-validated."""
    data = world.model_dump()
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = data
        for key in parents:
            if key not in node:
                raise KeyError(f"unknown world setting {dotted!r}")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"unknown world setting {dotted!r}")
        node[leaf] = value
    return World.model_validate(data)


def load_world(path: Path | None = None) -> World:
    return World.model_validate(yaml.safe_load((path or DEFAULT_WORLD_PATH).read_text("utf-8")))
