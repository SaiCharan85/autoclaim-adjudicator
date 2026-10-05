"""Auto claim facts: what intake extracts (LLM) and what code derives from it (deterministic).

Plain-English trace of the derivations, e.g. for a deer strike:
claim says deer -> cause animal -> comprehensive (US carrier config) -> policy has comprehensive?
-> comprehensive deductible $250 -> estimate $1,900 < 75% of ACV -> not a total loss
-> payout $1,900 - $250 = $1,650.
Every number the decision uses comes from here, never from an LLM.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from autoclaim.config import JurisdictionProfile
from autoclaim.lines.auto.claim import ClaimPackage

Cause = Literal[
    "animal", "collision_object", "collision_vehicle", "fire", "glass", "hail_weather",
    "hit_and_run", "mechanical_breakdown", "parked_hit", "theft", "vandalism", "unknown",
]  # fmt: skip
UseAtLoss = Literal["personal", "commute", "rideshare_active", "delivery_active", "unknown"]
Severity = Literal["none", "minor", "functional", "disabling", "unknown"]
DriverRole = Literal[
    "named_insured", "listed_driver", "excluded_driver", "permissive_unlisted", "unknown"
]
CoveragePart = Literal["collision", "comprehensive", "none", "unknown"]
BUSINESS_USES = ("rideshare_active", "delivery_active")


class ClaimFacts(BaseModel):
    """Intake output: only what the claimant's statement says (None = not stated)."""

    loss_date: date | None
    cause: Cause
    summary: str = Field(max_length=400, description="one or two neutral sentences")
    driver_name: str | None
    driver_relationship: str | None
    use_at_loss: UseAtLoss
    vehicle_parked: bool | None
    n_vehicles: int | None = Field(default=None, ge=1)
    injuries: int | None = Field(default=None, ge=0)
    damage_severity: Severity
    towed: bool | None
    other_driver_fled: bool | None
    insured_at_fault: bool | None
    police_report: bool | None
    police_report_hours: float | None = Field(default=None, ge=0)
    witnesses: int | None = Field(default=None, ge=0)
    attorney_involved: bool | None
    missing_info: list[str] = Field(description="decision-relevant facts the statement lacks")


class DerivedFacts(BaseModel):
    notice_days: int | None
    late_notice: bool
    driver_role: DriverRole
    coverage_part: CoveragePart
    part_on_policy: bool  # does the policy carry the coverage this loss falls under?
    deductible: float | None
    total_loss: bool
    gross_loss: float
    payout: float  # what an approval pays (0 = at or below the deductible)
    business_use_without_endorsement: bool
    hit_and_run_report_ok: bool | None  # None = not a hit-and-run


def _norm(name: str) -> str:
    return " ".join(name.lower().replace(".", " ").split())


def driver_role(name: str | None, pkg: ClaimPackage) -> DriverRole:
    """Look the driver up in the policy's driver lists (code, not judgment)."""
    if not name:
        return "unknown"
    n = _norm(name)
    pol = pkg.policy

    def match(candidate: str) -> bool:
        c = _norm(candidate)
        return n == c or (len(n.split()) == 1 and n == c.split()[0])  # first name only

    if any(match(x) for x in pol.excluded_drivers):
        return "excluded_driver"
    if match(pol.named_insured):
        return "named_insured"
    if any(match(x) for x in pol.listed_drivers):
        return "listed_driver"
    return "permissive_unlisted"


def coverage_part(cause: Cause, jur: JurisdictionProfile) -> CoveragePart:
    if cause == "unknown":
        return "unknown"
    if cause == "mechanical_breakdown":
        return "none"
    return "comprehensive" if cause in jur.comprehensive_causes else "collision"


def derive(facts: ClaimFacts, pkg: ClaimPackage, jur: JurisdictionProfile) -> DerivedFacts:
    pol = pkg.policy
    notice = (pkg.report_date - facts.loss_date).days if facts.loss_date else None
    part = coverage_part(facts.cause, jur)
    on_policy = {
        "collision": pol.coverage in ("collision", "collision_comprehensive"),
        "comprehensive": pol.coverage == "collision_comprehensive",
    }.get(part, False)
    deductible = {"collision": pol.collision_deductible,
                  "comprehensive": pol.comprehensive_deductible}.get(part)  # fmt: skip
    acv = pol.vehicle.actual_cash_value
    total = pkg.estimate_amount >= jur.total_loss_threshold * acv
    gross = acv if total else pkg.estimate_amount
    payout = round(max(gross - (deductible or 0.0), 0.0), 2)
    hit_and_run = facts.cause == "hit_and_run"
    report_ok = (
        bool(facts.police_report)
        and facts.police_report_hours is not None
        and facts.police_report_hours <= jur.hit_and_run_police_report_hours
    )
    return DerivedFacts(
        notice_days=notice,
        late_notice=notice is not None and notice > jur.late_notice_days,
        driver_role=driver_role(facts.driver_name, pkg),
        coverage_part=part,
        part_on_policy=on_policy,
        deductible=deductible,
        total_loss=total,
        gross_loss=round(gross, 2),
        payout=payout,
        business_use_without_endorsement=facts.use_at_loss in BUSINESS_USES
        and not pol.rideshare_endorsement,
        hit_and_run_report_ok=report_ok if hit_and_run else None,
    )
