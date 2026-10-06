"""Flood claims and their deterministic coverage math (code decides every number).

Plain-English trace of one real NFIP claim: tropical storm, building damage $6,948, building
deductible code "2" = $2,000, building limit $250,000 -> pays $6,948 - $2,000 = $4,948 (FEMA's
record shows $4,947.64 paid). Contents: no documented damage -> $0.
"""

from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

# FEMA data dictionary, buildingDeductibleCode / contentsDeductibleCode (checked 2026-10-05)
DEDUCTIBLE_CODES = {
    "0": 500.0, "1": 1_000.0, "2": 2_000.0, "3": 3_000.0, "4": 4_000.0, "5": 5_000.0,
    "9": 750.0, "A": 10_000.0, "B": 15_000.0, "C": 20_000.0, "D": 25_000.0, "E": 50_000.0,
    "F": 1_250.0, "G": 1_500.0, "H": 200.0,
}  # fmt: skip
# causeOfDamage codes that describe a flood, vs ones that need an adjuster's judgment
FLOOD_CAUSES = {"1": "tidal water overflow", "2": "river, stream or lake overflow",
                "3": "alluvial fan overflow", "4": "rainfall or snowmelt accumulation",
                "A": "closed basin lake"}  # fmt: skip
HANDLING_CODES = frozenset({"B", "C", "D"})  # expedited / remote adjustment: process, not cause
# Rules fixed 2026-10-05 from train/validation claims only (loss dates before 2024-07-01):
# - a blank damage amount means that part was not claimed (contents: 47 of 15,303 blanks paid)
# - a damage amount above $0 but under $10 is a placeholder (seen with payments up to $250,000)
# - handling codes alone (B/C/D) still mean a flood claim (the program only adjusts floods)
# - a loss dated before the record's policy start ESCALATES (the record's start date is not the
#   true inception for renewals/transfers: most such claims were in fact paid)
PLACEHOLDER_DAMAGE = 10.0
JUDGMENT_CAUSES = {"0": "other cause", "7": "erosion", "8": "erosion", "9": "earth movement"}

Part = Literal["building", "contents"]


class FloodClaim(BaseModel):
    """What the first notice + the adjuster's documented damage give us (structured)."""

    claim_id: str
    date_of_loss: date
    policy_start: date | None
    building_limit: float = Field(ge=0)
    contents_limit: float = Field(ge=0)
    building_damage: float | None = Field(default=None, ge=0)
    contents_damage: float | None = Field(default=None, ge=0)
    building_deductible_code: str | None
    contents_deductible_code: str | None
    cause_codes: list[str]
    flood_event: str | None = None
    flood_zone: str | None = None
    primary_residence: bool | None = None
    water_depth: float | None = None
    replacement_cost_basis: bool | None = None  # settled at replacement cost (ACV paid first)


def _num(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _code(v: Any) -> str | None:
    if v is None or pd.isna(v):
        return None
    text = str(v).strip()
    return text[:-2] if text.endswith(".0") else text


def from_nfip(row: pd.Series) -> FloodClaim:
    """A FEMA NFIP redacted claim as a FloodClaim (outcome columns are NOT read here)."""
    start = row.get("originalNBDate")
    causes = _code(row.get("causeOfDamage")) or ""
    return FloodClaim(
        claim_id=str(row["id"]),
        date_of_loss=pd.Timestamp(row["dateOfLoss"]).date(),
        policy_start=None if start is None or pd.isna(start) else pd.Timestamp(start).date(),
        building_limit=_num(row.get("totalBuildingInsuranceCoverage")) or 0.0,
        contents_limit=_num(row.get("totalContentsInsuranceCoverage")) or 0.0,
        building_damage=_num(row.get("buildingDamageAmount")),
        contents_damage=_num(row.get("contentsDamageAmount")),
        building_deductible_code=_code(row.get("buildingDeductibleCode")),
        contents_deductible_code=_code(row.get("contentsDeductibleCode")),
        cause_codes=sorted(set(causes)),
        flood_event=None if pd.isna(row.get("floodEvent")) else str(row.get("floodEvent")),
        flood_zone=None if pd.isna(row.get("ratedFloodZone")) else str(row.get("ratedFloodZone")),
        primary_residence=None if pd.isna(row.get("primaryResidenceIndicator"))
        else bool(row.get("primaryResidenceIndicator")),
        water_depth=_num(row.get("waterDepth")),
        replacement_cost_basis=None if pd.isna(row.get("replacementCostBasis"))
        else str(row.get("replacementCostBasis")) == "R",
    )  # fmt: skip


class PartResult(BaseModel):
    part: Part
    covered: bool  # the policy has this coverage (limit > 0)
    damage: float | None
    deductible: float | None
    payout: float


class FloodCoverage(BaseModel):
    building: PartResult
    contents: PartResult
    total: float
    before_inception: bool
    cause: str  # plain-English cause
    judgment_needed: list[str]  # why an adjuster must decide (empty = code can decide)
    clauses: list[str]  # the clauses that decided it


def _part(part: Part, limit: float, damage: float | None, code: str | None) -> PartResult:
    deductible = DEDUCTIBLE_CODES.get(code or "")
    covered = limit > 0
    payout = 0.0
    if covered and damage is not None and deductible is not None:
        payout = round(min(max(damage - deductible, 0.0), limit), 2)
    return PartResult(part=part, covered=covered, damage=damage, deductible=deductible,
                      payout=payout)  # fmt: skip


def assess(claim: FloodClaim) -> FloodCoverage:
    b = _part(
        "building", claim.building_limit, claim.building_damage, claim.building_deductible_code
    )
    c = _part(
        "contents", claim.contents_limit, claim.contents_damage, claim.contents_deductible_code
    )
    before = claim.policy_start is not None and claim.date_of_loss < claim.policy_start
    flood = [FLOOD_CAUSES[x] for x in claim.cause_codes if x in FLOOD_CAUSES]
    if not flood and claim.cause_codes and set(claim.cause_codes) <= HANDLING_CODES:
        flood = ["flood (cause not recorded; expedited handling)"]
    unclear = [JUDGMENT_CAUSES[x] for x in claim.cause_codes if x in JUDGMENT_CAUSES]
    judgment = []
    if not flood:
        judgment.append("cause_not_clearly_flood" + (f" ({', '.join(unclear)})" if unclear else ""))
    damages = [d for d in (claim.building_damage, claim.contents_damage) if d is not None]
    if not damages:
        judgment.append("no_damage_documented")
    for p in (b, c):
        if p.damage is not None and 0 < p.damage < PLACEHOLDER_DAMAGE and p.covered:
            judgment.append(f"{p.part}_damage_placeholder")
        if p.covered and p.damage is not None and p.deductible is None:
            judgment.append(f"{p.part}_deductible_unknown")
    if not b.covered and not c.covered:
        judgment.append("no_coverage_on_record")
    clauses = ["FLD-DEF-FLOOD", "FLD-DEDUCTIBLE", "FLD-LIMITS"]
    if b.covered:
        clauses.insert(0, "FLD-BUILDING")
    if c.covered:
        clauses.insert(1 if b.covered else 0, "FLD-CONTENTS")
    if before:
        clauses.append("FLD-EXC-BEFORE-INCEPTION")
    total = round(b.payout + c.payout, 2)
    if not before and total == 0 and not judgment:
        clauses.append("FLD-BELOW-DEDUCTIBLE")
    if any(j.endswith(("not_documented", "documented", "placeholder")) for j in judgment):
        clauses.append("FLD-COND-PROOF")
    if "cause_not_clearly_flood" in " ".join(judgment):
        clauses.append("FLD-EXC-NOT-FLOOD")
    return FloodCoverage(
        building=b, contents=c, total=0.0 if before else total, before_inception=before,
        cause=", ".join(flood) or "unclear", judgment_needed=judgment, clauses=clauses,
    )  # fmt: skip


def decide(cov: FloodCoverage) -> tuple[str, list[str], float | None]:
    """(outcome, reason codes, payout). Code decides; judgment cases escalate to an adjuster."""
    if cov.before_inception:
        return "escalate", ["check_policy_history"], None
    if cov.judgment_needed:
        return "escalate", ["adjuster_judgment_needed"], None
    if cov.total > 0:
        return "approve", ["covered_flood_loss"], cov.total
    return "deny", ["below_deductible"], None
