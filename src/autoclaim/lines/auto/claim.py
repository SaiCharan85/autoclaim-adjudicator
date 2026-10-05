"""The auto claim package the harness receives, and how one is built from a simulated row.

A real first notice of loss has three parts: the policy record (from the policy system), the
claimant's story (free text), and attachments (here: the repair estimate). Only the story is
written by an LLM (narratives.py); everything else is deterministic, seeded by the claim id.
Simulation truth (gt_* columns) and the fraud label never enter a package.
"""

import hashlib
from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

Coverage = Literal["collision", "collision_comprehensive", "liability_only"]

FIRST = ("James", "Maria", "Robert", "Linda", "Michael", "Aisha", "David", "Sofia", "Daniel",
         "Priya", "Kevin", "Grace", "Carlos", "Emily", "Andre", "Hannah", "Luis", "Mei", "Tyler",
         "Fatima", "Brandon", "Olivia", "Jamal", "Chloe", "Victor", "Nadia")  # fmt: skip
LAST = ("Garcia", "Smith", "Nguyen", "Johnson", "Patel", "Brown", "Lopez", "Williams", "Kim",
        "Davis", "Martinez", "Wilson", "Okafor", "Anderson", "Thomas", "Rivera", "Moore",
        "Jackson", "Lee", "Walker")  # fmt: skip
UNLISTED_RELATION = ("friend", "neighbor", "coworker", "nephew", "niece", "roommate", "cousin")
LISTED_RELATION = ("spouse", "son", "daughter", "partner")
LENDERS = ("Summit Auto Finance", "Lakeside Credit Union", "Pioneer Bank Auto Loans")


class Vehicle(BaseModel):
    year: int | None
    make: str
    model: str
    body_class: str
    actual_cash_value: float = Field(gt=0)
    adas: bool | None
    lienholder: str | None


class PolicyRecord(BaseModel):
    policy_id: str
    policy_state: str
    policy_start_date: date
    coverage: Coverage
    collision_deductible: float | None
    comprehensive_deductible: float | None
    rideshare_endorsement: bool
    named_insured: str
    listed_drivers: list[str]
    excluded_drivers: list[str]
    vehicle: Vehicle
    prior_claims_3y: int
    address_change_days: int | None  # days since the insured's last move; None = none in 2 years


class ClaimPackage(BaseModel):
    claim_id: str
    channel: Literal["phone", "web", "email", "app"]
    report_date: date
    estimate_amount: float = Field(gt=0)
    narrative: str
    policy: PolicyRecord


class DriverAtLoss(BaseModel):
    name: str
    relationship: str  # how the claimant describes them


def _rng_int(claim_id: str, salt: str, n: int) -> int:
    digest = hashlib.sha256(f"{claim_id}|{salt}".encode()).hexdigest()
    return int(digest, 16) % n


def _name(claim_id: str, salt: str) -> str:
    return (
        f"{FIRST[_rng_int(claim_id, salt + 'f', len(FIRST))]} "
        + (LAST[_rng_int(claim_id, salt + "l", len(LAST))])
    )


def people(row: pd.Series) -> tuple[str, list[str], list[str], DriverAtLoss]:
    """Seeded names: (named insured, listed drivers, excluded drivers, who drove at the loss)."""
    cid = str(row["claim_id"])
    insured = _name(cid, "insured")
    surname = insured.split()[1]
    listed = [f"{FIRST[_rng_int(cid, 'listed', len(FIRST))]} {surname}"]
    excluded = [f"{FIRST[_rng_int(cid, 'excluded', len(FIRST))]} {surname}"]
    if excluded[0] in (insured, *listed):
        excluded = [f"{FIRST[(_rng_int(cid, 'excluded', len(FIRST)) + 1) % len(FIRST)]} {surname}"]
    role = row["driver_role"]
    if role == "named_insured":
        driver = DriverAtLoss(name=insured, relationship="self (named insured)")
    elif role == "listed_driver":
        rel = LISTED_RELATION[_rng_int(cid, "rel", len(LISTED_RELATION))]
        driver = DriverAtLoss(name=listed[0], relationship=rel)
    elif role == "excluded_driver":
        rel = LISTED_RELATION[_rng_int(cid, "rel", len(LISTED_RELATION))]
        driver = DriverAtLoss(name=excluded[0], relationship=rel)
    else:  # permissive_unlisted: someone not on the policy, driving with permission
        rel = UNLISTED_RELATION[_rng_int(cid, "rel", len(UNLISTED_RELATION))]
        driver = DriverAtLoss(name=_name(cid, "friend"), relationship=rel)
    return insured, listed, excluded, driver


def _opt_float(v: Any) -> float | None:
    return None if pd.isna(v) else float(v)


def policy_record(row: pd.Series) -> PolicyRecord:
    insured, listed, excluded, _ = people(row)
    year = row.get("vehicle_model_year")
    return PolicyRecord(
        policy_id=str(row["policy_id"]),
        policy_state=str(row["policy_state"]),
        policy_start_date=pd.Timestamp(row["policy_start_date"]).date(),
        coverage=row["coverage"],
        collision_deductible=_opt_float(row["collision_deductible"]),
        comprehensive_deductible=_opt_float(row["comprehensive_deductible"]) or None,
        rideshare_endorsement=bool(row["rideshare_endorsement"]),
        named_insured=insured,
        listed_drivers=listed,
        excluded_drivers=excluded,
        vehicle=Vehicle(
            year=None if pd.isna(year) else int(year),
            make=str(row["vehicle_make"]),
            model=str(row["vehicle_model"]),
            body_class=str(row["body_class"]),
            actual_cash_value=float(row["vehicle_acv"]),
            adas=None if pd.isna(row["adas"]) else bool(row["adas"]),
            lienholder=LENDERS[_rng_int(str(row["claim_id"]), "lender", len(LENDERS))]
            if bool(row["financed"])
            else None,
        ),
        prior_claims_3y=int(row["prior_claims_3y"]),
        address_change_days=None
        if pd.isna(row["address_change_days"])
        else int(row["address_change_days"]),
    )


# What the claimant knows and may say. Policy facts are not in the story (they come from the
# policy system); the fraud label and simulation truth are never shown to the generator.
STORY_FIELDS = (
    "loss_date", "loss_hour", "area", "weather", "light", "cause", "n_vehicles", "injury_count",
    "damage_extent", "towed", "at_fault", "vehicle_role", "loss_state", "use_at_loss",
    "police_report", "police_report_hours", "witness_count", "attorney_involved",
)  # fmt: skip


def story_facts(row: pd.Series) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for col in STORY_FIELDS:
        v = row.get(col)
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            continue
        facts[col] = v.item() if hasattr(v, "item") else v
    facts["loss_date"] = str(pd.Timestamp(row["loss_date"]).date())
    _, _, _, driver = people(row)
    facts["driver"] = driver.model_dump()
    v = policy_record(row).vehicle
    facts["vehicle"] = f"{v.year or ''} {v.make} {v.model}".strip()
    return facts
