"""When each claim column becomes known. Only `FNOL` columns may ever be model features.

FNOL = first notice of loss: what the insurer knows when the claim is opened (claimant's
statement, policy records, vehicle lookup). The fraud model scores claims at that moment, so
anything learned later (investigation outcomes) or never knowable (simulation truth) is excluded.
"""

from collections.abc import Iterable
from enum import StrEnum


class Availability(StrEnum):
    ID = "id"  # identifiers: never features
    PROVENANCE = "provenance"  # where the row came from: never features (would leak real-vs-sim)
    FNOL = "fnol"
    POST_FNOL = "post_fnol"  # known only after investigation (e.g. confirmed fraud = the label)
    GROUND_TRUTH = "ground_truth"  # simulation truth, never knowable by an insurer


_FNOL = (
    "loss_date", "loss_hour", "region", "area", "weather", "light", "vehicle_make",
    "vehicle_model", "vehicle_model_year", "body_class", "cause", "n_vehicles", "injury_count",
    "damage_extent", "towed", "at_fault", "vehicle_role", "loss_state", "policy_state",
    "coverage", "collision_deductible", "comprehensive_deductible", "rideshare_endorsement",
    "driver_role", "driver_age", "use_at_loss", "address_change_days", "prior_claims_3y",
    "vehicle_age", "vehicle_acv", "financed", "adas", "police_report", "police_report_hours",
    "witness_count", "notice_days", "attorney_involved", "claimed_amount", "policy_start_date",
    "report_date",
    # derived at cleaning time from FNOL columns only
    "days_policy_to_loss", "days_policy_to_claim", "claim_to_acv", "loss_month", "loss_dow",
    "out_of_state",
    # fitted feature stages (ml/stages.py), computed from FNOL columns only
    "expected_log_amount", "amount_residual",
)  # fmt: skip

AVAILABILITY: dict[str, Availability] = {
    "claim_id": Availability.ID,
    "policy_id": Availability.ID,
    "record_origin": Availability.PROVENANCE,
    "source_record": Availability.PROVENANCE,
    "fraud_confirmed": Availability.POST_FNOL,
    **dict.fromkeys(_FNOL, Availability.FNOL),
}


class LeakageError(ValueError):
    pass


def availability(column: str) -> Availability:
    if column.startswith("gt_"):
        return Availability.GROUND_TRUTH
    if column not in AVAILABILITY:
        raise LeakageError(f"column {column!r} has no availability tag; add it to columns.py")
    return AVAILABILITY[column]


def assert_fnol_only(features: Iterable[str]) -> None:
    """Raise if any model feature is not known at first notice of loss (or is untagged)."""
    bad = {f: availability(f).value for f in features if availability(f) is not Availability.FNOL}
    if bad:
        raise LeakageError(f"non-FNOL columns used as features: {bad}")
