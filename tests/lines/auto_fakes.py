"""A hand-built claim package and fact sets for auto-line tests (no data, no network)."""

from datetime import date

from autoclaim.lines.auto.claim import ClaimPackage, PolicyRecord, Vehicle
from autoclaim.lines.auto.facts import ClaimFacts


def package(**policy_changes) -> ClaimPackage:
    policy = PolicyRecord(
        policy_id="POL-1",
        policy_state="OH",
        policy_start_date=date(2023, 1, 1),
        coverage="collision_comprehensive",
        collision_deductible=500.0,
        comprehensive_deductible=250.0,
        rideshare_endorsement=False,
        named_insured="Maria Garcia",
        listed_drivers=["Daniel Garcia"],
        excluded_drivers=["Kevin Garcia"],
        vehicle=Vehicle(
            year=2019,
            make="Honda",
            model="Civic",
            body_class="car",
            actual_cash_value=15_000.0,
            adas=True,
            lienholder=None,
        ),
        prior_claims_3y=0,
        address_change_days=None,
    ).model_copy(update=policy_changes)
    return ClaimPackage(
        claim_id="CLM-T1",
        channel="web",
        report_date=date(2024, 8, 3),
        estimate_amount=1_900.0,
        narrative="A deer jumped out on route 9 last night and smashed my headlight.",
        policy=policy,
    )


def facts(**changes) -> ClaimFacts:
    base = ClaimFacts(
        loss_date=date(2024, 8, 2),
        cause="animal",
        summary="Deer strike on a rural road at night.",
        driver_name="Maria Garcia",
        driver_relationship="self",
        use_at_loss="personal",
        vehicle_parked=False,
        n_vehicles=1,
        injuries=0,
        damage_severity="minor",
        towed=False,
        other_driver_fled=None,
        insured_at_fault=False,
        police_report=False,
        police_report_hours=None,
        witnesses=0,
        attorney_involved=False,
        missing_info=[],
    )
    return base.model_copy(update=changes)
