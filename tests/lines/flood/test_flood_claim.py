import pandas as pd
import pytest

from autoclaim.lines.flood import claim as fc

ROW = {
    "id": "c1", "dateOfLoss": "2023-09-23", "originalNBDate": "2022-09-26",
    "totalBuildingInsuranceCoverage": 250000, "totalContentsInsuranceCoverage": 100000,
    "buildingDamageAmount": 6948.0, "contentsDamageAmount": float("nan"),
    "buildingDeductibleCode": "2", "contentsDeductibleCode": "2", "causeOfDamage": "1",
    "floodEvent": "Tropical Storm Ophelia", "ratedFloodZone": "AE",
    "primaryResidenceIndicator": True, "waterDepth": 99, "replacementCostBasis": "A",
    "amountPaidOnBuildingClaim": 4947.64, "amountPaidOnContentsClaim": 0.0,
}  # fmt: skip


def claim(**over: object) -> fc.FloodClaim:
    return fc.from_nfip(pd.Series({**ROW, **over}))


def test_real_record_reproduces_the_fema_payment() -> None:
    cov = fc.assess(claim())  # FEMA paid $4,947.64 on this record
    assert cov.building.deductible == 2000.0 and cov.building.payout == 4948.0
    assert cov.contents.payout == 0.0  # blank contents damage = no contents claim
    assert fc.decide(cov) == ("approve", ["covered_flood_loss"], 4948.0)
    assert cov.cause == "tidal water overflow" and "FLD-BUILDING" in cov.clauses


def test_from_nfip_parsing() -> None:
    c = claim(buildingDeductibleCode=5.0, causeOfDamage="2D", replacementCostBasis="R")
    assert c.building_deductible_code == "5" and c.cause_codes == ["2", "D"]
    assert c.replacement_cost_basis is True and c.contents_damage is None


def test_limit_caps_the_payout() -> None:
    cov = fc.assess(claim(buildingDamageAmount=400000.0, totalBuildingInsuranceCoverage=250000))
    assert cov.building.payout == 250000.0


def test_both_parts_pay_separately_after_their_own_deductibles() -> None:
    cov = fc.assess(claim(contentsDamageAmount=3000.0, contentsDeductibleCode="1"))
    assert cov.contents.payout == 2000.0 and cov.total == 6948.0


def test_below_deductible_is_denied() -> None:
    cov = fc.assess(claim(buildingDamageAmount=1500.0))
    assert fc.decide(cov) == ("deny", ["below_deductible"], None)
    assert "FLD-BELOW-DEDUCTIBLE" in cov.clauses


def test_no_contents_coverage_pays_nothing_for_contents() -> None:
    cov = fc.assess(claim(totalContentsInsuranceCoverage=0, contentsDamageAmount=5000.0))
    assert not cov.contents.covered and cov.contents.payout == 0.0


@pytest.mark.parametrize(
    ("over", "why"),
    [({"causeOfDamage": "0"}, "cause_not_clearly_flood (other cause)"),
     ({"causeOfDamage": "9"}, "cause_not_clearly_flood (earth movement)"),
     ({"buildingDamageAmount": float("nan")}, "no_damage_documented"),
     ({"buildingDamageAmount": 1.0}, "building_damage_placeholder"),
     ({"buildingDeductibleCode": "Z"}, "building_deductible_unknown"),
     ({"totalBuildingInsuranceCoverage": 0, "totalContentsInsuranceCoverage": 0},
      "no_coverage_on_record")],
)  # fmt: skip
def test_judgment_cases_escalate(over: dict, why: str) -> None:
    cov = fc.assess(claim(**over))
    assert why in cov.judgment_needed
    assert fc.decide(cov)[0] == "escalate"


def test_zero_damage_is_not_a_placeholder() -> None:
    cov = fc.assess(claim(buildingDamageAmount=0.0, contentsDamageAmount=0.0))
    assert not cov.judgment_needed and fc.decide(cov)[0] == "deny"


def test_handling_codes_alone_still_mean_a_flood() -> None:
    cov = fc.assess(claim(causeOfDamage="D"))
    assert not cov.judgment_needed and "expedited" in cov.cause


def test_loss_before_the_record_start_escalates_not_denies() -> None:
    cov = fc.assess(claim(originalNBDate="2024-01-01"))
    assert cov.before_inception and cov.total == 0.0
    assert fc.decide(cov) == ("escalate", ["check_policy_history"], None)


def test_deductible_codes_from_the_fema_dictionary() -> None:
    assert fc.DEDUCTIBLE_CODES["9"] == 750.0 and fc.DEDUCTIBLE_CODES["A"] == 10_000.0
    assert fc.DEDUCTIBLE_CODES["H"] == 200.0
