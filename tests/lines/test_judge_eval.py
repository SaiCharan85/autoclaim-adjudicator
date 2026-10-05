import json

import judgekit
import pandas as pd
import pytest

from autoclaim.config import load_carrier_config
from autoclaim.core.judge import load_rubric
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH
from autoclaim.lines.auto.policy import POLICY_PATH
from autoclaim.retrieval.corpus import load_policy

JUR = load_carrier_config().active_jurisdiction
POLICY = load_policy(POLICY_PATH)
RUBRIC = load_rubric(RUBRIC_PATH)

BASE = {
    "claim_id": "CLM-000001", "policy_id": "POL-000001", "loss_date": "2023-05-15",
    "report_date": "2023-05-16", "cause": "glass", "use_at_loss": "personal",
    "vehicle_role": "in_transport", "n_vehicles": 1, "injury_count": 0.0,
    "damage_extent": "minor", "towed": 0.0, "at_fault": 0.0, "police_report": False,
    "police_report_hours": float("nan"), "witness_count": 0, "attorney_involved": False,
    "claimed_amount": 1161.91, "policy_state": "OH", "policy_start_date": "2023-01-01",
    "coverage": "collision_comprehensive", "collision_deductible": 500.0,
    "comprehensive_deductible": 250.0, "rideshare_endorsement": False,
    "driver_role": "named_insured", "vehicle_model_year": 2018, "vehicle_make": "Honda",
    "vehicle_model": "Civic", "body_class": "car", "vehicle_acv": 15000.0, "adas": False,
    "financed": False, "prior_claims_3y": 0, "address_change_days": float("nan"),
    "gt_decision": "approve", "gt_reasons": float("nan"), "gt_payout": 911.91,
}  # fmt: skip


def row(**over: object) -> pd.Series:
    return pd.Series({**BASE, **over})


def sample(**over: object) -> judgekit.Sample:
    s = je.reference_sample(row(**over), JUR, POLICY)
    assert s is not None
    return s


def decision(s: judgekit.Sample) -> dict:
    return json.loads(s.fields["decision"])


# ---------------------------------------------------------------- reference cases


def test_approval_reference_is_correct_by_construction() -> None:
    s = sample()
    d = decision(s)
    assert (d["outcome"], d["reasons"], d["payout"]) == ("approve", ["covered_loss"], 911.91)
    text = s.fields["explanation"]
    assert "comprehensive coverage [INS-COMPREHENSIVE]" in text
    assert "$250.00 comprehensive deductible [LIM-DEDUCTIBLE]" in text
    assert "we approve a payment of $911.91" in text
    assert d["cited_clauses"] == ["INS-COMPREHENSIVE", "LIM-DEDUCTIBLE"]
    assert "[INS-COMPREHENSIVE]" in s.fields["clauses"]  # real clause text is included


def test_render_matches_the_live_judge_format() -> None:
    case = je.render(sample())
    lines = case.split("\n")
    assert lines[0].startswith("facts: {") and lines[1].startswith("deterministic: {")
    assert lines[2] == "cited clauses:"
    assert any(x.startswith('decision: {"outcome":"approve"') for x in lines)
    assert lines[-1].startswith("explanation: The loss on 2023-05-15")


@pytest.mark.parametrize(
    ("over", "reason", "clause", "phrase"),
    [
        ({"coverage": "liability_only", "collision_deductible": float("nan"),
          "comprehensive_deductible": float("nan")},
         "no_physical_damage_coverage", "DEC-LIABILITY-ONLY", "liability-only"),
        ({"coverage": "collision", "comprehensive_deductible": float("nan")},
         "no_comprehensive_coverage", "INS-COMPREHENSIVE", "does not carry"),
        ({"driver_role": "excluded_driver", "cause": "collision_vehicle"},
         "excluded_driver", "EXC-EXCLUDED-DRIVER", "excluded by name"),
        ({"use_at_loss": "rideshare_active", "cause": "collision_vehicle"},
         "business_use_exclusion", "EXC-COMMERCIAL-USE", "no rideshare endorsement"),
        ({"cause": "hit_and_run", "police_report": True, "police_report_hours": 48.0},
         "hit_and_run_report_condition", "COND-HIT-AND-RUN-POLICE-REPORT", "after 48 hours"),
        ({"claimed_amount": 200.0, "gt_payout": 0.0},
         "below_deductible", "LIM-LOSS-BELOW-DEDUCTIBLE", "nothing is payable"),
    ],
)  # fmt: skip
def test_denial_references(over: dict, reason: str, clause: str, phrase: str) -> None:
    s = sample(gt_decision="deny", gt_reasons=reason, **over)
    d = decision(s)
    assert d["outcome"] == "deny" and d["payout"] is None and d["reasons"] == [reason]
    assert clause in d["cited_clauses"]
    assert phrase in s.fields["explanation"]
    assert s.fields["explanation"].endswith("We deny the claim.")


def test_late_notice_referral() -> None:
    s = sample(report_date="2023-07-20", gt_decision="escalate",
               gt_reasons="late_notice_prejudice_review")  # fmt: skip
    text = s.fields["explanation"]
    assert "reported 66 days after" in text and f"the {JUR.late_notice_days} days" in text
    assert text.endswith("We refer the claim to an adjuster.")


def test_multiple_reasons_each_get_a_sentence() -> None:
    s = sample(coverage="liability_only", collision_deductible=float("nan"),
               comprehensive_deductible=float("nan"), gt_decision="deny",
               gt_reasons="no_physical_damage_coverage;no_comprehensive_coverage")  # fmt: skip
    assert decision(s)["cited_clauses"] == ["DEC-LIABILITY-ONLY", "INS-COMPREHENSIVE"]


def test_hit_and_run_approval_states_the_timely_report() -> None:
    s = sample(cause="hit_and_run", police_report=True, police_report_hours=3.0,
               claimed_amount=2000.0, gt_payout=1500.0)  # fmt: skip
    assert "within 3 hours" in s.fields["explanation"]
    assert "COND-HIT-AND-RUN-POLICE-REPORT" in decision(s)["cited_clauses"]


def test_total_loss_uses_actual_cash_value() -> None:
    s = sample(claimed_amount=14000.0, gt_payout=14750.0)  # >= 75% of $15,000 ACV
    assert "total loss" in s.fields["explanation"]
    assert "LIM-TOTAL-LOSS" in decision(s)["cited_clauses"]
    assert decision(s)["payout"] == 14750.0


# ---------------------------------------------------------------- rows without a faithful reference


def test_fraud_referrals_are_skipped() -> None:
    assert je.reference_sample(row(gt_decision="escalate", gt_reasons="suspected_fraud_siu"),
                               JUR, POLICY) is None  # fmt: skip


def test_inconsistent_payout_is_skipped() -> None:
    assert je.reference_sample(row(gt_payout=500.0), JUR, POLICY) is None  # inflated estimate


def test_reason_the_facts_do_not_support_is_skipped() -> None:
    assert je.reference_sample(row(gt_decision="deny", gt_reasons="excluded_driver"),
                               JUR, POLICY) is None  # fmt: skip


def test_true_reasons_treat_blank_approval_as_covered_loss() -> None:
    assert je.true_reasons(row()) == ["covered_loss"]
    assert je.true_reasons(row(gt_decision="deny", gt_reasons=float("nan"))) == []


# ---------------------------------------------------------------- selection and error types


def test_select_samples_balances_reasons_and_period() -> None:
    rows = [row(claim_id=f"CLM-{i:06d}", loss_date="2023-05-15") for i in range(6)]
    rows += [row(claim_id=f"CLM-{i:06d}", gt_decision="deny", gt_reasons="excluded_driver",
                 driver_role="excluded_driver", cause="collision_vehicle")
             for i in range(6, 9)]  # fmt: skip
    rows += [row(claim_id="CLM-000099", loss_date="2025-01-01")]  # outside the period
    out = je.select_samples(pd.DataFrame(rows), "2000-01-01", "2024-07-01", 4, JUR, POLICY, 0)
    reasons = [decision(s)["reasons"][0] for s in out]
    assert len(out) == 4 and reasons.count("excluded_driver") == 2  # round robin, not 4 approvals
    assert "CLM-000099" not in {s.id for s in out}


def test_error_types_name_real_rubric_items() -> None:
    for e in je.ERROR_TYPES:
        assert e.expected_items <= set(RUBRIC.ids), e.name


def test_every_error_type_plants_into_some_reference() -> None:
    denial = sample(claim_id="CLM-000002", gt_decision="deny", gt_reasons="excluded_driver",
                    driver_role="excluded_driver", cause="collision_vehicle")  # fmt: skip
    refs = [sample(), denial]
    suite = judgekit.plant(refs, je.ERROR_TYPES, seed=0)
    assert set(suite.counts()) == {e.name for e in je.ERROR_TYPES}
    for p in suite.planted:
        original = next(r for r in refs if r.id == p.sample_id)
        assert p.sample.fields["explanation"] != original.fields["explanation"]
        assert {k: v for k, v in p.sample.fields.items() if k != "explanation"} == {
            k: v for k, v in original.fields.items() if k != "explanation"
        }  # only the explanation is edited: the evidence stays true


def test_contradicted_conclusion_flips_the_stated_outcome() -> None:
    suite = judgekit.plant([sample()], [je.ERROR_TYPES[-1]], seed=0)
    assert "we deny a payment" in suite.planted[0].sample.fields["explanation"]
