"""Ground-truth coverage oracle for simulated claims. EVALUATION ONLY.

The harness must never import this module (a test enforces it): it knows the latent truth
(true damage, real fraud) that a real insurer never sees at first notice.

Precedence (plain English):
1. Deny if the policy can't pay: liability-only (no own-damage cover), excluded driver,
   mechanical breakdown / wear and tear, business use without endorsement, a comprehensive
   cause on a policy without comprehensive, or a hit-and-run without a timely police report.
2. Otherwise escalate if notice was late (most states require the insurer to show prejudice,
   which is a human judgment) or the claim is actually fraudulent (refer to investigators).
3. Otherwise approve and pay (repair cost, or vehicle value for a total loss) minus deductible.
   A covered loss below the deductible is denied as "below_deductible".
"""

import numpy as np
import pandas as pd

from autoclaim.config import JurisdictionProfile

BUSINESS_USES = ("rideshare_active", "delivery_active")
DENIAL_REASONS = (
    "no_physical_damage_coverage",
    "excluded_driver",
    "wear_and_tear_mechanical",
    "business_use_exclusion",
    "no_comprehensive_coverage",
    "hit_and_run_report_condition",
)
ESCALATION_REASONS = ("late_notice_prejudice_review", "suspected_fraud_siu")


def _join(flags: dict[str, np.ndarray]) -> np.ndarray:
    names = np.array(list(flags), dtype=object)
    matrix = np.column_stack(list(flags.values()))
    return np.array([";".join(names[row]) for row in matrix], dtype=object)


def trap_families(claims: pd.DataFrame, jur: JurisdictionProfile) -> np.ndarray:
    """Which of the six coverage traps each claim exercises (covered or not)."""
    return _join(
        {
            "animal_strike": (claims["cause"] == "animal").to_numpy(),
            "business_use": claims["use_at_loss"].isin(BUSINESS_USES).to_numpy(),
            "driver_status": claims["driver_role"]
            .isin(["permissive_unlisted", "excluded_driver"])
            .to_numpy(),
            "wear_and_tear": (claims["cause"] == "mechanical_breakdown").to_numpy(),
            "hit_and_run": (claims["cause"] == "hit_and_run").to_numpy(),
            "late_notice": (claims["notice_days"] > jur.late_notice_days).to_numpy(),
        }
    )


def adjudicate(claims: pd.DataFrame, jur: JurisdictionProfile) -> pd.DataFrame:
    cause = claims["cause"].to_numpy()
    coverage = claims["coverage"].to_numpy()
    comp_cause = np.isin(cause, jur.comprehensive_causes)
    hours = claims["police_report_hours"].to_numpy(dtype=float)
    late_report = ~claims["police_report"].to_numpy(dtype=bool) | ~(
        hours <= jur.hit_and_run_police_report_hours
    )

    deny = {
        "no_physical_damage_coverage": coverage == "liability_only",
        "excluded_driver": (claims["driver_role"] == "excluded_driver").to_numpy(),
        "wear_and_tear_mechanical": cause == "mechanical_breakdown",
        "business_use_exclusion": claims["use_at_loss"].isin(BUSINESS_USES).to_numpy()
        & ~claims["rideshare_endorsement"].to_numpy(dtype=bool),
        "no_comprehensive_coverage": comp_cause & (coverage != "collision_comprehensive"),
        "hit_and_run_report_condition": (cause == "hit_and_run") & late_report,
    }
    escalate = {
        "late_notice_prejudice_review": (claims["notice_days"] > jur.late_notice_days).to_numpy(),
        "suspected_fraud_siu": claims["gt_is_fraud"].to_numpy(dtype=bool),
    }
    denied = np.logical_or.reduce(list(deny.values()))
    escalated = ~denied & np.logical_or.reduce(list(escalate.values()))

    part = np.where(
        cause == "mechanical_breakdown", "none", np.where(comp_cause, "comprehensive", "collision")
    )
    deductible = np.where(
        part == "comprehensive",
        claims["comprehensive_deductible"].to_numpy(dtype=float),
        claims["collision_deductible"].to_numpy(dtype=float),
    )
    acv = claims["vehicle_acv"].to_numpy(dtype=float)
    true_damage = claims["gt_true_damage"].to_numpy(dtype=float)
    total_loss = true_damage >= jur.total_loss_threshold * acv
    gross = np.where(total_loss, acv, true_damage)
    payout = np.round(np.maximum(gross - np.nan_to_num(deductible), 0.0), 2)

    approve = ~denied & ~escalated
    below = approve & (payout <= 0)
    decision = np.select([denied | below, escalated], ["deny", "escalate"], default="approve")
    reasons = np.where(
        denied,
        _join(deny),
        np.where(escalated, _join({k: v & escalated for k, v in escalate.items()}), ""),
    )
    reasons = np.where(below, "below_deductible", reasons)

    return pd.DataFrame(
        {
            "gt_decision": decision,
            "gt_reasons": reasons,
            "gt_coverage_part": part,
            "gt_deductible": np.where(part == "none", np.nan, deductible),
            "gt_total_loss": total_loss,
            "gt_payout": np.where(decision == "approve", payout, np.nan),
            "gt_traps": trap_families(claims, jur),
        },
        index=claims.index,
    )
