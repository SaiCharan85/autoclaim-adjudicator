"""Planted-error evaluation of the claim judge. EVALUATION ONLY (reads simulation truth; the
harness never imports this module, a test enforces it).

JudgeKit does the generic work (planting, running, rates, CIs). This module adds what is
insurance-specific:

1. Reference cases built from simulated claims WITHOUT any LLM: the true facts, the code-derived
   numbers, the real clause text, the ground-truth decision, and a templated explanation that is
   correct by construction. Rows where the derived numbers disagree with the ground truth are
   skipped, so every clean sample really is clean. Fraud referrals are skipped too: their reason
   (latent fraud) is not visible to the judge, so no faithful explanation of it exists.
2. Insurance error types: a wrong amount, an invented fact, a misstated coverage, an omitted
   material fact, and a conclusion that contradicts the decision; each names the rubric item
   that should catch it.

Plain-English trace of one reference: deer strike on 2024-03-05 -> comprehensive -> policy has
comprehensive [INS-COMPREHENSIVE] -> estimate $1,900 -> $250 deductible [LIM-DEDUCTIBLE] ->
"we approve a payment of $1,650.00".
"""

import json
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
from judgekit import ErrorType, Sample
from judgekit.mutators import drop_sentence, insert_sentence, number_swap, replace_phrase

from autoclaim.config import JurisdictionProfile
from autoclaim.lines.auto.claim import ClaimPackage, people, policy_record
from autoclaim.lines.auto.facts import ClaimFacts, DerivedFacts, derive
from autoclaim.lines.auto.line import clauses_text, render_judge_case
from autoclaim.retrieval.corpus import PolicyDoc

CAUSE_TEXT = {
    "animal": "an animal strike",
    "collision_object": "a collision with an object",
    "collision_vehicle": "a collision with another vehicle",
    "fire": "a fire",
    "glass": "glass breakage",
    "hail_weather": "hail or weather damage",
    "hit_and_run": "a hit-and-run",
    "mechanical_breakdown": "a mechanical breakdown",
    "parked_hit": "damage while parked",
    "theft": "a theft",
    "vandalism": "vandalism",
}
ROLE_TEXT = {
    "named_insured": "the named insured",
    "listed_driver": "a driver listed on the policy",
    "excluded_driver": "a driver excluded by name on the policy",
    "permissive_unlisted": "a permissive driver who is not listed on the policy",
}
USE_TEXT = {
    "personal": "personal use",
    "commute": "commuting",
    "rideshare_active": "an active rideshare trip",
    "delivery_active": "an active delivery",
}
PART_CLAUSE = {"collision": "INS-COLLISION", "comprehensive": "INS-COMPREHENSIVE"}
SUPPORTED_REASONS = frozenset({
    "covered_loss", "no_physical_damage_coverage", "no_comprehensive_coverage", "excluded_driver",
    "wear_and_tear_mechanical", "business_use_exclusion", "hit_and_run_report_condition",
    "below_deductible", "late_notice_prejudice_review",
})  # fmt: skip
CASE_FIELDS = ("facts", "deterministic", "clauses", "decision", "explanation")


def money(x: float) -> str:
    return f"${x:,.2f}"


def _opt(v: Any, cast: type) -> Any:
    return None if pd.isna(v) else cast(v)


# ---------------------------------------------------------------- reference cases


def true_facts(row: pd.Series) -> ClaimFacts:
    """The facts as they really were (what a perfect intake would extract)."""
    _, _, _, driver = people(row)
    loss = pd.Timestamp(row["loss_date"]).date()
    cause = str(row["cause"])
    return ClaimFacts(
        loss_date=loss,
        cause=cause,  # type: ignore[arg-type]
        summary=f"{CAUSE_TEXT.get(cause, cause)} on {loss}".capitalize(),
        driver_name=driver.name,
        driver_relationship=driver.relationship,
        use_at_loss=row["use_at_loss"],
        vehicle_parked=row["vehicle_role"] == "parked",
        n_vehicles=_opt(row["n_vehicles"], int),
        injuries=_opt(row["injury_count"], int),
        damage_severity=row["damage_extent"],
        towed=_opt(row["towed"], bool),
        other_driver_fled=cause == "hit_and_run",
        insured_at_fault=_opt(row["at_fault"], bool),
        police_report=bool(row["police_report"]),
        police_report_hours=_opt(row["police_report_hours"], float),
        witnesses=_opt(row["witness_count"], int),
        attorney_involved=bool(row["attorney_involved"]),
        missing_info=[],
    )


def package(row: pd.Series) -> ClaimPackage:
    return ClaimPackage(
        claim_id=str(row["claim_id"]),
        channel="web",
        report_date=pd.Timestamp(row["report_date"]).date(),
        estimate_amount=float(row["claimed_amount"]),
        narrative="(not used by the judge)",
        policy=policy_record(row),
    )


def _truth(pkg: ClaimPackage, x: DerivedFacts) -> dict[str, bool]:
    """Which reason codes the deterministic facts support (mirrors the critic)."""
    return {
        "no_physical_damage_coverage": pkg.policy.coverage == "liability_only",
        "no_comprehensive_coverage": x.coverage_part == "comprehensive" and not x.part_on_policy,
        "excluded_driver": x.driver_role == "excluded_driver",
        "wear_and_tear_mechanical": x.coverage_part == "none",
        "business_use_exclusion": x.business_use_without_endorsement,
        "hit_and_run_report_condition": x.hit_and_run_report_ok is False,
        "below_deductible": x.payout <= 0,
        "late_notice_prejudice_review": x.late_notice,
        "covered_loss": x.part_on_policy
        and x.payout > 0
        and x.driver_role != "excluded_driver"
        and not x.business_use_without_endorsement
        and x.hit_and_run_report_ok is not False,
    }


def explain(
    facts: ClaimFacts,
    x: DerivedFacts,
    pkg: ClaimPackage,
    outcome: str,
    reasons: Sequence[str],
    jur: JurisdictionProfile,
) -> tuple[str, list[str]]:
    """A correct templated explanation and the clauses it cites, for the true decision."""
    cited: list[str] = []
    s: list[str] = []
    cause = CAUSE_TEXT.get(facts.cause, facts.cause)
    if x.coverage_part in PART_CLAUSE:
        s.append(f"The loss on {facts.loss_date} was {cause}, which falls under "
                 f"{x.coverage_part} coverage.")  # fmt: skip
    else:
        s.append(f"The loss on {facts.loss_date} was {cause}.")
    s.append(f"The car was driven by {facts.driver_name}, {ROLE_TEXT[x.driver_role]}.")
    if not facts.vehicle_parked:
        s.append(f"It was being used for {USE_TEXT[facts.use_at_loss]} at the time.")
    hours = jur.hit_and_run_police_report_hours
    for r in reasons:
        if r == "no_physical_damage_coverage":
            cited.append("DEC-LIABILITY-ONLY")
            s.append("The policy is liability-only and carries no collision or comprehensive "
                     "coverage for the insured's own car [DEC-LIABILITY-ONLY].")  # fmt: skip
        elif r == "no_comprehensive_coverage":
            cited.append("INS-COMPREHENSIVE")
            s.append("This loss needs comprehensive coverage, which the policy does not carry "
                     "[INS-COMPREHENSIVE].")  # fmt: skip
        elif r == "excluded_driver":
            cited.append("EXC-EXCLUDED-DRIVER")
            s.append(f"{facts.driver_name} is excluded by name, and the policy excludes any loss "
                     "while an excluded driver is driving [EXC-EXCLUDED-DRIVER].")  # fmt: skip
        elif r == "wear_and_tear_mechanical":
            cited.append("EXC-MECHANICAL-BREAKDOWN")
            s.append("The policy excludes mechanical breakdown that is not caused by a covered "
                     "loss [EXC-MECHANICAL-BREAKDOWN].")  # fmt: skip
        elif r == "business_use_exclusion":
            cited.append("EXC-COMMERCIAL-USE")
            s.append(f"The car was in {USE_TEXT[facts.use_at_loss]} and the policy has no "
                     "rideshare endorsement, so the business-use exclusion applies "
                     "[EXC-COMMERCIAL-USE].")  # fmt: skip
        elif r == "hit_and_run_report_condition":
            cited.append("COND-HIT-AND-RUN-POLICE-REPORT")
            when = (
                "was not reported to police"
                if not facts.police_report
                else f"was reported to police after {facts.police_report_hours:g} hours"
            )
            s.append(f"The hit-and-run {when}, but the policy requires a police report within "
                     f"{hours:g} hours [COND-HIT-AND-RUN-POLICE-REPORT].")  # fmt: skip
        elif r == "below_deductible":
            cited.append("LIM-LOSS-BELOW-DEDUCTIBLE")
            s.append(f"The covered loss of {money(x.gross_loss)} does not exceed the "
                     f"{money(x.deductible or 0.0)} deductible, so nothing is payable "
                     "[LIM-LOSS-BELOW-DEDUCTIBLE].")  # fmt: skip
        elif r == "late_notice_prejudice_review":
            cited.append("COND-LATE-NOTICE")
            s.append(f"The loss was reported {x.notice_days} days after it happened, more than "
                     f"the {jur.late_notice_days} days the policy allows [COND-LATE-NOTICE]. "
                     "Late notice only bars payment if it harmed the investigation, which needs "
                     "an adjuster's review.")  # fmt: skip
        elif r == "covered_loss":
            if x.hit_and_run_report_ok:
                cited.append("COND-HIT-AND-RUN-POLICE-REPORT")
                s.append(f"The hit-and-run was reported to police within "
                         f"{facts.police_report_hours:g} hours, which meets the {hours:g}-hour "
                         "condition [COND-HIT-AND-RUN-POLICE-REPORT].")  # fmt: skip
            part_clause = PART_CLAUSE[x.coverage_part]
            cited.append(part_clause)
            s.append(f"The policy carries {x.coverage_part} coverage [{part_clause}].")
            if x.total_loss:
                cited.append("LIM-TOTAL-LOSS")
                s.append(f"The repair estimate of {money(pkg.estimate_amount)} makes the car a "
                         f"total loss, so its actual cash value of {money(x.gross_loss)} "
                         "applies [LIM-TOTAL-LOSS].")  # fmt: skip
            else:
                s.append(f"The repair estimate is {money(x.gross_loss)}.")
            cited.append("LIM-DEDUCTIBLE")
            s.append(f"After the {money(x.deductible or 0.0)} {x.coverage_part} deductible "
                     f"[LIM-DEDUCTIBLE], we approve a payment of {money(x.payout)}.")  # fmt: skip
    if outcome == "deny":
        s.append("We deny the claim.")
    elif outcome == "escalate":
        s.append("We refer the claim to an adjuster.")
    return " ".join(s), list(dict.fromkeys(cited))


def true_reasons(row: pd.Series) -> list[str]:
    """Ground-truth reason codes; an approval carries none in the data, so it is a covered loss."""
    raw = row["gt_reasons"]
    reasons = [] if pd.isna(raw) else [r for r in str(raw).split(";") if r]
    return reasons or (["covered_loss"] if row["gt_decision"] == "approve" else [])


def reference_sample(row: pd.Series, jur: JurisdictionProfile, policy: PolicyDoc) -> Sample | None:
    """A clean judge case for one simulated claim, or None if no faithful reference exists."""
    reasons = true_reasons(row)
    outcome = str(row["gt_decision"])
    if not reasons or not set(reasons) <= SUPPORTED_REASONS:
        return None
    facts = true_facts(row)
    pkg = package(row)
    x = derive(facts, pkg, jur)
    truth = _truth(pkg, x)
    if not all(truth[r] for r in reasons):
        return None
    payout = x.payout if outcome == "approve" else None
    if outcome == "approve" and abs(x.payout - float(row["gt_payout"])) > 1.0:
        return None  # e.g. the estimate is inflated: the derived payout is not the true one
    explanation, cited = explain(facts, x, pkg, outcome, reasons, jur)
    if any(c not in policy.by_id for c in cited):
        raise KeyError(f"reference cites unknown clauses: {cited}")
    decision = {"outcome": outcome, "payout": payout, "reasons": reasons,
                "cited_clauses": cited}  # fmt: skip
    return Sample(
        id=str(row["claim_id"]),
        fields={
            "facts": facts.model_dump_json(),
            "deterministic": x.model_dump_json(),
            "clauses": clauses_text(policy.by_id, cited),
            "decision": json.dumps(decision),
            "explanation": explanation,
        },
    )


def render(sample: Sample) -> str:
    """The judge case exactly as the live judge node renders it."""
    f = sample.fields
    return render_judge_case(
        ClaimFacts.model_validate_json(f["facts"]),
        DerivedFacts.model_validate_json(f["deterministic"]),
        f["clauses"],
        json.loads(f["decision"]),
        f["explanation"],
    )


def select_samples(
    claims: pd.DataFrame,
    start: str,
    end: str,
    n: int,
    jur: JurisdictionProfile,
    policy: PolicyDoc,
    seed: int,
) -> list[Sample]:
    """Up to `n` clean samples from loss dates in [start, end), balanced across the primary
    reason (so denials and referrals are not swamped by routine approvals)."""
    pool = claims[(claims["loss_date"] >= start) & (claims["loss_date"] < end)]
    pool = pool.sample(frac=1.0, random_state=seed)
    by_reason: dict[str, list[Sample]] = {}
    for _, row in pool.iterrows():
        primary = (true_reasons(row) or [""])[0]
        if primary not in SUPPORTED_REASONS:
            continue
        bucket = by_reason.setdefault(primary, [])
        if len(bucket) >= n:
            if len(by_reason) == len(SUPPORTED_REASONS) and all(
                len(b) >= n for b in by_reason.values()
            ):
                break
            continue
        sample = reference_sample(row, jur, policy)
        if sample is not None:
            bucket.append(sample)
    return _round_robin(by_reason.values(), n)


def _round_robin(buckets: Iterable[list[Sample]], n: int) -> list[Sample]:
    queues = [list(b) for b in buckets if b]
    out: list[Sample] = []
    while queues and len(out) < n:
        for q in list(queues):
            if len(out) >= n:
                break
            out.append(q.pop(0))
            if not q:
                queues.remove(q)
    return out


# ---------------------------------------------------------------- insurance error types

INVENTED_FACTS = (
    "A neighbor's security camera recorded the whole incident.",
    "The other driver admitted fault at the scene.",
    "The claimant sent dashcam footage of the loss.",
    "The car had a new transmission installed last month.",
    "A tow-truck driver confirmed the damage was fresh.",
)
COVERAGE_SWAPS = (
    ("collision coverage", "comprehensive coverage"),
    ("excludes any loss", "covers any loss"),
    ("carries no collision or comprehensive coverage", "carries full collision coverage"),
)
CONCLUSION_SWAPS = (("we approve", "we deny"), ("We refer the claim", "We approve the claim"))
MATERIAL_SENTENCE = r"driven by|used for|police|reported|endorsement|excluded"

ERROR_TYPES: tuple[ErrorType, ...] = (
    ErrorType("wrong_amount", frozenset({"numbers_consistent"}), number_swap(),
              "an amount in the explanation does not match the deterministic numbers"),
    ErrorType("invented_fact", frozenset({"facts_supported"}), insert_sentence(INVENTED_FACTS),
              "a statement with no support in the claim facts"),
    ErrorType("misstated_coverage", frozenset({"faithful_to_clauses", "facts_supported"}),
              replace_phrase(COVERAGE_SWAPS), "what the policy covers is misstated"),
    ErrorType("omitted_material_fact", frozenset({"material_facts_addressed"}),
              drop_sentence(match=MATERIAL_SENTENCE), "a coverage-deciding fact is left out"),
    ErrorType("contradicted_conclusion", frozenset({"outcome_consistent"}),
              replace_phrase(CONCLUSION_SWAPS), "the conclusion contradicts the decision"),
)  # fmt: skip
