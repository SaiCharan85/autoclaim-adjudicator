"""The auto physical-damage line of business: the nodes the shared harness runs.

Code decides facts and numbers (facts.py); LLMs extract, read policy text and write reasoning;
the critic (code) and the judge (another model family) verify. Tool outputs always win.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from autoclaim.config import JurisdictionProfile
from autoclaim.core.decision import CheckResult, Decision, Issue
from autoclaim.core.judge import Judge
from autoclaim.core.lob import NodeResult
from autoclaim.core.memory import MemoryText
from autoclaim.core.state import ClaimState
from autoclaim.lines.auto.claim import ClaimPackage
from autoclaim.lines.auto.facts import BUSINESS_USES, ClaimFacts, DerivedFacts, derive
from autoclaim.lines.auto.fraud_tools import FraudSignals, FraudToolkit
from autoclaim.llm.client import LLMClient
from autoclaim.retrieval.corpus import PolicyDoc
from autoclaim.retrieval.retriever import HybridRetriever

DENY_CODES = {
    "no_physical_damage_coverage": "the policy has no collision/comprehensive coverage at all",
    "no_comprehensive_coverage": "a comprehensive-type loss on a policy without comprehensive",
    "excluded_driver": "a driver excluded by name on the policy was driving",
    "wear_and_tear_mechanical": "breakdown, wear and tear (not caused by a covered loss)",
    "business_use_exclusion": "rideshare/delivery use without the rideshare endorsement",
    "hit_and_run_report_condition": "hit-and-run not reported to police within 24 hours",
    "below_deductible": "the covered loss is at or below the deductible",
}
ESCALATE_CODES = {
    "late_notice_prejudice_review": "reported more than 30 days after the loss",
    "suspected_fraud_siu": "fraud signals strong enough for investigators",
    "missing_information": "a fact that decides coverage is missing or unclear",
}
APPROVE_CODES = {"covered_loss": "a covered loss, paid minus the deductible"}
REASON_CODES = DENY_CODES | ESCALATE_CODES | APPROVE_CODES

# ---------------------------------------------------------------- guardrails

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|system prompt|you are now|"
    r"approve this claim|disregard (the )?(policy|rules)",
    re.IGNORECASE,
)
MIN_NARRATIVE, MAX_NARRATIVE = 30, 6000


def guard(claim: dict[str, Any]) -> dict[str, Any]:
    """Schema/length checks, PII redaction, prompt-injection flags. Pure code, no LLM."""
    try:
        pkg = ClaimPackage.model_validate(claim)
    except ValidationError as exc:
        errs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        return {"rejected": True, "errors": errs[:10]}
    text = pkg.narrative.strip()
    if not MIN_NARRATIVE <= len(text) <= MAX_NARRATIVE:
        return {"rejected": True, "errors": [f"narrative length {len(text)} outside "
                                             f"{MIN_NARRATIVE}-{MAX_NARRATIVE}"]}  # fmt: skip
    pii = []
    if _SSN.search(text):
        pii.append("ssn")
        text = _SSN.sub("[REDACTED-SSN]", text)
    if _CARD.search(text):
        pii.append("payment_card")
        text = _CARD.sub("[REDACTED-CARD]", text)
    flags = ["suspicious_instructions"] if _INJECTION.search(text) else []
    return {"rejected": False, "errors": [], "pii_redacted": pii, "flags": flags,
            "narrative": text}  # fmt: skip


# ---------------------------------------------------------------- LLM contracts

INTAKE = """You extract facts from a policyholder's first notice of an auto loss. Use only what
the statement says; null when it is not stated. Map everyday words to the allowed values:
deer/animal/bird -> animal; another car hit us and drove off -> hit_and_run; found it damaged
while parked (by a car) -> parked_hit; keyed/smashed/graffiti -> vandalism; stolen/carjacked ->
theft; hail/storm/flood/falling branch -> hail_weather; windshield/window cracked by a rock ->
glass; engine/transmission failed by itself -> mechanical_breakdown; hit a pole/wall/debris ->
collision_object; crash with another vehicle -> collision_vehicle.
use_at_loss: Uber/Lyft/rideshare app on -> rideshare_active; DoorDash/Instacart/delivery app ->
delivery_active; to/from work -> commute; otherwise personal; unknown if unclear.
driver_name/relationship: the person who was driving and how the policyholder describes them
("self" if the policyholder drove).
police_report_hours: hours from the loss to the police report ("right away" 1, "same evening"
6, "next morning" 12, "next day" 24, "two days later" 48).
Resolve relative dates ("yesterday") against the report date.
missing_info: list which of these the statement does not give: loss_date, cause, driver,
use_at_loss, police_report, police_report_timing, injuries, damage_severity, witnesses."""

COVERAGE = """You are an auto claims coverage analyst for a US personal auto policy (physical
damage only). Decide which coverage part this loss falls under and whether any exclusion or
condition in the clauses applies. Cite ONLY clause ids from the list. The deterministic facts
(driver_role from the policy's driver lists, coverage_part from the carrier's cause mapping,
notice_days, hit_and_run_report_ok) are authoritative: never contradict them. covered:
yes = covered subject to the deductible; no = an exclusion, a missing coverage part or an unmet
condition removes coverage; needs_review = a decisive fact is missing or a human judgment is
required (for example late notice). Keep each finding under 30 words."""

FRAUD = """You are a fraud analyst. The fraud tools' numbers below are authoritative and must
not be restated differently. In under 80 words, explain to an adjuster what drives the score
and note any inconsistency between the claimant's statement and the record (timing, amounts,
story). Do not decide the claim."""

ADJUDICATE = f"""You adjudicate US auto physical-damage claims for Demo Mutual. Decide approve,
deny or escalate. Precedence: (1) deny if coverage does not apply (no coverage part, excluded
driver, wear/mechanical, business use without endorsement, unmet hit-and-run police condition);
(2) otherwise escalate if notice was late (a human must judge prejudice), fraud signals are
strong, or a decisive fact is missing; (3) otherwise approve and pay the deterministic payout
(deny with below_deductible if that payout is 0).
Rules: use only the facts, numbers and clause texts given; every deny cites at least one clause
id; payout must equal the deterministic payout for approve and be null otherwise; an unlisted
driver driving with permission is covered; a deer or other animal strike is comprehensive, not
collision. Explanation: plain English for the policyholder, under 120 words.
Reason codes (use exactly): {json.dumps(REASON_CODES)}"""


COVERAGE_KEYS = (
    "coverage_part",
    "covered",
    "exclusions_triggered",
    "conditions_unmet",
    "reasoning",
)


class ReasoningStep(BaseModel):
    clause_id: str
    finding: str = Field(max_length=300)


class CoverageResult(BaseModel):
    coverage_part: Literal["collision", "comprehensive", "none", "unknown"]
    covered: Literal["yes", "no", "needs_review"]
    exclusions_triggered: list[str] = Field(default_factory=list)
    conditions_unmet: list[str] = Field(default_factory=list)
    reasoning: list[ReasoningStep] = Field(min_length=1, max_length=8)


class FraudAssessment(BaseModel):
    summary: str = Field(max_length=700)
    inconsistencies: list[str] = Field(default_factory=list, max_length=5)


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def clauses_text(by_id: Mapping[str, Any], ids: Sequence[str]) -> str:
    """Full text of the cited clauses that exist, once each, in citation order."""
    out = []
    for cid in dict.fromkeys(ids):
        c = by_id.get(cid)
        if c is not None:
            out.append(f"[{c.id}] {c.title}: {c.text}")
    return "\n".join(out)


def render_judge_case(
    facts: ClaimFacts,
    derived: DerivedFacts,
    clauses_text: str,
    decision: dict[str, Any],
    explanation: str,
) -> str:
    """What the judge sees: evidence first (facts, numbers, cited clause text), then the
    decision and its explanation. Shared by the live judge node and the judge evals."""
    return (f"facts: {_compact(facts.model_dump(mode='json'))}\n"
            f"deterministic: {_compact(derived.model_dump(mode='json'))}\n"
            f"cited clauses:\n{clauses_text or '(none)'}\n"
            f"decision: {_compact({k: decision[k] for k in ('outcome', 'payout', 'reasons')})}\n"
            f"explanation: {explanation}")  # fmt: skip


# ---------------------------------------------------------------- the line


FewShot = Callable[[ClaimState], str]


@dataclass
class AutoLine:
    client: LLMClient
    policy: PolicyDoc
    retriever: HybridRetriever
    toolkit: FraudToolkit
    judge_impl: Judge
    jurisdiction: JurisdictionProfile
    fraud_review_score: float
    top_k: int = 5
    graph_hops: int = 1
    max_expanded: int = 6
    few_shot: FewShot | None = None
    name: str = "auto_physical_damage"
    _by_id: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._by_id = self.policy.by_id

    # -------------------------------------------------------- helpers

    @staticmethod
    def package(state: ClaimState) -> ClaimPackage:
        return ClaimPackage.model_validate(state["claim"])

    def _facts(self, state: ClaimState) -> tuple[ClaimFacts, DerivedFacts]:
        f = state["facts"]
        return ClaimFacts.model_validate(f["extracted"]), DerivedFacts.model_validate(f["derived"])

    def _clauses_text(self, ids: Sequence[str]) -> str:
        return clauses_text(self._by_id, ids)

    @staticmethod
    def _declarations(pkg: ClaimPackage) -> dict[str, Any]:
        p = pkg.policy
        return {"coverage": p.coverage, "collision_deductible": p.collision_deductible,
                "comprehensive_deductible": p.comprehensive_deductible,
                "rideshare_endorsement": p.rideshare_endorsement,
                "policy_start_date": p.policy_start_date, "policy_state": p.policy_state,
                "vehicle_acv": p.vehicle.actual_cash_value,
                "lienholder": p.vehicle.lienholder}  # fmt: skip

    # -------------------------------------------------------- nodes

    def guardrails(self, state: ClaimState) -> NodeResult:
        g = guard(state["claim"])
        return NodeResult(
            {"guardrails": g}, outputs={k: g.get(k) for k in ("rejected", "pii_redacted", "flags")}
        )

    def intake(self, state: ClaimState) -> NodeResult:
        pkg = self.package(state)
        user = (f"channel: {pkg.channel}\nreport_date: {pkg.report_date}\nstatement:\n"
                f"{state['guardrails']['narrative']}")  # fmt: skip
        res = self.client.structured("intake", INTAKE, user, ClaimFacts)
        derived = derive(res.value, pkg, self.jurisdiction)
        facts = {"extracted": res.value.model_dump(mode="json"),
                 "derived": derived.model_dump(mode="json")}  # fmt: skip
        return NodeResult({"facts": facts}, [res.meta],
                          {"cause": res.value.cause, "driver_role": derived.driver_role,
                           "missing": res.value.missing_info})  # fmt: skip

    def retrieval_query(self, facts: ClaimFacts, derived: DerivedFacts) -> str:
        parts = [
            facts.summary,
            f"cause {facts.cause.replace('_', ' ')}",
            f"driver {facts.driver_relationship or 'unknown'} ({derived.driver_role})",
        ]
        if facts.use_at_loss in BUSINESS_USES:
            parts.append("rideshare or delivery app use for a fee")
        if derived.late_notice:
            parts.append("late notice of loss reported after 30 days")
        if facts.cause == "hit_and_run":
            parts.append("hit-and-run police report")
        return "; ".join(parts)

    def coverage(self, state: ClaimState) -> NodeResult:
        pkg = self.package(state)
        facts, derived = self._facts(state)
        hits = self.retriever.search(self.retrieval_query(facts, derived), self.top_k,
                                     self.graph_hops, max_expanded=self.max_expanded)  # fmt: skip
        ids = [h.clause.id for h in hits]
        user = (f"declarations: {_compact(self._declarations(pkg))}\n"
                f"facts: {_compact(facts.model_dump(mode='json'))}\n"
                f"deterministic: {_compact(derived.model_dump(mode='json'))}\n"
                f"clauses:\n{self._clauses_text(ids)}")  # fmt: skip
        res = self.client.structured("coverage", COVERAGE, user, CoverageResult)
        cov = res.value
        known = set(ids)
        unknown = [s.clause_id for s in cov.reasoning if s.clause_id not in known]
        out = cov.model_dump(mode="json") | {"retrieved": ids, "uncited_ids_dropped": unknown}
        out["reasoning"] = [s for s in out["reasoning"] if s["clause_id"] in known]
        return NodeResult({"coverage": out}, [res.meta],
                          {"covered": cov.covered, "part": cov.coverage_part,
                           "retrieved": len(ids), "dropped_ids": unknown})  # fmt: skip

    def fraud_record(self, state: ClaimState) -> dict[str, Any]:
        """The claim in the fraud model's (simulator) schema; unknowns stay missing (NaN)."""
        pkg = self.package(state)
        facts, derived = self._facts(state)
        p, v = pkg.policy, pkg.policy.vehicle
        loss = facts.loss_date
        b = {True: 1.0, False: 0.0, None: float("nan")}
        return {
            "loss_date": pd.Timestamp(loss) if loss else pd.NaT,
            "policy_start_date": pd.Timestamp(p.policy_start_date),
            "report_date": pd.Timestamp(pkg.report_date),
            "vehicle_make": v.make, "body_class": v.body_class, "cause": facts.cause,
            "n_vehicles": facts.n_vehicles, "injury_count": facts.injuries,
            "damage_extent": facts.damage_severity, "towed": b[facts.towed],
            "at_fault": b[facts.insured_at_fault],
            "vehicle_role": "parked" if facts.vehicle_parked else "in_transport",
            "loss_state": p.policy_state, "policy_state": p.policy_state, "coverage": p.coverage,
            "collision_deductible": p.collision_deductible,
            "comprehensive_deductible": p.comprehensive_deductible or 0.0,
            "rideshare_endorsement": p.rideshare_endorsement, "driver_role": derived.driver_role,
            "use_at_loss": facts.use_at_loss, "address_change_days": p.address_change_days,
            "prior_claims_3y": p.prior_claims_3y,
            "vehicle_age": (loss.year - v.year) if (loss and v.year) else None,
            "vehicle_acv": v.actual_cash_value, "financed": v.lienholder is not None,
            "adas": v.adas, "police_report": facts.police_report,
            "police_report_hours": facts.police_report_hours,
            "witness_count": facts.witnesses, "notice_days": derived.notice_days,
            "attorney_involved": facts.attorney_involved, "claimed_amount": pkg.estimate_amount,
        }  # fmt: skip

    def fraud(self, state: ClaimState) -> NodeResult:
        sig: FraudSignals = self.toolkit.assess(self.fraud_record(state))
        out: dict[str, Any] = {"signals": sig.model_dump(mode="json"),
                               "model_score": sig.model_score}  # fmt: skip
        calls = []
        if sig.model_score >= self.fraud_review_score:  # the LLM only explains; skip if moot
            pkg = self.package(state)
            user = (f"signals: {_compact(sig.model_dump(mode='json'))}\n"
                    f"statement: {state['guardrails']['narrative']}\n"
                    f"estimate: {pkg.estimate_amount}; vehicle value: "
                    f"{pkg.policy.vehicle.actual_cash_value}")  # fmt: skip
            res = self.client.structured("fraud", FRAUD, user, FraudAssessment)
            out["assessment"] = res.value.model_dump()
            calls.append(res.meta)
        else:
            reasons = ", ".join(f"{r.feature} {r.direction}" for r in sig.top_reasons[:3])
            out["assessment"] = {
                "summary": f"Low fraud score {sig.model_score:.2f} (top factors: {reasons}).",
                "inconsistencies": [],
            }
        return NodeResult({"fraud": out}, calls,
                          {"score": sig.model_score, "flags": [h.rule_id for h in sig.rules_fired],
                           "llm": bool(calls)})  # fmt: skip

    def adjudicate(self, state: ClaimState) -> NodeResult:
        pkg = self.package(state)
        facts, derived = self._facts(state)
        cov, fr = state["coverage"], state["fraud"]
        cited = (
            [s["clause_id"] for s in cov["reasoning"]]
            + cov["exclusions_triggered"]
            + (cov["conditions_unmet"])
        )
        clause_ids = list(dict.fromkeys(cited + cov["retrieved"]))
        sig = fr["signals"]
        fraud_view = {"score": sig["model_score"], "anomaly": sig["anomaly_score"],
                      "red_flags": [h["rule_id"] for h in sig["rules_fired"]],
                      "assessment": fr["assessment"]["summary"]}  # fmt: skip
        parts = [f"declarations: {_compact(self._declarations(pkg))}",
                 f"facts: {_compact(facts.model_dump(mode='json'))}",
                 f"deterministic: {_compact(derived.model_dump(mode='json'))}",
                 f"coverage_analysis: {_compact({k: cov[k] for k in COVERAGE_KEYS})}",
                 f"fraud: {_compact(fraud_view)}",
                 f"clauses:\n{self._clauses_text(clause_ids)}"]  # fmt: skip
        if self.few_shot is not None:
            examples = self.few_shot(state)
            if examples:
                parts.insert(0, f"similar past decisions (with adjuster corrections):\n{examples}")
        if state.get("issues"):
            fixes = "; ".join(f"[{i['source']}:{i['code']}] {i['detail']}" for i in state["issues"])
            parts.append(f"your previous decision was rejected; fix: {fixes}")
        res = self.client.structured("adjudicator", ADJUDICATE, "\n".join(parts), Decision)
        d = res.value
        stated = d.payout
        d.payout = derived.payout if d.outcome == "approve" else None  # tool numbers win
        out = d.model_dump(mode="json") | {"llm_stated_payout": stated,
                                           "model_family": res.meta.family}  # fmt: skip
        return NodeResult({"decision": out}, [res.meta],
                          {"outcome": d.outcome, "reasons": d.reasons, "payout": d.payout,
                           "payout_overwritten": stated is not None and d.payout is not None
                           and abs(stated - d.payout) > 1})  # fmt: skip

    # -------------------------------------------------------- checks

    def check_decision(self, state: ClaimState) -> CheckResult:
        """Independent code checks: clauses exist, codes valid, no contradiction with the facts."""
        d = state["decision"]
        _, x = self._facts(state)
        pkg = self.package(state)
        issues: list[Issue] = []

        def bad(code: str, detail: str) -> None:
            issues.append(Issue(source="critic", code=code, detail=detail))

        unknown = [c for c in d["cited_clauses"] if c not in self._by_id]
        if unknown:
            bad("unknown_clause", f"cited clauses do not exist: {unknown}")
        codes = set(d["reasons"])
        if codes - set(REASON_CODES):
            bad("unknown_reason_code", f"not allowed: {sorted(codes - set(REASON_CODES))}")
        outcome = d["outcome"]
        if outcome == "deny":
            if not d["cited_clauses"]:
                bad("deny_without_clause", "a denial must cite at least one policy clause")
            if not codes & set(DENY_CODES):
                bad("deny_without_reason", "a denial needs a deny reason code")
        if outcome == "escalate" and not codes & set(ESCALATE_CODES):
            bad("escalate_without_reason", "an escalation needs an escalation reason code")
        if outcome == "approve":
            blockers = {
                "no_physical_damage_coverage": pkg.policy.coverage == "liability_only",
                "no_coverage_part": not x.part_on_policy,
                "excluded_driver": x.driver_role == "excluded_driver",
                "business_use_exclusion": x.business_use_without_endorsement,
                "hit_and_run_report_condition": x.hit_and_run_report_ok is False,
                "below_deductible": x.payout <= 0,
            }
            for code, hit in blockers.items():
                if hit:
                    bad("approve_contradicts_facts", f"approve conflicts with {code}")
        # a deny reason must be true according to the deterministic facts
        truth = {
            "no_physical_damage_coverage": pkg.policy.coverage == "liability_only",
            "no_comprehensive_coverage": x.coverage_part == "comprehensive"
            and not x.part_on_policy,
            "excluded_driver": x.driver_role == "excluded_driver",
            "wear_and_tear_mechanical": x.coverage_part == "none",
            "business_use_exclusion": x.business_use_without_endorsement,
            "hit_and_run_report_condition": x.hit_and_run_report_ok is False,
            "below_deductible": x.payout <= 0,
            "late_notice_prejudice_review": x.late_notice,
        }
        for code in codes & set(truth):
            if not truth[code]:
                bad("reason_contradicts_facts", f"{code} is not supported by the facts")
        return CheckResult(passed=not issues, issues=issues)

    def critic(self, state: ClaimState) -> NodeResult:
        res = self.check_decision(state)
        return NodeResult(
            {"critic": res.model_dump()},
            outputs={"passed": res.passed, "issues": [i.code for i in res.issues]},
        )

    def judge_case(self, state: ClaimState) -> str:
        d = state["decision"]
        facts, derived = self._facts(state)
        return render_judge_case(
            facts, derived, self._clauses_text(d["cited_clauses"]), d, d["explanation"]
        )

    def judge(self, state: ClaimState) -> NodeResult:
        family = state["decision"].get("model_family") or ""
        res, calls = self.judge_impl.evaluate(self.judge_case(state), frozenset({family}))
        return NodeResult(
            {"judge": res.model_dump()},
            calls,
            {"passed": res.passed, "issues": [i.code for i in res.issues]},
        )

    # -------------------------------------------------------- feedback memory

    def memory_text(self, state: ClaimState) -> MemoryText:
        """How a case is described in feedback memory: the coverage-deciding facts (what makes
        two claims similar), dated by the loss date for the memory's leakage cutoff."""
        pkg = self.package(state)
        if "facts" not in state:  # failed before intake: only the story is known
            return MemoryText(f"unprocessed claim: {pkg.narrative[:300]}", pkg.report_date)
        facts, x = self._facts(state)
        on_policy = "on" if x.part_on_policy else "NOT on"
        notice = f"notice {x.notice_days} days{' (late)' if x.late_notice else ''}"
        text = (f"{facts.cause} loss; {x.coverage_part} coverage {on_policy} policy; "
                f"driver {x.driver_role}; use {facts.use_at_loss}; {notice}; "
                f"estimate {pkg.estimate_amount:.0f}, approval would pay {x.payout:.0f}; "
                f"{facts.summary}")  # fmt: skip
        return MemoryText(text, facts.loss_date or pkg.report_date)

    # -------------------------------------------------------- routing inputs

    def hard_escalations(self, state: ClaimState) -> list[str]:
        out = list(state.get("guardrails", {}).get("flags", []))
        if "facts" not in state:
            return out
        facts, derived = self._facts(state)
        decision = state.get("decision") or {}
        denied = decision.get("outcome") == "deny"
        if derived.late_notice and not denied:  # a clear exclusion wins over late notice
            out.append("late_notice")
        if facts.loss_date is None:
            out.append("loss_date_unknown")
        if facts.cause == "unknown":
            out.append("cause_unknown")
        if derived.driver_role == "unknown":
            out.append("driver_unknown")
        if (
            facts.cause == "hit_and_run"
            and facts.police_report
            and (facts.police_report_hours is None)
        ):
            out.append("hit_and_run_report_timing_unknown")
        return out

    def fraud_score(self, state: ClaimState) -> float | None:
        fr = state.get("fraud")
        return None if not fr else float(fr["model_score"])

    def case_summary(self, state: ClaimState) -> dict[str, Any]:
        """Everything an adjuster needs on one screen; deterministic, no LLM call."""
        out: dict[str, Any] = {"claim_id": state["claim_id"]}
        if "claim" in state:
            pkg = self.package(state)
            out["declarations"] = self._declarations(pkg)
            out["estimate"] = pkg.estimate_amount
            out["statement"] = state.get("guardrails", {}).get("narrative", pkg.narrative)
        for key in ("facts", "coverage", "decision", "critic", "judge", "failsafe"):
            if state.get(key):
                out[key] = state[key]
        if state.get("fraud"):
            fr = state["fraud"]
            out["fraud"] = {
                "score": fr["model_score"],
                "assessment": fr["assessment"],
                "top_reasons": fr["signals"]["top_reasons"],
                "red_flags": [h["rule_id"] for h in fr["signals"]["rules_fired"]],
            }
        return out
