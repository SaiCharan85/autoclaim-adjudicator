"""The flood line of business: plugs into the shared harness core with no core changes.

Flood first notices arrive as structured records (no narrative), so intake and coverage are code.
Code decides the outcome and every number; the adjudicator node only explains it (a template, or
optionally an LLM writing the explanation around the code's numbers), the critic checks it, and
the judge (JudgeKit) grades the explanation when an LLM wrote it. Anything needing judgment
(cause unclear, damage undocumented, a policy-history conflict) escalates to an adjuster.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from autoclaim.core.decision import CheckResult, Issue
from autoclaim.core.judge import Judge
from autoclaim.core.lob import NodeResult
from autoclaim.core.memory import MemoryText
from autoclaim.core.state import ClaimState
from autoclaim.lines.flood.claim import FloodClaim, FloodCoverage, assess, decide
from autoclaim.llm.client import LLMClient
from autoclaim.retrieval.corpus import PolicyDoc

EXPLAIN = """You write the explanation of a flood claim decision that has ALREADY been made by code.
Use only the facts, numbers and clause texts given; never change the outcome or any amount, and
never add facts. Cite clause ids in square brackets. Two to five plain sentences."""


class Explanation(BaseModel):
    explanation: str = Field(min_length=20, max_length=1200)


def money(x: float | None) -> str:
    return "-" if x is None else f"${x:,.2f}"


def template_explanation(claim: FloodClaim, cov: FloodCoverage, outcome: str, payout: float | None
                         ) -> str:  # fmt: skip
    """A correct explanation built from the code's own numbers (no LLM)."""
    s = [f"The loss on {claim.date_of_loss} was {cov.cause}."]
    for p, clause in ((cov.building, "FLD-BUILDING"), (cov.contents, "FLD-CONTENTS")):
        if not p.covered:
            s.append(f"The policy has no {p.part} coverage.")
        elif p.damage is not None and p.deductible is not None:
            s.append(f"{p.part.capitalize()} damage of {money(p.damage)} minus the "
                     f"{money(p.deductible)} {p.part} deductible [FLD-DEDUCTIBLE] gives "
                     f"{money(p.payout)} [{clause}].")  # fmt: skip
    if outcome == "approve":
        s.append(f"We approve a payment of {money(payout)}, within the policy limits [FLD-LIMITS].")
    elif outcome == "deny":
        s.append("Neither part exceeds its deductible, so nothing is payable "
                 "[FLD-BELOW-DEDUCTIBLE]. We deny the claim.")  # fmt: skip
    else:
        s.append("An adjuster must review this claim: " + "; ".join(
            cov.judgment_needed or ["the policy history must be checked"]) + ".")  # fmt: skip
    return " ".join(s)


@dataclass
class FloodLine:
    policy: PolicyDoc
    client: LLMClient | None = None  # None: template explanations, no LLM calls at all
    judge_impl: Judge | None = None  # None: explanations are code-built, nothing to judge
    name: str = "residential_flood"
    _by_id: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._by_id = self.policy.by_id

    @staticmethod
    def claim(state: ClaimState) -> FloodClaim:
        return FloodClaim.model_validate(state["claim"])

    def guardrails(self, state: ClaimState) -> NodeResult:
        try:
            self.claim(state)
        except ValidationError as exc:
            errors = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()][:5]
            return NodeResult({"guardrails": {"rejected": True, "errors": errors, "flags": []}},
                              outputs={"rejected": True})  # fmt: skip
        return NodeResult({"guardrails": {"rejected": False, "errors": [], "flags": []}},
                          outputs={"rejected": False})  # fmt: skip

    def intake(self, state: ClaimState) -> NodeResult:
        c = self.claim(state)  # structured first notice: facts are read, not extracted
        return NodeResult({"facts": {"claim": c.model_dump(mode="json")}},
                          outputs={"causes": c.cause_codes})  # fmt: skip

    def coverage(self, state: ClaimState) -> NodeResult:
        cov = assess(self.claim(state))
        return NodeResult(
            {"coverage": cov.model_dump(mode="json")},
            outputs={"total": cov.total, "judgment": cov.judgment_needed},
        )

    def fraud(self, state: ClaimState) -> NodeResult:
        return NodeResult({"fraud": {"score": None, "note": "no flood fraud model"}})

    def adjudicate(self, state: ClaimState) -> NodeResult:
        c = self.claim(state)
        cov = FloodCoverage.model_validate(state["coverage"])
        outcome, reasons, payout = decide(cov)
        explanation, calls = template_explanation(c, cov, outcome, payout), []
        if self.client is not None:
            user = (f"claim: {c.model_dump_json()}\ncoverage (code): {cov.model_dump_json()}\n"
                    f"decision (code): {json.dumps({'outcome': outcome, 'payout': payout})}\n"
                    f"clauses:\n{self._clauses(cov.clauses)}")  # fmt: skip
            res = self.client.structured("adjudicator", EXPLAIN, user, Explanation)
            explanation, calls = res.value.explanation, [res.meta]
        decision: dict[str, Any] = {"outcome": outcome, "payout": payout, "reasons": reasons,
                    "cited_clauses": cov.clauses, "explanation": explanation,
                    "confidence": 1.0, "self_check": {},
                    "model_family": calls[0].family if calls else ""}  # fmt: skip
        return NodeResult({"decision": decision}, calls,
                          {"outcome": outcome, "payout": payout, "reasons": reasons})  # fmt: skip

    def _clauses(self, ids: list[str]) -> str:
        return "\n".join(f"[{i}] {self._by_id[i].title}: {self._by_id[i].text}"
                         for i in dict.fromkeys(ids) if i in self._by_id)  # fmt: skip

    def critic(self, state: ClaimState) -> NodeResult:
        d, cov = state["decision"], FloodCoverage.model_validate(state["coverage"])
        issues = []
        unknown = [x for x in d["cited_clauses"] if x not in self._by_id]
        if unknown:
            issues.append(Issue(source="critic", code="unknown_clause", detail=str(unknown)))
        if d["outcome"] == "deny" and not d["cited_clauses"]:
            issues.append(Issue(source="critic", code="deny_without_clause", detail="cite one"))
        if d["outcome"] == "approve":
            if abs((d["payout"] or 0) - cov.total) > 0.01:
                issues.append(Issue(source="critic", code="payout_not_code_number",
                                    detail=f"{d['payout']} vs {cov.total}"))  # fmt: skip
            limit = (cov.building.payout <= self.claim(state).building_limit
                     and cov.contents.payout <= self.claim(state).contents_limit)  # fmt: skip
            if not limit:
                issues.append(Issue(source="critic", code="over_limit", detail="payout > limit"))
        res = CheckResult(passed=not issues, issues=issues)
        return NodeResult(
            {"critic": res.model_dump()},
            outputs={"passed": res.passed, "issues": [i.code for i in issues]},
        )

    def judge(self, state: ClaimState) -> NodeResult:
        if self.judge_impl is None or self.client is None:
            return NodeResult({"judge": {"passed": True, "issues": []}},
                              outputs={"skipped": "explanation built by code"})  # fmt: skip
        d = state["decision"]
        case = (f"claim: {json.dumps(state['claim'], default=str)}\n"
                f"deterministic: {json.dumps(state['coverage'])}\n"
                f"cited clauses:\n{self._clauses(d['cited_clauses'])}\n"
                f"decision: {json.dumps({k: d[k] for k in ('outcome', 'payout', 'reasons')})}\n"
                f"explanation: {d['explanation']}")  # fmt: skip
        res, calls = self.judge_impl.evaluate(case, frozenset({d.get("model_family") or ""}))
        return NodeResult(
            {"judge": res.model_dump()},
            calls,
            {"passed": res.passed, "issues": [i.code for i in res.issues]},
        )

    def hard_escalations(self, state: ClaimState) -> list[str]:
        return list(state.get("guardrails", {}).get("flags", []))

    def fraud_score(self, state: ClaimState) -> float | None:
        return None

    def case_summary(self, state: ClaimState) -> dict[str, Any]:
        out: dict[str, Any] = {"claim_id": state["claim_id"]}
        for key in ("claim", "coverage", "decision", "critic", "judge", "failsafe"):
            if state.get(key):
                out[key] = state[key]
        return out

    def memory_text(self, state: ClaimState) -> MemoryText:
        c = self.claim(state)
        cov = state.get("coverage") or {}
        text = (f"flood {', '.join(c.cause_codes) or 'unknown cause'}; zone {c.flood_zone}; "
                f"building limit {c.building_limit:.0f}, contents limit {c.contents_limit:.0f}; "
                f"computed payout {cov.get('total', 0):.0f}; "
                f"judgment: {', '.join(cov.get('judgment_needed', [])) or 'none'}")  # fmt: skip
        return MemoryText(text, c.date_of_loss)
