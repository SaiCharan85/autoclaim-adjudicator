"""Line-agnostic decision, critic and judge contracts."""

from typing import Literal

from pydantic import BaseModel, Field

Outcome = Literal["approve", "deny", "escalate"]


class SelfCheck(BaseModel):
    """The adjudicator's own checklist (verified independently by the critic and judge)."""

    every_fact_from_claim: bool
    every_denial_cites_clause: bool
    numbers_from_tools: bool
    fraud_signals_considered: bool


class Decision(BaseModel):
    outcome: Outcome
    payout: float | None = Field(default=None, ge=0, description="USD; only for approve")
    reasons: list[str] = Field(default_factory=list, description="reason codes")
    cited_clauses: list[str] = Field(default_factory=list)
    explanation: str = Field(max_length=1500)
    confidence: float = Field(ge=0, le=1)
    self_check: SelfCheck


class Issue(BaseModel):
    source: Literal["critic", "judge"]
    code: str
    detail: str


class CheckResult(BaseModel):
    passed: bool
    issues: list[Issue] = Field(default_factory=list)


class HumanDecision(BaseModel):
    """What an adjuster (console or oracle) sends back to resume an interrupted claim."""

    outcome: Outcome
    payout: float | None = Field(default=None, ge=0)
    reason: str
    adjuster: str
