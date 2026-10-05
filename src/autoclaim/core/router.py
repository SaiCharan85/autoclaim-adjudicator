"""Deterministic routing: hard rules first, then thresholds. No LLM decides where a claim goes."""

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["auto_decide", "human_review"]


class RouterConfig(BaseModel):
    min_confidence: float = Field(ge=0, le=1)
    fraud_review_score: float = Field(ge=0, le=1)  # approve + score >= this -> human
    authority_limit_usd: float = Field(gt=0)  # auto-approval limit per claim


class RouteDecision(BaseModel):
    route: Route
    reasons: list[str]


def route_claim(
    decision: dict[str, Any] | None,
    hard_escalations: Sequence[str],
    fraud_score: float | None,
    checks_passed: bool,
    failsafe: str | None,
    cfg: RouterConfig,
) -> RouteDecision:
    reasons: list[str] = []
    if failsafe:
        reasons.append(f"failsafe:{failsafe}")
    if decision is None:
        reasons.append("no_decision")
    if not checks_passed:
        reasons.append("checks_failed_after_retries")
    reasons += [f"hard_rule:{r}" for r in hard_escalations]
    if decision is not None:
        if decision["outcome"] == "escalate":
            reasons.append("adjudicator_escalated")
        if decision["confidence"] < cfg.min_confidence:
            reasons.append("low_confidence")
        if decision["outcome"] == "approve":
            if fraud_score is not None and fraud_score >= cfg.fraud_review_score:
                reasons.append("approve_with_high_fraud_score")
            if (decision.get("payout") or 0) > cfg.authority_limit_usd:
                reasons.append("over_authority_limit")
    return RouteDecision(route="human_review" if reasons else "auto_decide", reasons=reasons)
