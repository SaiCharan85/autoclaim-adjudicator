"""Score flood harness runs against what FEMA actually paid (real outcomes).

Headline metrics (fixed 2026-10-05, before any graph run):
- auto_rate: claims decided without an adjuster
- auto_agreement: auto-approved claims that were in fact paid + auto-denied claims that were not
- wrongly_denied_rate / wrongly_paid_rate: the two costly mistakes, per claim
- acv_within_5pct: actual-cash-value claims whose auto payout is within 5% of the real payment
- rc_first_payment_ratio: replacement-cost claims: our (ACV-first) payment / the final total
"""

from collections.abc import Callable, Sequence
from typing import Any

import judgekit
import pandas as pd
from pydantic import BaseModel, ConfigDict

HEADLINE = ("auto_rate", "auto_agreement", "wrongly_denied_rate", "wrongly_paid_rate",
            "acv_within_5pct", "rc_first_payment_ratio")  # fmt: skip


class FloodScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    auto: bool
    outcome: str
    payout: float | None
    actual_paid: float
    replacement_cost: bool | None
    route_reasons: tuple[str, ...]

    @property
    def agrees(self) -> bool:
        if self.outcome == "approve":
            return self.actual_paid > 0
        if self.outcome == "deny":
            return self.actual_paid <= 0
        return False


def actual_paid(row: pd.Series) -> float:
    parts = ("amountPaidOnBuildingClaim", "amountPaidOnContentsClaim")
    return float(sum(0.0 if pd.isna(row[k]) else row[k] for k in parts))


def score(final_or_interrupt: dict[str, Any], row: pd.Series, replacement_cost: bool | None
          ) -> FloodScore:  # fmt: skip
    """Score a graph result: `final` (auto-decided) or a pending human review (interrupt)."""
    if "final" in final_or_interrupt:
        final = final_or_interrupt["final"]
        return FloodScore(claim_id=final_or_interrupt["claim_id"], auto=True,
                          outcome=final["outcome"], payout=final.get("payout"),
                          actual_paid=actual_paid(row), replacement_cost=replacement_cost,
                          route_reasons=())  # fmt: skip
    req = final_or_interrupt["__interrupt__"][0].value
    prop = req.get("proposed_decision") or {}
    return FloodScore(claim_id=req["claim_id"], auto=False, outcome="human",
                      payout=prop.get("payout"), actual_paid=actual_paid(row),
                      replacement_cost=replacement_cost,
                      route_reasons=tuple(req.get("route_reasons", [])))  # fmt: skip


def metric(scores: Sequence[FloodScore], name: str) -> float | None:
    n = len(scores)
    autos = [s for s in scores if s.auto]
    acv = [s for s in autos if s.outcome == "approve" and s.replacement_cost is False]
    rc = [s for s in autos if s.outcome == "approve" and s.replacement_cost and s.actual_paid > 0]
    if name == "auto_rate":
        return len(autos) / n if n else None
    if name == "auto_agreement":
        return sum(s.agrees for s in autos) / len(autos) if autos else None
    if name == "wrongly_denied_rate":
        return (
            sum(s.auto and s.outcome == "deny" and s.actual_paid > 0 for s in scores) / n
            if n
            else None
        )
    if name == "wrongly_paid_rate":
        return (
            sum(s.auto and s.outcome == "approve" and s.actual_paid <= 0 for s in scores) / n
            if n
            else None
        )
    if name == "acv_within_5pct":
        ok = [abs((s.payout or 0) - s.actual_paid) <= 0.05 * max(s.actual_paid, 1.0) for s in acv]
        return sum(ok) / len(ok) if ok else None
    if name == "rc_first_payment_ratio":
        ratios = sorted((s.payout or 0) / s.actual_paid for s in rc)
        return ratios[len(ratios) // 2] if ratios else None  # median
    raise ValueError(f"unknown metric {name!r}")


def summarize(scores: Sequence[FloodScore], n_boot: int = 1000, seed: int = 0
              ) -> dict[str, judgekit.Estimate]:  # fmt: skip
    def fn(name: str) -> Callable[[Sequence[FloodScore]], float | None]:
        def compute(units: Sequence[FloodScore]) -> float | None:
            return metric(units, name)

        return compute

    return {m: judgekit.bootstrap(list(scores), fn(m), n_boot, 0.05, seed) for m in HEADLINE}


def route_counts(scores: Sequence[FloodScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in scores:
        for r in s.route_reasons:
            counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
