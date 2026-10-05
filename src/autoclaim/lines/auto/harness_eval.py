"""Harness evaluation against simulation truth. EVALUATION ONLY (the harness never imports it).

Each claim's final state is scored against the ground truth: did the harness decide it alone,
was that decision right, what did it pay, and would a coverage trap or a fraud have slipped
through? Escalated claims are answered by the simulated adjuster, so the loop completes; their
PROPOSED decision is scored too (it is what feedback memory learns from).

Plain-English trace: deer strike on a collision-only policy (a trap) -> the harness denies it
alone -> truth is deny -> correct auto-decision, trap caught. The same claim auto-approved for
$1,650 -> wrong auto-approval, trap leaked, $1,650 overpaid.

The headline metrics are fixed here, BEFORE any run (no picking metrics after seeing results);
arms are compared on the same claims with a paired bootstrap.
"""

from collections.abc import Callable, Sequence
from typing import Any

import judgekit
import pandas as pd
from pydantic import BaseModel, ConfigDict

from autoclaim.core.lob import LineOfBusiness, NodeResult
from autoclaim.core.state import ClaimState

# Fixed in advance (2026-10-05). Direction: higher is better unless named in LOWER_IS_BETTER.
HEADLINE = (
    "auto_rate",  # share of claims decided without a human
    "auto_accuracy",  # share of auto-decisions whose outcome matches the truth
    "wrong_auto_approve_rate",  # auto-approved claims the truth denies or escalates, per claim
    "trap_leak_rate",  # coverage traps decided wrongly without a human, per trap claim
    "fraud_auto_paid_rate",  # true frauds auto-approved, per fraud claim
    "proposal_accuracy",  # every claim: the harness's own proposal vs the truth
    "calls_per_claim",
)
LOWER_IS_BETTER = frozenset({"wrong_auto_approve_rate", "trap_leak_rate", "fraud_auto_paid_rate",
                             "calls_per_claim"})  # fmt: skip


class Truth(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: str
    payout: float | None
    reasons: tuple[str, ...]
    traps: tuple[str, ...]
    is_fraud: bool


def truth_of(row: pd.Series) -> Truth:
    raw = row["gt_reasons"]
    reasons = () if pd.isna(raw) else tuple(r for r in str(raw).split(";") if r)
    traps_raw = row["gt_traps"]
    traps = () if pd.isna(traps_raw) else tuple(t for t in str(traps_raw).split(";") if t)
    outcome = str(row["gt_decision"])
    return Truth(
        outcome=outcome,
        payout=float(row["gt_payout"]) if outcome == "approve" else None,
        reasons=reasons or (("covered_loss",) if outcome == "approve" else ()),
        traps=traps,
        is_fraud=bool(row["gt_is_fraud"]),
    )


class ClaimScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    arm: str
    auto: bool  # decided without a human
    final_outcome: str
    final_payout: float | None
    proposed_outcome: str | None
    truth: Truth
    route_reasons: tuple[str, ...]
    failsafe: bool
    llm_calls: int
    tokens: int

    @property
    def correct(self) -> bool:
        return self.final_outcome == self.truth.outcome

    @property
    def wrong_auto_approve(self) -> bool:
        return self.auto and self.final_outcome == "approve" and self.truth.outcome != "approve"

    @property
    def trap_leaked(self) -> bool:
        return bool(self.truth.traps) and self.auto and not self.correct

    @property
    def fraud_auto_paid(self) -> bool:
        return self.truth.is_fraud and self.auto and self.final_outcome == "approve"

    @property
    def overpaid(self) -> float:
        """Dollars paid that the truth would not pay (wrong approval or a too-high payout)."""
        if not self.auto or self.final_outcome != "approve":
            return 0.0
        paid = self.final_payout or 0.0
        return max(paid - (self.truth.payout or 0.0), 0.0)


QUOTA_MARKERS = ("daily safety limit", "rate limited", "HTTP 503", "HTTP 429")


def quota_exhausted(final_state: dict[str, Any]) -> bool:
    """The claim failed safe only because free-tier quota ran out (not a real harness outcome):
    an evaluation must stop and resume later instead of scoring it as an escalation."""
    reason = str(final_state.get("failsafe") or "")
    return "every model failed" in reason and any(m in reason for m in QUOTA_MARKERS)


def score(final_state: dict[str, Any], truth: Truth, arm: str) -> ClaimScore:
    """Score one finished graph run (`final` set: auto-decided, or resumed by the adjuster)."""
    final = final_state["final"]
    proposal = final_state.get("decision") or {}
    return ClaimScore(
        claim_id=final_state["claim_id"],
        arm=arm,
        auto=final.get("decided_by") == "auto",
        final_outcome=final["outcome"],
        final_payout=final.get("payout"),
        proposed_outcome=proposal.get("outcome"),
        truth=truth,
        route_reasons=tuple(final_state.get("route", {}).get("reasons", [])),
        failsafe=bool(final_state.get("failsafe")),
        llm_calls=int(final_state.get("llm_calls", 0)),
        tokens=int(final_state.get("tokens", 0)),
    )


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def metric(scores: Sequence[ClaimScore], name: str) -> float | None:
    n = len(scores)
    autos = [s for s in scores if s.auto]
    traps = [s for s in scores if s.truth.traps]
    frauds = [s for s in scores if s.truth.is_fraud]
    values: dict[str, Callable[[], float | None]] = {
        "auto_rate": lambda: _rate(len(autos), n),
        "auto_accuracy": lambda: _rate(sum(s.correct for s in autos), len(autos)),
        "wrong_auto_approve_rate": lambda: _rate(sum(s.wrong_auto_approve for s in scores), n),
        "trap_leak_rate": lambda: _rate(sum(s.trap_leaked for s in traps), len(traps)),
        "fraud_auto_paid_rate": lambda: _rate(sum(s.fraud_auto_paid for s in frauds), len(frauds)),
        "proposal_accuracy": lambda: _rate(
            sum(s.proposed_outcome == s.truth.outcome for s in scores), n
        ),
        "calls_per_claim": lambda: sum(s.llm_calls for s in scores) / n if n else None,
        "tokens_per_claim": lambda: sum(s.tokens for s in scores) / n if n else None,
        "overpaid_per_claim": lambda: sum(s.overpaid for s in scores) / n if n else None,
        "failsafe_rate": lambda: _rate(sum(s.failsafe for s in scores), n),
    }
    if name not in values:
        raise ValueError(f"unknown metric {name!r}")
    return values[name]()


ALL_METRICS = (*HEADLINE, "tokens_per_claim", "overpaid_per_claim", "failsafe_rate")


def summarize(scores: Sequence[ClaimScore], n_boot: int = 2000, seed: int = 0
              ) -> dict[str, judgekit.Estimate]:  # fmt: skip
    """Every metric with a bootstrap CI over claims."""
    return {m: judgekit.bootstrap(list(scores), _metric_fn(m), n_boot, 0.05, seed)
            for m in ALL_METRICS}  # fmt: skip


def _metric_fn(name: str) -> Callable[[Sequence[ClaimScore]], float | None]:
    def fn(units: Sequence[ClaimScore]) -> float | None:
        return metric(units, name)

    return fn


def paired(
    a: Sequence[ClaimScore], b: Sequence[ClaimScore], name: str, n_boot: int = 2000, seed: int = 0
) -> judgekit.Estimate:
    """metric(b) - metric(a) on the claims both arms scored, resampling claims (paired)."""
    by_a = {s.claim_id: s for s in a}
    pairs = [(by_a[s.claim_id], s) for s in b if s.claim_id in by_a]

    def diff(units: Sequence[tuple[ClaimScore, ClaimScore]]) -> float | None:
        ma, mb = metric([x for x, _ in units], name), metric([y for _, y in units], name)
        return None if ma is None or mb is None else mb - ma

    return judgekit.bootstrap(pairs, diff, n_boot, 0.05, seed)


# ---------------------------------------------------------------- ablations


class AblatedLine:
    """The real line with one verification step switched off (it always passes, no LLM call).
    Everything else delegates to the wrapped line."""

    def __init__(self, line: LineOfBusiness, no_judge: bool = False, no_critic: bool = False):
        self._line = line
        self.no_judge = no_judge
        self.no_critic = no_critic

    def __getattr__(self, name: str) -> Any:
        return getattr(self._line, name)

    def judge(self, state: ClaimState) -> NodeResult:
        if self.no_judge:
            return NodeResult({"judge": {"passed": True, "issues": []}}, outputs={"ablated": True})
        return self._line.judge(state)

    def critic(self, state: ClaimState) -> NodeResult:
        if self.no_critic:
            return NodeResult({"critic": {"passed": True, "issues": []}}, outputs={"ablated": True})
        return self._line.critic(state)


def markdown_table(arms: dict[str, Sequence[ClaimScore]], n_boot: int = 2000, seed: int = 0
                   ) -> str:  # fmt: skip
    """Per arm: every metric with its CI; for each non-baseline arm, the paired difference vs the
    first arm (the baseline) on the headline metrics."""
    names = list(arms)
    summaries = {a: summarize(arms[a], n_boot, seed) for a in names}

    def num(x: float, pct: bool) -> str:
        return f"{x:.1%}" if pct else f"{x:,.2f}"

    def fmt(e: judgekit.Estimate, pct: bool) -> str:
        if e.value is None:
            return "n/a"
        ci = "" if e.low is None or e.high is None else f" [{num(e.low, pct)}, {num(e.high, pct)}]"
        return f"{num(e.value, pct)}{ci}"

    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m in ALL_METRICS:
        pct = not m.endswith(("per_claim",))
        lines.append(f"| {m} | " + " | ".join(fmt(summaries[a][m], pct) for a in names) + " |")
    base = names[0]
    for other in names[1:]:
        lines += ["", f"Paired difference {other} - {base} (same claims):", "",
                  "| metric | difference [95% CI] | real gap? |", "|---|---|---|"]  # fmt: skip
        for m in HEADLINE:
            d = paired(arms[base], arms[other], m, n_boot, seed)
            pct = not m.endswith("per_claim")
            lines.append(f"| {m} | {fmt(d, pct)} | {'yes' if d.excludes(0.0) else 'no'} |")
    return "\n".join(lines)
