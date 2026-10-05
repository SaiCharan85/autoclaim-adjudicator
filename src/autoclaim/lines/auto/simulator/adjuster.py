"""A simulated (noisy) adjuster that answers interrupted claims. EVALUATION ONLY.

It plays the human in human review: it returns the ground-truth decision (what a careful
adjuster would conclude after investigating), except that, like a real person, it makes a
seeded mistake on a share of claims (`error_rate`). Mistakes are recorded in `errors` so an
evaluation can measure how much one wrong correction hurts a learning loop; the decision itself
does not reveal them (a real adjuster doesn't flag their own mistakes).

Plain-English trace: claim escalated for late notice -> truth says "approve, pay $1,650"
-> 95% of the time the adjuster approves $1,650; 5% of the time it picks another outcome.
"""

import random
from collections.abc import Mapping
from typing import Any

import pandas as pd

from autoclaim.core.decision import HumanDecision

OUTCOMES = ("approve", "deny", "escalate")


class OracleAdjuster:
    def __init__(
        self, claims: pd.DataFrame, error_rate: float = 0.05, seed: int = 0, name: str = "oracle"
    ) -> None:
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be in [0, 1]")
        cols = ["claim_id", "gt_decision", "gt_reasons", "gt_payout"]
        self.truth = claims[cols].set_index("claim_id")
        self.error_rate = error_rate
        self.seed = seed
        self.name = name
        self.errors: set[str] = set()

    def decide(self, request: Mapping[str, Any]) -> HumanDecision:
        """Answer one interrupt request (`claim_id`, `proposed_decision`, ...)."""
        claim_id = str(request["claim_id"])
        if claim_id not in self.truth.index:
            raise ValueError(f"no ground truth for claim {claim_id!r}")
        row: dict[str, Any] = {str(k): v for k, v in self.truth.loc[claim_id].items()}
        outcome = str(row["gt_decision"])
        raw = row["gt_reasons"]
        reasons: list[str] = [] if pd.isna(raw) else [r for r in str(raw).split(";") if r]
        if outcome == "approve" and not reasons:
            reasons = ["covered_loss"]
        payout = float(row["gt_payout"]) if outcome == "approve" else None
        rng = random.Random(f"{self.seed}:{claim_id}")
        if rng.random() < self.error_rate:
            outcome = rng.choice([o for o in OUTCOMES if o != outcome])
            proposed = request.get("proposed_decision") or {}
            payout = (proposed.get("payout") or float(row["gt_payout"]) or 0.0
                      if outcome == "approve" else None)  # fmt: skip
            reasons = ["judgment_call"]
            self.errors.add(claim_id)
        return HumanDecision(
            outcome=outcome,  # type: ignore[arg-type]
            payout=None if payout is None else round(payout, 2),
            reason="reasons: " + ", ".join(reasons),
            adjuster=self.name,
        )
