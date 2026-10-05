"""LLM fraud scoring for auto claims, in two arms of the ML / LLM / hybrid benchmark.

- llm:    the LLM sees only the claim's first-notice facts (the same columns the ML model uses).
- hybrid: the same facts plus the deterministic fraud tools' output (CatBoost score, top SHAP
          reasons, Isolation Forest percentile, red flags that fired).

Several claims go in one request (requests/day is the binding free-tier limit). The prompt is a
static prefix (instructions + glossary + schema) followed by the batch, so provider prefix caching
applies. Claims get batch-local ids (C01...), never real identifiers.
"""

import math
from collections.abc import Sequence
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from autoclaim.lines.auto.fraud_tools import FraudSignals

Arm = Literal["llm", "hybrid"]

GLOSSARY = """Fields (personal auto physical-damage claim, US, known at first notice of loss):
cause: what happened (collision_vehicle, collision_object, animal, theft, vandalism, hail_weather,
glass, fire, ...). damage_extent: minor/functional/disabling. towed, at_fault, police_report,
financed, adas (driver-assist sensors), rideshare_endorsement, attorney_involved: 1 yes, 0 no.
vehicle_acv: the car's actual cash value in USD. claimed_amount: USD asked for. claim_to_acv:
claimed_amount / vehicle_acv. collision_/comprehensive_deductible: USD the insured pays first.
days_policy_to_loss / days_policy_to_claim: days from policy start to the loss / to the report.
notice_days: days from loss to report. police_report_hours: hours from loss to police report.
address_change_days: days since the insured last moved (10000 = no move in 2 years).
prior_claims_3y: insured's claims in the past 3 years. witness_count, n_vehicles, injury_count:
counts. out_of_state: loss state differs from policy state. loss_dow: 0 = Monday."""

TOOLS_GLOSSARY = """Each claim also has tool outputs from the insurer's fraud tools:
ml: fraud probability from a gradient-boosted model trained on past claims (well calibrated).
shap: the features that moved the ml score most, as feature=value(log-odds contribution; + raises
risk). anomaly: percentile of how unusual the claim is vs past claims (not a fraud score).
flags: red-flag rules that fired (a reason to look closer, not proof)."""

_TASK = """You screen auto insurance claims for fraud. For every claim, estimate the probability
that an investigation would confirm fraud. About 6% of claims are fraudulent overall. Rank
carefully: the fraud team reviews only the riskiest 5% of claims. Judge each claim on its facts;
missing fields are unknown, not evidence. Return one entry per claim id, in input order, with a
reason of at most 12 words."""

INSTRUCTIONS: dict[Arm, str] = {
    "llm": f"{_TASK}\n\n{GLOSSARY}",
    "hybrid": f"{_TASK} Use the tool outputs as evidence; you may disagree with the ml score when"
    f" the facts justify it.\n\n{GLOSSARY}\n\n{TOOLS_GLOSSARY}",
}


class ClaimScore(BaseModel):
    id: str
    p: float = Field(ge=0, le=1, description="fraud probability")
    why: str = Field(max_length=200)


class BatchScores(BaseModel):
    scores: list[ClaimScore]


def _fmt(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.3g}"
    return str(value)


def describe_claim(features: pd.Series) -> str:
    """Compact `field=value` facts; unknown (NaN) fields are left out to save tokens."""
    parts = []
    for name, value in features.items():
        text = _fmt(value)
        if text is not None:
            parts.append(f"{name}={text}")
    return "; ".join(parts)


def describe_signals(sig: FraudSignals) -> str:
    shap = ", ".join(
        f"{r.feature}={_fmt(_maybe_float(r.value))}({r.contribution:+.2f})" for r in sig.top_reasons
    )
    flags = ", ".join(h.rule_id for h in sig.rules_fired) or "none"
    return (
        f"ml={sig.model_score:.3f}; anomaly={sig.anomaly_score:.2f}; shap: {shap}; flags: {flags}"
    )


def _maybe_float(value: str) -> object:
    try:
        return float(value)
    except ValueError:
        return value


def batch_ids(n: int) -> list[str]:
    width = max(2, len(str(n)))
    return [f"C{i + 1:0{width}d}" for i in range(n)]


def batch_prompt(facts: Sequence[str], tools: Sequence[str] | None = None) -> str:
    """The variable part of the request: one block per claim."""
    ids = batch_ids(len(facts))
    lines = []
    for i, claim_id in enumerate(ids):
        lines.append(f"[{claim_id}] {facts[i]}")
        if tools is not None:
            lines.append(f"  tools: {tools[i]}")
    return "\n".join(lines)


def align_scores(result: BatchScores, n: int) -> list[float]:
    """Scores in input order; skipped or invented ids give NaN (reported, never guessed)."""
    by_id = {s.id.strip().upper(): s.p for s in result.scores}
    return [by_id.get(claim_id, float("nan")) for claim_id in batch_ids(n)]
