"""Deterministic fraud tools for the auto line.

These outputs are the source of truth for every fraud number; the fraud node's LLM only writes
the assessment around them (Step 5).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from autoclaim.core.rules import RuleHit, RuleSet
from autoclaim.ml import fraud_model
from autoclaim.ml.explain import ShapReason, top_reasons
from autoclaim.ml.features import clean

RULES_PATH = Path(__file__).with_name("fraud_rules.yaml")
_RAW_DATE_COLUMNS = {"Month", "MonthClaimed", "WeekOfMonth", "WeekOfMonthClaimed", "Age"}


class FraudSignals(BaseModel):
    model_score: float = Field(ge=0, le=1, description="CatBoost fraud probability")
    top_reasons: list[ShapReason]
    anomaly_score: float = Field(ge=0, le=1, description="Isolation Forest percentile")
    rules_fired: list[RuleHit]
    rule_score: float = Field(ge=0, le=1, description="noisy-OR of fired rule weights")
    rules_unchecked: list[str] = Field(description="rules skipped because input fields were absent")
    model_version: str


class FraudToolkit:
    def __init__(self, artifacts: fraud_model.FraudArtifacts, rules: RuleSet, k_reasons: int = 5):
        self.artifacts = artifacts
        self.rules = rules
        self.k_reasons = k_reasons

    @classmethod
    def load(cls, model_dir: Path | None = None, rules_path: Path = RULES_PATH) -> "FraudToolkit":
        return cls(fraud_model.load(model_dir), RuleSet.from_yaml(rules_path))

    def _prepare(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
        """Clean raw rows and NaN-fill absent features. Also returns the columns the caller really
        provided (plus derived ones), so absent inputs are reported instead of silently filled."""
        if set(frame.columns) >= _RAW_DATE_COLUMNS:
            frame = clean(frame)
        provided = set(frame.columns)
        missing = [f for f in self.artifacts.features if f not in provided]
        if missing:
            frame = frame.assign(**{f: float("nan") for f in missing})
        return frame, provided

    def assess_frame(self, df: pd.DataFrame) -> list[FraudSignals]:
        """Vectorized: one FraudSignals per row (model, SHAP and anomaly run once for the batch)."""
        frame, provided = self._prepare(df.reset_index(drop=True))
        scores = self.artifacts.predict_proba(frame).to_numpy()
        anomaly = self.artifacts.anomaly.score(frame)
        reasons = top_reasons(self.artifacts.model, frame, self.artifacts.features, self.k_reasons)
        hits = self.rules.evaluate(frame)
        rule_scores = self.rules.score(hits).to_numpy()
        unchecked = sorted(self.rules.missing_fields(provided))
        by_id = {r.id: r for r in self.rules.rules}
        out = []
        for i in range(len(frame)):
            fired = [
                RuleHit(rule_id=rid, description=by_id[rid].description, weight=by_id[rid].weight)
                for rid in hits.columns
                if bool(hits.at[i, rid])
            ]
            out.append(
                FraudSignals(
                    model_score=round(float(scores[i]), 6),
                    top_reasons=reasons[i],
                    anomaly_score=round(float(anomaly[i]), 6),
                    rules_fired=fired,
                    rule_score=round(float(rule_scores[i]), 6),
                    rules_unchecked=unchecked,
                    model_version=self.artifacts.version,
                )
            )
        return out

    def assess(self, record: Mapping[str, Any]) -> FraudSignals:
        return self.assess_frame(pd.DataFrame([dict(record)]))[0]

    def assess_many(self, records: Sequence[Mapping[str, Any]]) -> list[FraudSignals]:
        return self.assess_frame(pd.DataFrame([dict(r) for r in records]))
