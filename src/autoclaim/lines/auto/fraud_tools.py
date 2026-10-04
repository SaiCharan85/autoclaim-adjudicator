"""Deterministic fraud tools for the auto line.

These outputs are the source of truth for every fraud number; the fraud node's LLM only writes
the assessment around them (Step 5). Input rows are raw claim records in the dataset's own schema
(the artifact remembers which dataset spec it was trained on); missing fields are tolerated and
reported, never silently treated as "no red flag".
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from autoclaim.config import load_carrier_config
from autoclaim.core.rules import RuleHit, RuleSet
from autoclaim.lines.auto.datasets import get_spec
from autoclaim.ml import fraud_model
from autoclaim.ml.explain import ShapReason, top_reasons
from autoclaim.ml.spec import DatasetSpec

RULES_PATH = Path(__file__).with_name("fraud_rules.yaml")


class FraudSignals(BaseModel):
    model_score: float = Field(ge=0, le=1, description="CatBoost fraud probability")
    top_reasons: list[ShapReason]
    anomaly_score: float = Field(
        ge=0, le=1, description="Isolation Forest percentile: how unusual vs training claims"
    )
    rules_fired: list[RuleHit]
    rule_score: float = Field(ge=0, le=1, description="noisy-OR of fired rule weights")
    rules_unchecked: list[str] = Field(description="rules this claim lacked the facts to check")
    model_version: str


class FraudToolkit:
    def __init__(
        self,
        artifacts: fraud_model.FraudArtifacts,
        rules: RuleSet,
        spec: DatasetSpec,
        k_reasons: int = 5,
    ) -> None:
        self.artifacts = artifacts
        self.rules = rules
        self.spec = spec
        self.k_reasons = k_reasons

    @classmethod
    def load(cls, model_dir: Path | None = None, rules_path: Path = RULES_PATH) -> "FraudToolkit":
        artifacts = fraud_model.load(model_dir)
        spec = get_spec(artifacts.dataset, load_carrier_config().fraud_model)
        return cls(artifacts, RuleSet.from_yaml(rules_path), spec)

    def assess_frame(self, raw: pd.DataFrame) -> list[FraudSignals]:
        """Vectorized: one FraudSignals per row (model, SHAP and anomaly run once for the batch)."""
        raw = raw.reset_index(drop=True)
        features = self.spec.features_frame(raw)
        for f in self.artifacts.features:
            if f not in features:
                features[f] = float("nan")
        canonical = self.spec.canonical(raw)
        art = self.artifacts
        scores = art.predict_proba(features).to_numpy()
        anomaly = art.anomaly.score(features)
        reasons = top_reasons(art.model, features, art.features, art.categorical, self.k_reasons)
        hits = self.rules.evaluate(canonical)
        rule_scores = self.rules.score(hits).to_numpy()
        unchecked = self.rules.unchecked(canonical)
        by_id = {r.id: r for r in self.rules.rules}
        out = []
        for i in range(len(raw)):
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
                    rules_unchecked=[
                        rid for rid in unchecked.columns if bool(unchecked.at[i, rid])
                    ],
                    model_version=art.version,
                )
            )
        return out

    def assess(self, record: Mapping[str, Any]) -> FraudSignals:
        return self.assess_frame(pd.DataFrame([dict(record)]))[0]

    def assess_many(self, records: Sequence[Mapping[str, Any]]) -> list[FraudSignals]:
        return self.assess_frame(pd.DataFrame([dict(r) for r in records]))
