"""Deterministic fraud tools for the auto line.

These outputs are the source of truth for every fraud number; the fraud node's LLM only writes
the assessment around them (Step 5). Input rows are raw claim records in the dataset's own schema
(the artifact remembers which dataset spec it was trained on); missing fields are tolerated and
reported, never silently treated as "no red flag".
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from autoclaim.config import load_carrier_config
from autoclaim.core.rules import RuleHit, RuleSet
from autoclaim.lines.auto.datasets import get_spec
from autoclaim.ml import fraud_model
from autoclaim.ml.explain import ShapReason, top_reasons
from autoclaim.ml.spec import DatasetSpec
from autoclaim.paths import models_dir

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
    # set only when the claim has an independent appraisal (two-stage triage)
    stage2_score: float | None = Field(default=None, ge=0, le=1)
    two_stage_referral: bool | None = None
    stage2_model_version: str | None = None


def two_stage_refer(s1: float, s2: float, stage1_cut: float, stage2_threshold: float) -> bool:
    """The frozen two-stage policy for one claim: refer if the first-notice score is in the top
    share (at or above the stage-1 cut) or its post-appraisal score reaches the threshold."""
    return s1 >= stage1_cut or s2 >= stage2_threshold


@dataclass(frozen=True)
class StageTwo:
    """The post-appraisal model and the frozen policy numbers (fraud_model.triage in config)."""

    artifacts: fraud_model.FraudArtifacts
    spec: DatasetSpec
    stage1_cut: float
    threshold: float


class FraudToolkit:
    def __init__(
        self,
        artifacts: fraud_model.FraudArtifacts,
        rules: RuleSet,
        spec: DatasetSpec,
        k_reasons: int = 5,
        stage2: StageTwo | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.rules = rules
        self.spec = spec
        self.k_reasons = k_reasons
        self.stage2 = stage2

    @classmethod
    def load(cls, model_dir: Path | None = None, rules_path: Path = RULES_PATH,
             stage2_dir: Path | None = None) -> "FraudToolkit":  # fmt: skip
        """`stage2_dir`: the post-appraisal model (default models/fraud_appraisal); stage 2 is
        off when that model or the frozen policy numbers are missing."""
        cfg = load_carrier_config().fraud_model
        artifacts = fraud_model.load(model_dir)
        spec = get_spec(artifacts.dataset, cfg)
        stage2 = None
        tri = cfg.triage
        directory = stage2_dir or models_dir() / "fraud_appraisal"
        if (tri is not None and tri.stage2_threshold is not None
                and tri.stage1_cut_score is not None and directory.exists()):  # fmt: skip
            art2 = fraud_model.load(directory)
            stage2 = StageTwo(art2, get_spec(art2.dataset, cfg), tri.stage1_cut_score,
                              tri.stage2_threshold)  # fmt: skip
        return cls(artifacts, RuleSet.from_yaml(rules_path), spec, stage2=stage2)

    def stage2_scores(self, raw: pd.DataFrame) -> np.ndarray:
        """Post-appraisal fraud probability per row (NaN where a row has no appraisal)."""
        out = np.full(len(raw), np.nan)
        if self.stage2 is None or "appraised_amount" not in raw:
            return out
        has = pd.to_numeric(raw["appraised_amount"], errors="coerce").notna().to_numpy()
        if not has.any():
            return out
        rows = raw.loc[has].reset_index(drop=True)
        features = self.stage2.spec.features_frame(rows)
        for f in self.stage2.artifacts.features:
            if f not in features:
                features[f] = float("nan")
        out[has] = self.stage2.artifacts.predict_proba(features).to_numpy()
        return out

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
        s2 = self.stage2_scores(raw)
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
                    **self._stage2_fields(float(scores[i]), float(s2[i])),
                )
            )
        return out

    def _stage2_fields(self, s1: float, s2: float) -> dict[str, Any]:
        if self.stage2 is None or np.isnan(s2):
            return {}
        refer = two_stage_refer(s1, s2, self.stage2.stage1_cut, self.stage2.threshold)
        return {"stage2_score": round(s2, 6), "two_stage_referral": refer,
                "stage2_model_version": self.stage2.artifacts.version}  # fmt: skip

    def assess(self, record: Mapping[str, Any]) -> FraudSignals:
        return self.assess_frame(pd.DataFrame([dict(record)]))[0]

    def assess_many(self, records: Sequence[Mapping[str, Any]]) -> list[FraudSignals]:
        return self.assess_frame(pd.DataFrame([dict(r) for r in records]))
