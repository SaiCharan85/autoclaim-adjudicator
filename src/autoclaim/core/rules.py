"""Line-agnostic, data-driven red-flag rules engine.

A rule fires when *all* its conditions hold. Rules are deterministic, so they are an auditable fact
source alongside the model. A condition on a missing field or a missing value never fires; missing
fields are reported so the caller can tell "no red flag" from "couldn't check".

Combined score: noisy-OR of fired rule weights, 1 - prod(1 - w), so it stays in [0, 1] and each
additional red flag raises it with diminishing returns.
"""

import operator
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

Op = Literal["eq", "ne", "in", "not_in", "lt", "le", "gt", "ge"]

_COMPARE: dict[str, Callable[[Any, Any], Any]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


class Condition(BaseModel):
    field: str
    op: Op
    value: Any

    @model_validator(mode="after")
    def _list_for_membership(self) -> "Condition":
        if self.op in ("in", "not_in") and not isinstance(self.value, list):
            raise ValueError(f"op {self.op!r} needs a list value")
        return self

    def mask(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.field]
        if self.op == "in":
            hit = col.isin(self.value)
        elif self.op == "not_in":
            hit = ~col.isin(self.value)
        else:
            hit = _COMPARE[self.op](col, self.value)
        return pd.Series(hit, index=df.index).fillna(False).astype(bool) & col.notna()


class Rule(BaseModel):
    id: str
    description: str
    weight: float = Field(gt=0, le=1)
    all_of: list[Condition] = Field(min_length=1)

    @property
    def fields(self) -> set[str]:
        return {c.field for c in self.all_of}


class RuleHit(BaseModel):
    rule_id: str
    description: str
    weight: float


class RuleSet(BaseModel):
    name: str
    version: str
    rules: list[Rule]

    @model_validator(mode="after")
    def _unique_ids(self) -> "RuleSet":
        ids = [r.id for r in self.rules]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate rule ids: {sorted(dupes)}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "RuleSet":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Boolean frame (rows x rule ids). Rules with a missing column are all False."""
        out = {}
        for rule in self.rules:
            if rule.fields - set(df.columns):
                out[rule.id] = pd.Series(False, index=df.index)
                continue
            mask = pd.Series(True, index=df.index)
            for cond in rule.all_of:
                mask &= cond.mask(df)
            out[rule.id] = mask
        return pd.DataFrame(out, index=df.index)

    def unchecked(self, df: pd.DataFrame) -> pd.DataFrame:
        """Boolean frame (rows x rule ids): True where a rule *could still have fired* but a fact
        it needs is missing. A rule already ruled out by a known-false condition (e.g. "cause is
        theft" on a collision claim) is not unchecked: missing facts couldn't change its answer."""
        out = {}
        for rule in self.rules:
            missing = pd.Series(False, index=df.index)
            known_false = pd.Series(False, index=df.index)
            for cond in rule.all_of:
                if cond.field not in df.columns:
                    missing |= True
                    continue
                absent = df[cond.field].isna()
                missing |= absent
                known_false |= ~absent & ~cond.mask(df)
            out[rule.id] = missing & ~known_false
        return pd.DataFrame(out, index=df.index)

    def missing_fields(self, columns: set[str]) -> dict[str, list[str]]:
        """Rule id -> fields it needs that are absent (those rules could not be checked)."""
        return {r.id: sorted(r.fields - columns) for r in self.rules if r.fields - columns}

    def score(self, hits: pd.DataFrame) -> pd.Series:
        weights = np.array([r.weight for r in self.rules])
        survival = np.where(hits[[r.id for r in self.rules]].to_numpy(), 1 - weights, 1.0)
        return pd.Series(1 - survival.prod(axis=1), index=hits.index)

    def evaluate_record(self, record: Mapping[str, Any]) -> tuple[list[RuleHit], float]:
        """Single-claim convenience: fired rules and combined score."""
        hits = self.evaluate(pd.DataFrame([dict(record)]))
        fired = [
            RuleHit(rule_id=r.id, description=r.description, weight=r.weight)
            for r in self.rules
            if bool(hits.at[0, r.id])
        ]
        return fired, float(self.score(hits).to_numpy()[0])

    def report(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Per rule on labeled data: support, precision, and lift over the base rate."""
        hits = self.evaluate(df)
        base = df[target].mean()
        rows = []
        for r in self.rules:
            fired = hits[r.id]
            n = int(fired.sum())
            precision = float(df.loc[fired, target].mean()) if n else float("nan")
            rows.append(
                {
                    "rule": r.id,
                    "fires_on": n,
                    "support": round(n / len(df), 4),
                    "fraud_rate_when_fired": round(precision, 4) if n else None,
                    "lift": round(precision / base, 2) if n and base else None,
                }
            )
        return pd.DataFrame(rows).set_index("rule")
