"""Policy documents as typed clauses with typed cross-references (line-of-business agnostic)."""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

EdgeType = Literal[
    "uses_term", "subject_to", "has_exception", "overridden_by", "requires", "see_also"
]


class Edge(BaseModel):
    type: EdgeType
    target: str


class Clause(BaseModel):
    id: str
    section: str
    title: str
    text: str
    edges: list[Edge] = Field(default_factory=list)

    @property
    def document(self) -> str:
        """What gets indexed: the title carries key terms, so it goes in front."""
        return f"{self.title}. {self.text}"


class PolicyMeta(BaseModel):
    id: str
    name: str
    version: str
    jurisdiction: str


class PolicyDoc(BaseModel):
    policy: PolicyMeta
    clauses: list[Clause] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "PolicyDoc":
        ids = [c.id for c in self.clauses]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate clause ids: {dupes}")
        known = set(ids)
        dangling = [
            (c.id, e.target) for c in self.clauses for e in c.edges if e.target not in known
        ]
        if dangling:
            raise ValueError(f"edges to unknown clauses: {dangling[:5]}")
        return self

    @property
    def by_id(self) -> dict[str, Clause]:
        return {c.id: c for c in self.clauses}

    def fingerprint(self) -> str:
        """Content hash: an index built from an older policy text is never reused."""
        payload = "\n".join(f"{c.id}\t{c.document}" for c in self.clauses)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_policy(path: Path) -> PolicyDoc:
    return PolicyDoc.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
