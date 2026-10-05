"""Append-only audit trail: one event per node run (what ran, what it produced, at what cost).

Events live in the graph state (so a checkpoint carries them) and are also appended to a JSONL file
per claim. The file is only ever opened in append mode.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    claim_id: str
    node: str
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    status: str = "ok"  # ok | skipped | failsafe | error
    outputs: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # free tiers only, so this stays 0; kept for the audit contract
    latency_s: float = 0.0
    cached: bool = False


class AuditSink:
    def __init__(self, directory: Path | None) -> None:
        self.directory = directory  # None = keep events in the state only (tests)

    def write(self, event: AuditEvent) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"{event.claim_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def read(self, claim_id: str) -> list[AuditEvent]:
        if self.directory is None:
            return []
        path = self.directory / f"{claim_id}.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [AuditEvent.model_validate(json.loads(line)) for line in lines if line]
