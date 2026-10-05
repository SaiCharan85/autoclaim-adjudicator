"""Idempotent finalization: a claim is decided (and paid) at most once.

The first finalize() for a claim id wins; later calls (a retried graph run, a resumed checkpoint, a
double click in the console) get the stored decision back instead of a second payment.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class FinalizationLedger:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS finalized (claim_id TEXT PRIMARY KEY, ts TEXT NOT NULL,"
            " payload TEXT NOT NULL)"
        )

    def finalize(self, claim_id: str, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """(stored payload, created now?). Never overwrites an existing finalization."""
        with self._db:
            cur = self._db.execute(
                "INSERT INTO finalized VALUES (?, ?, ?) ON CONFLICT (claim_id) DO NOTHING",
                (claim_id, datetime.now(UTC).isoformat(), json.dumps(payload, sort_keys=True)),
            )
        created = cur.rowcount == 1
        stored = self.get(claim_id)
        assert stored is not None
        return stored, created

    def get(self, claim_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT payload FROM finalized WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        return None if row is None else dict(json.loads(row[0]))
