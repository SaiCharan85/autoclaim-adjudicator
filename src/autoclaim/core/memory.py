"""Feedback memory: what adjusters decided on past escalated cases, retrieved as context.

Episodic: each human review stores one episode (a description of the case, the harness's
proposed decision, the adjuster's decision and reason, and whether that was a correction).
Similar past episodes can be shown to the adjudicator as few-shot context. Similarity is used to
RETRIEVE CONTEXT, never to reuse an answer: no decision is ever copied from memory.

Leakage rules (enforced here):
- `cutoff`: cases on or after this date are never remembered, so the memory can only ever hold
  training/validation-period cases (evaluation runs use the test period).
- `read_only`: an evaluation opens a frozen memory and cannot write to it.

Line-agnostic: the line of business supplies `describe(state) -> MemoryText` (what to embed and
show, plus the case date). Backends are swappable: `ExactIndex` (numpy, small memories and
tests) or `HNSWMemoryIndex` (FAISS HNSW, persisted vectors are re-indexed on load, so the
embedding model never runs twice for one episode).
"""

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict

from autoclaim.config import HNSWConfig
from autoclaim.core.state import ClaimState
from autoclaim.retrieval.dense import Embedder, l2_normalize

_FAISS: Any = faiss


@dataclass(frozen=True)
class MemoryText:
    text: str  # what is embedded and shown as context
    when: date | None  # the case date, checked against the cutoff


class Episode(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    text: str
    case_date: date | None
    proposed_outcome: str | None
    proposed_reasons: list[str]
    human_outcome: str
    human_payout: float | None
    human_reason: str
    adjuster: str
    corrected: bool  # the adjuster's outcome differs from the proposal (or there was none)
    recorded_at: datetime


class VectorIndex(Protocol):
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


class ExactIndex:
    """Brute-force cosine search; exact, fine up to tens of thousands of episodes."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        self.ids += list(ids)
        self._vectors = np.vstack([self._vectors, l2_normalize(vectors).reshape(-1, self.dim)])

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids or k <= 0:
            return []
        scores = self._vectors @ l2_normalize(vector.reshape(-1))
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self.ids)


class HNSWMemoryIndex:
    """FAISS HNSW (inner product on normalized vectors = cosine); supports incremental adds."""

    def __init__(self, dim: int, cfg: HNSWConfig) -> None:
        self.cfg = cfg
        self.ids: list[str] = []
        self.index = faiss.IndexHNSWFlat(dim, cfg.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = cfg.ef_construction
        self.index.hnsw.efSearch = cfg.ef_search

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        self.index.add(l2_normalize(vectors).reshape(len(ids), -1))
        self.ids += list(ids)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids or k <= 0:
            return []
        k = min(k, len(self.ids))
        params = _FAISS.SearchParametersHNSW(efSearch=max(k, self.cfg.ef_search))
        scores, idx = self.index.search(l2_normalize(vector.reshape(1, -1)), k, params=params)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=True) if i >= 0]

    def __len__(self) -> int:
        return len(self.ids)


Describe = Callable[[ClaimState], MemoryText]
IndexFactory = Callable[[int], VectorIndex]


class FeedbackMemory:
    """A `MemoryStore` (the graph calls `remember` after every human review)."""

    def __init__(
        self,
        embedder: Embedder,
        describe: Describe,
        path: Path | None = None,
        index_factory: IndexFactory | None = None,
        cutoff: date | None = None,
        read_only: bool = False,
    ) -> None:
        self.embedder = embedder
        self.describe = describe
        self.cutoff = cutoff
        self.read_only = read_only
        self.skipped_after_cutoff = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(":memory:" if path is None else str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS episodes"
            " (claim_id TEXT PRIMARY KEY, episode TEXT NOT NULL, vector BLOB NOT NULL)"
        )
        self._make_index = index_factory or ExactIndex
        self._episodes: dict[str, Episode] = {}
        self._reindex()

    def _reindex(self) -> None:
        """Rebuild the vector index from stored vectors (no re-embedding)."""
        self.index = self._make_index(self.embedder.dim)
        rows = self._db.execute("SELECT claim_id, episode, vector FROM episodes").fetchall()
        self._episodes = {cid: Episode.model_validate_json(ep) for cid, ep, _ in rows}
        if rows:
            vectors = np.stack([np.frombuffer(v, dtype=np.float32) for _, _, v in rows])
            self.index.add([cid for cid, _, _ in rows], vectors)

    def __len__(self) -> int:
        return len(self._episodes)

    def remember(self, state: ClaimState) -> None:
        human = state.get("human")
        if self.read_only or not human:
            return
        desc = self.describe(state)
        if self.cutoff is not None and (desc.when is None or desc.when >= self.cutoff):
            self.skipped_after_cutoff += 1
            return  # never learn from the evaluation period (or from an undated case)
        proposed = state.get("decision") or {}
        episode = Episode(
            claim_id=state["claim_id"],
            text=desc.text,
            case_date=desc.when,
            proposed_outcome=proposed.get("outcome"),
            proposed_reasons=list(proposed.get("reasons", [])),
            human_outcome=human["outcome"],
            human_payout=human.get("payout"),
            human_reason=human["reason"],
            adjuster=human["adjuster"],
            corrected=proposed.get("outcome") != human["outcome"],
            recorded_at=datetime.now(UTC),
        )
        vector = self.embedder.embed_documents([desc.text])[0].astype(np.float32)
        replacing = episode.claim_id in self._episodes
        self._db.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?)",
            (episode.claim_id, episode.model_dump_json(), vector.tobytes()),
        )
        self._db.commit()
        if replacing:  # a re-reviewed case: keep one episode per claim
            self._reindex()
        else:
            self._episodes[episode.claim_id] = episode
            self.index.add([episode.claim_id], vector.reshape(1, -1))

    def similar(self, text: str, k: int, exclude: str | None = None) -> list[Episode]:
        """The `k` most similar past episodes (never the case itself); corrections first among
        near-equals, since they carry what the harness got wrong."""
        if k <= 0 or not self._episodes:
            return []
        # case-to-case similarity is symmetric: embed like a stored episode (no query prefix)
        vector = self.embedder.embed_documents([text])[0]
        hits = self.index.search(vector, 2 * k + 1)
        found = [self._episodes[cid] for cid, _ in hits if cid != exclude]
        ranked = sorted(found[: 2 * k], key=lambda e: not e.corrected)  # stable: keeps similarity
        return ranked[:k]

    def episodes(self) -> list[Episode]:
        return sorted(self._episodes.values(), key=lambda e: e.claim_id)


def few_shot_block(episodes: Sequence[Episode]) -> str:
    """Compact context lines for the adjudicator prompt."""
    lines = []
    for e in episodes:
        proposed = (
            f"{e.proposed_outcome} {json.dumps(e.proposed_reasons)}"
            if e.proposed_outcome
            else "none"
        )
        payout = f" payout {e.human_payout:.2f}" if e.human_payout is not None else ""
        tag = "CORRECTED" if e.corrected else "confirmed"
        lines.append(f"- {e.text} | harness proposed: {proposed} | adjuster ({tag}): "
                     f"{e.human_outcome}{payout}, because {e.human_reason}")  # fmt: skip
    return "\n".join(lines)


def make_few_shot(memory: FeedbackMemory, k: int) -> Callable[[ClaimState], str]:
    """A few-shot provider for the adjudicator: similar past adjuster decisions."""

    def few_shot(state: ClaimState) -> str:
        if k <= 0 or not len(memory):
            return ""
        desc = memory.describe(state)
        return few_shot_block(memory.similar(desc.text, k, exclude=state.get("claim_id")))

    return few_shot
