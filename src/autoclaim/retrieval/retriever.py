"""Hybrid policy retrieval: BM25 + dense HNSW, fused with Reciprocal Rank Fusion, then a
k-hop expansion over the policy graph.

RRF (Cormack et al., 2009) adds 1 / (k + rank) across rankers: it needs no score calibration
between BM25 and cosine, and a clause ranked well by either ranker rises.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from autoclaim.retrieval.bm25 import BM25Index
from autoclaim.retrieval.corpus import Clause, PolicyDoc
from autoclaim.retrieval.dense import Embedder, HNSWIndex
from autoclaim.retrieval.graph import PolicyGraph


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Hit:
    clause: Clause
    score: float  # RRF score for search hits; 0 for graph-only additions
    hop: int  # 0 = found by search; 1..k = added by the policy graph
    via: str
    sources: tuple[str, ...] = field(default=())  # which rankers found it


class HybridRetriever:
    def __init__(
        self,
        policy: PolicyDoc,
        bm25: BM25Index | None,
        dense: tuple[Embedder, HNSWIndex] | None,
        graph: PolicyGraph | None,
        rrf_k: int = 60,
        candidates: int = 20,
    ) -> None:
        if bm25 is None and dense is None:
            raise ValueError("need at least one ranker")
        self.policy = policy
        self.by_id = policy.by_id
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
        self.rrf_k = rrf_k
        self.candidates = candidates

    def rankings(self, query: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if self.bm25 is not None:
            out["bm25"] = [i for i, _ in self.bm25.search(query, self.candidates)]
        if self.dense is not None:
            embedder, index = self.dense
            found = index.search(embedder.embed_query(query), self.candidates)
            out["dense"] = [i for i, _ in found]
        return out

    def search(
        self,
        query: str,
        k: int = 5,
        hops: int = 0,
        edge_types: frozenset[str] | None = None,
        max_expanded: int | None = 6,
    ) -> list[Hit]:
        """Top-k fused search hits, then (hops > 0) graph neighbors appended after them."""
        ranked = self.rankings(query)
        fused = rrf(list(ranked.values()), self.rrf_k)[:k]
        hits = [
            Hit(
                self.by_id[cid],
                score,
                0,
                "search",
                tuple(name for name, r in ranked.items() if cid in r),
            )
            for cid, score in fused
        ]
        if hops > 0 and self.graph is not None:
            seeds = [h.clause.id for h in hits]
            for ex in self.graph.expand(seeds, hops, edge_types, max_expanded):
                if ex.hop > 0:
                    hits.append(Hit(self.by_id[ex.id], 0.0, ex.hop, ex.via))
        return hits
