"""Retrieval benchmark over labeled queries: BM25 vs dense (bge small/base/large) vs hybrid RRF,
with and without policy-graph expansion, plus HNSW-vs-exact search recall.

Labels per query: `direct` = clauses a good search should find from the words alone;
`relevant` = everything a correct coverage decision must see (some only reachable via the graph).
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from autoclaim.retrieval.retriever import Hit


class LabeledQuery(BaseModel):
    id: str
    query: str
    relevant: list[str] = Field(min_length=1)
    direct: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _direct_subset(self) -> "LabeledQuery":
        if not set(self.direct) <= set(self.relevant):
            raise ValueError(f"{self.id}: direct must be a subset of relevant")
        return self


def load_queries(path: Path, known_ids: set[str]) -> list[LabeledQuery]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = [LabeledQuery.model_validate(q) for q in raw["queries"]]
    unknown = {i for q in queries for i in q.relevant if i not in known_ids}
    if unknown:
        raise ValueError(f"queries reference unknown clauses: {sorted(unknown)[:5]}")
    return queries


@dataclass(frozen=True)
class QueryScore:
    direct_recall: float  # share of `direct` clauses among the search hits (hop 0)
    context_recall: float  # share of `relevant` clauses anywhere in the returned context
    mrr: float  # 1 / rank of the first relevant search hit (0 if none)
    context_size: int
    latency_ms: float


def score_query(q: LabeledQuery, hits: Sequence[Hit], latency_ms: float) -> QueryScore:
    search_ids = [h.clause.id for h in hits if h.hop == 0]
    context = {h.clause.id for h in hits}
    relevant = set(q.relevant)
    first = next((r for r, cid in enumerate(search_ids, start=1) if cid in relevant), None)
    return QueryScore(
        direct_recall=len(set(q.direct) & set(search_ids)) / len(q.direct),
        context_recall=len(relevant & context) / len(relevant),
        mrr=1.0 / first if first else 0.0,
        context_size=len(hits),
        latency_ms=latency_ms,
    )


SearchFn = Callable[[str], Sequence[Hit]]


def evaluate(
    name: str, search: SearchFn, queries: Sequence[LabeledQuery]
) -> dict[str, float | str]:
    scores = []
    for q in queries:
        start = time.perf_counter()
        hits = search(q.query)
        scores.append(score_query(q, hits, (time.perf_counter() - start) * 1000))
    df = pd.DataFrame([s.__dict__ for s in scores])
    return {
        "config": name,
        "direct_recall@k": round(float(df["direct_recall"].mean()), 3),
        "context_recall": round(float(df["context_recall"].mean()), 3),
        "mrr": round(float(df["mrr"].mean()), 3),
        "avg_context": round(float(df["context_size"].mean()), 1),
        "p50_ms": round(float(df["latency_ms"].median()), 1),
    }


def hnsw_recall(index: faiss.Index, vectors: np.ndarray, queries: np.ndarray, k: int) -> float:
    """Share of exact (brute-force) top-k neighbors that HNSW also returns."""
    exact = faiss.IndexFlatIP(vectors.shape[1])
    exact.add(vectors)
    k = min(k, len(vectors))
    _, truth = exact.search(queries, k)
    _, approx = index.search(queries, k)
    hits = [len(set(t) & set(a)) / k for t, a in zip(truth, approx, strict=True)]
    return float(np.mean(hits))
