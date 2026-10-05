import numpy as np
import pytest
import yaml
from retrieval_fakes import policy

from autoclaim.config import HNSWConfig
from autoclaim.retrieval.benchmark import (
    LabeledQuery,
    evaluate,
    hnsw_recall,
    load_queries,
    score_query,
)
from autoclaim.retrieval.bm25 import BM25Index
from autoclaim.retrieval.dense import HNSWIndex
from autoclaim.retrieval.graph import PolicyGraph
from autoclaim.retrieval.retriever import Hit, HybridRetriever

Q = LabeledQuery(id="q", query="deer", relevant=["INS-COMP", "DEF-COLL"], direct=["INS-COMP"])


def _hit(cid: str, hop: int = 0) -> Hit:
    return Hit(policy().by_id[cid], 0.0, hop, "x")


def test_score_query_separates_search_and_context() -> None:
    s = score_query(Q, [_hit("EXC-WEAR"), _hit("INS-COMP"), _hit("DEF-COLL", hop=1)], 2.0)
    assert s.direct_recall == 1 and s.context_recall == 1 and s.mrr == 0.5
    assert s.context_size == 3
    miss = score_query(Q, [_hit("EXC-WEAR")], 1.0)
    assert miss.direct_recall == 0 and miss.mrr == 0


def test_direct_must_be_subset() -> None:
    with pytest.raises(ValueError, match="subset"):
        LabeledQuery(id="q", query="x", relevant=["A"], direct=["B"])


def test_load_queries_checks_ids(tmp_path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump({"queries": [Q.model_dump()]}), encoding="utf-8")
    assert load_queries(path, set(policy().by_id))[0].id == "q"
    with pytest.raises(ValueError, match="unknown"):
        load_queries(path, {"INS-COMP"})


def test_evaluate_graph_raises_context_recall() -> None:
    doc = policy()
    ids, docs = [c.id for c in doc.clauses], [c.document for c in doc.clauses]
    r = HybridRetriever(doc, BM25Index(ids, docs), None, PolicyGraph(doc.clauses))
    flat = evaluate("bm25", lambda q: r.search(q, k=1), [Q])
    graph = evaluate("bm25+g", lambda q: r.search(q, k=1, hops=1), [Q])
    assert flat["context_recall"] == 0.5 and graph["context_recall"] == 1.0
    assert graph["avg_context"] > flat["avg_context"]


def test_hnsw_recall_is_one_on_small_corpus() -> None:
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(40, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cfg = HNSWConfig(M=8, ef_construction=40, ef_search=32)
    index = HNSWIndex.build([str(i) for i in range(40)], vecs, cfg)
    assert hnsw_recall(index.index, vecs, vecs[:5], 5) == 1.0
