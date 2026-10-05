import pytest
from retrieval_fakes import HashEmbedder, policy

from autoclaim.config import HNSWConfig
from autoclaim.retrieval.bm25 import BM25Index
from autoclaim.retrieval.dense import HNSWIndex
from autoclaim.retrieval.graph import PolicyGraph
from autoclaim.retrieval.retriever import HybridRetriever, rrf

CFG = HNSWConfig(M=8, ef_construction=40, ef_search=16)


def test_rrf_rewards_agreement_and_breaks_ties_by_id() -> None:
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [d for d, _ in fused[:2]] == ["a", "b"]  # tie on score -> alphabetical
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf([]) == []


def _retriever(dense: bool = True, graph: bool = True) -> HybridRetriever:
    doc, emb = policy(), HashEmbedder()
    ids, docs = [c.id for c in doc.clauses], [c.document for c in doc.clauses]
    hnsw = HNSWIndex.build(ids, emb.embed_documents(docs), CFG)
    return HybridRetriever(
        doc,
        BM25Index(ids, docs),
        (emb, hnsw) if dense else None,
        PolicyGraph(doc.clauses) if graph else None,
    )


def test_search_then_graph_adds_definition() -> None:
    hits = _retriever().search("a deer ran into the road", k=1, hops=1)
    assert hits[0].clause.id == "INS-COMP" and hits[0].hop == 0
    assert hits[0].sources and set(hits[0].sources) <= {"bm25", "dense"}
    added = {h.clause.id: h for h in hits[1:]}
    assert added["DEF-COLL"].hop == 1 and added["DEF-COLL"].score == 0


def test_no_graph_and_bm25_only() -> None:
    hits = _retriever(dense=False, graph=False).search("worn tires", k=2, hops=2)
    assert all(h.hop == 0 for h in hits) and hits[0].clause.id == "EXC-WEAR"
    assert hits[0].sources == ("bm25",)


def test_needs_a_ranker() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(policy(), None, None, None)
