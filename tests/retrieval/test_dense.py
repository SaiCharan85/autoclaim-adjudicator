import numpy as np
from retrieval_fakes import HashEmbedder, policy

from autoclaim.config import HNSWConfig
from autoclaim.retrieval.dense import HNSWIndex, cached_index, l2_normalize

CFG = HNSWConfig(M=8, ef_construction=40, ef_search=16)


def test_l2_normalize() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8]) and np.allclose(out[1], 0)


def test_hnsw_finds_exact_neighbor_with_cosine_scores() -> None:
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 16))
    index = HNSWIndex.build([f"d{i}" for i in range(50)], vecs, CFG)
    hits = index.search(vecs[7] * 3.0, 3)  # scale must not matter (normalized)
    assert hits[0][0] == "d7" and abs(hits[0][1] - 1.0) < 1e-5
    assert len(index.search(vecs[0], 500)) == 50  # k larger than corpus


def test_cached_index_roundtrip_and_rebuild_on_new_ids(tmp_path) -> None:
    doc, emb = policy(), HashEmbedder()
    ids, docs = [c.id for c in doc.clauses], [c.document for c in doc.clauses]
    path = tmp_path / "idx.faiss"
    first = cached_index(ids, docs, emb, CFG, path)
    again = cached_index(ids, docs, emb, CFG, path)
    q = emb.embed_query("deer")
    assert first.search(q, 2) == again.search(q, 2)
    assert path.exists() and path.with_suffix(".ids").exists()
    smaller = cached_index(ids[:3], docs[:3], emb, CFG, path)
    assert len(smaller.ids) == 3
