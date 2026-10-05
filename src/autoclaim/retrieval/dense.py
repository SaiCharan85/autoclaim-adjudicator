"""Dense retrieval: local embeddings (fastembed, ONNX on CPU) + FAISS HNSW.

Vectors are L2-normalized, so inner product = cosine similarity (IndexHNSWFlat with
METRIC_INNER_PRODUCT). Indexes are cached on disk keyed by model + corpus fingerprint, so the
embedding model only runs when the policy text or the model changes.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from autoclaim.config import HNSWConfig

_FAISS: Any = faiss  # untyped access to classes missing from faiss's bundled stubs


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    out: np.ndarray = x / np.clip(norms, 1e-12, None)
    return out


class FastEmbedEmbedder:
    """Local, free embeddings (default BAAI/bge-large-en-v1.5). The model downloads once."""

    def __init__(self, model_name: str, cache_dir: Path, query_prefix: str = "") -> None:
        from fastembed import TextEmbedding  # heavy import (onnxruntime): only when used

        self.name = model_name
        self.query_prefix = query_prefix
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self.dim = int(self._model.embedding_size)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalize(np.stack(list(self._model.embed(list(texts), batch_size=32))))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(np.stack(list(self._model.embed([self.query_prefix + text])))[0])


class HNSWIndex:
    def __init__(self, ids: Sequence[str], index: faiss.Index) -> None:
        self.ids = list(ids)
        self.index = index

    @classmethod
    def build(cls, ids: Sequence[str], vectors: np.ndarray, cfg: HNSWConfig) -> "HNSWIndex":
        vectors = l2_normalize(vectors)
        index = faiss.IndexHNSWFlat(vectors.shape[1], cfg.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.ef_construction
        index.hnsw.efSearch = cfg.ef_search
        index.add(vectors)
        return cls(ids, index)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = l2_normalize(query_vector.reshape(1, -1))
        k = min(k, len(self.ids))
        # HNSW returns at most efSearch results: widen the beam for this query when k is larger
        hnsw: Any = faiss.downcast_index(self.index)  # faiss's stubs miss HNSW attributes
        params = _FAISS.SearchParametersHNSW(efSearch=max(k, int(hnsw.hnsw.efSearch)))
        scores, idx = self.index.search(q, k, params=params)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=True) if i >= 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        path.with_suffix(".ids").write_text("\n".join(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, ef_search: int) -> "HNSWIndex":
        index = faiss.read_index(str(path))
        hnsw: Any = faiss.downcast_index(index)  # the stubs type it as the base Index
        hnsw.hnsw.efSearch = ef_search
        ids = path.with_suffix(".ids").read_text(encoding="utf-8").split("\n")
        return cls(ids, index)


def cached_index(
    ids: Sequence[str], documents: Sequence[str], embedder: Embedder, cfg: HNSWConfig, path: Path
) -> HNSWIndex:
    if path.exists() and path.with_suffix(".ids").exists():
        loaded = HNSWIndex.load(path, cfg.ef_search)
        if loaded.ids == list(ids):
            return loaded
    built = HNSWIndex.build(ids, embedder.embed_documents(documents), cfg)
    built.save(path)
    return built
