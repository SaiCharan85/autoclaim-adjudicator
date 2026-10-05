"""Lexical retrieval: BM25 (Okapi) over clause documents."""

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    ("a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "if", "in",
     "is", "it", "its", "of", "on", "or", "our", "that", "the", "this", "to", "was", "we", "were",
     "will", "with", "you", "your", "my", "me", "i")
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, ids: Sequence[str], documents: Sequence[str]) -> None:
        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        self.ids = list(ids)
        self._bm25 = BM25Okapi([tokenize(d) for d in documents])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (id, score) with score > 0, best first."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=float)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
