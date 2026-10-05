"""A tiny policy and a deterministic bag-of-words embedder (no model download in tests)."""

import hashlib

import numpy as np

from autoclaim.retrieval.bm25 import tokenize
from autoclaim.retrieval.corpus import PolicyDoc

POLICY = {
    "policy": {"id": "p", "name": "Test policy", "version": "1", "jurisdiction": "US"},
    "clauses": [
        {"id": "INS-COLL", "section": "insuring_agreement", "title": "Collision coverage",
         "text": "We pay for direct loss to your covered auto caused by collision.",
         "edges": [{"type": "uses_term", "target": "DEF-COLL"},
                   {"type": "subject_to", "target": "EXC-WEAR"}]},
        {"id": "INS-COMP", "section": "insuring_agreement", "title": "Comprehensive coverage",
         "text": "We pay for loss other than collision, such as theft, hail or hitting a deer.",
         "edges": [{"type": "uses_term", "target": "DEF-COLL"}]},
        {"id": "DEF-COLL", "section": "definitions", "title": "Collision",
         "text": "Upset of the vehicle or impact with another object. Contact with an animal is "
                 "not collision."},
        {"id": "EXC-WEAR", "section": "exclusions", "title": "Wear and tear",
         "text": "We do not pay for mechanical breakdown, rust or worn tires.",
         "edges": [{"type": "has_exception", "target": "GEN-EXCEPT"}]},
        {"id": "GEN-EXCEPT", "section": "general", "title": "Resulting loss",
         "text": "Breakdown caused by a covered theft is paid."},
    ],
}  # fmt: skip


def policy() -> PolicyDoc:
    return PolicyDoc.model_validate(POLICY)


class HashEmbedder:
    name = "hash-bow"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)
