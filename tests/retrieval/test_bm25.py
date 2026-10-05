import pytest
from retrieval_fakes import policy

from autoclaim.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    assert tokenize("The deer hit MY car!") == ["deer", "hit", "car"]


def test_search_ranks_matching_clause_first() -> None:
    doc = policy()
    index = BM25Index([c.id for c in doc.clauses], [c.document for c in doc.clauses])
    hits = index.search("worn tires and rust", 3)
    assert hits[0][0] == "EXC-WEAR"
    assert all(score > 0 for _, score in hits)
    assert index.search("the and of", 3) == []  # only stopwords
    assert index.search("zebra", 3) == []  # no overlap


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        BM25Index(["a"], [])
