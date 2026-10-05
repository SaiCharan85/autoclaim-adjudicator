from retrieval_fakes import policy

from autoclaim.retrieval.graph import PolicyGraph


def test_expand_hops_both_directions() -> None:
    g = PolicyGraph(policy().clauses)
    one = {e.id: e for e in g.expand(["EXC-WEAR"], hops=1)}
    assert one["EXC-WEAR"].hop == 0
    assert one["GEN-EXCEPT"].hop == 1 and "has_exception" in one["GEN-EXCEPT"].via
    assert one["INS-COLL"].hop == 1  # reached against the edge direction
    two = {e.id for e in g.expand(["EXC-WEAR"], hops=2)}
    assert "DEF-COLL" in two  # EXC-WEAR <- INS-COLL -> DEF-COLL


def test_expand_filters_and_caps() -> None:
    g = PolicyGraph(policy().clauses)
    only_terms = {e.id for e in g.expand(["INS-COLL"], 1, frozenset({"uses_term"}))}
    assert only_terms == {"INS-COLL", "DEF-COLL"}
    assert len(g.expand(["INS-COLL"], 2, max_new=1)) == 2
    assert g.expand(["NOPE"], 2) == []
    assert [e.id for e in g.expand(["INS-COLL", "INS-COLL"], 0)] == ["INS-COLL"]


def test_decisive_edges_expand_first() -> None:
    g = PolicyGraph(policy().clauses)
    order = [nb for nb, _ in g.neighbors("INS-COLL", None)]
    assert order == ["DEF-COLL", "EXC-WEAR"]  # uses_term before subject_to
    from autoclaim.retrieval.corpus import EdgeType
    from autoclaim.retrieval.graph import EDGE_PRIORITY

    assert set(EDGE_PRIORITY) == set(EdgeType.__args__)
