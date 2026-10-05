"""Policy graph: clauses as nodes, typed cross-references as edges (NetworkX).

Search finds the clauses that match the claim's words; the graph adds what those clauses depend
on but rarely share words with: the definition that says a deer strike is not a "collision", the
endorsement that gives back an excluded use, the condition a coverage requires.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx

from autoclaim.retrieval.corpus import Clause

# Lower = expanded first: what can flip a decision before what merely restricts or explains it.
EDGE_PRIORITY = {
    "requires": 0,
    "has_exception": 1,
    "overridden_by": 2,
    "uses_term": 3,
    "subject_to": 4,
    "see_also": 5,
}


@dataclass(frozen=True)
class Expansion:
    id: str
    hop: int
    via: str  # "<from id> -<edge type>-> <to id>" (direction as written in the policy)


class PolicyGraph:
    def __init__(self, clauses: Sequence[Clause]) -> None:
        self.g = nx.MultiDiGraph()
        for c in clauses:
            self.g.add_node(c.id, section=c.section)
        for c in clauses:
            for e in c.edges:
                self.g.add_edge(c.id, e.target, type=e.type)

    def neighbors(self, node: str, edge_types: frozenset[str] | None) -> list[tuple[str, str]]:
        """(neighbor, description) over edges in BOTH directions, most decisive edge type first
        (a capped expansion keeps the conditions and exceptions, not a long "subject to" list)."""
        out = []
        for _, tgt, data in self.g.out_edges(node, data=True):
            if edge_types is None or data["type"] in edge_types:
                out.append((EDGE_PRIORITY[data["type"]], tgt, f"{node} -{data['type']}-> {tgt}"))
        for src, _, data in self.g.in_edges(node, data=True):
            if edge_types is None or data["type"] in edge_types:
                out.append((EDGE_PRIORITY[data["type"]], src, f"{src} -{data['type']}-> {node}"))
        return [(nb, via) for _, nb, via in sorted(out)]

    def expand(
        self,
        seeds: Iterable[str],
        hops: int,
        edge_types: frozenset[str] | None = None,
        max_new: int | None = None,
    ) -> list[Expansion]:
        """Breadth-first k-hop expansion; seeds come back at hop 0, each node once (nearest hop)."""
        seen: dict[str, Expansion] = {}
        frontier = []
        for s in seeds:
            if s in self.g and s not in seen:
                seen[s] = Expansion(s, 0, "search")
                frontier.append(s)
        added = 0
        for hop in range(1, hops + 1):
            nxt = []
            for node in frontier:
                for nb, via in self.neighbors(node, edge_types):
                    if nb in seen:
                        continue
                    if max_new is not None and added >= max_new:
                        return list(seen.values())
                    seen[nb] = Expansion(nb, hop, via)
                    nxt.append(nb)
                    added += 1
            frontier = nxt
        return list(seen.values())
