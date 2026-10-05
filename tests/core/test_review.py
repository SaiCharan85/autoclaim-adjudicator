"""End to end: a claim pauses for human review, the adjuster answers, the graph finalizes it
once, and feedback memory records the correction."""

from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_graph import CFG, FakeLine
from test_memory import BagEmbedder

from autoclaim.core.audit import AuditSink
from autoclaim.core.decision import HumanDecision
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.graph import Harness
from autoclaim.core.memory import FeedbackMemory, MemoryText
from autoclaim.core.review import pending, resume


def describe(state) -> MemoryText:
    return MemoryText(f"case {state['claim_id']} {state['facts']['cause']}", date(2023, 1, 1))


def setup(tmp_path, line: FakeLine):
    memory = FeedbackMemory(BagEmbedder(), describe)
    harness = Harness(line, CFG, AuditSink(tmp_path / "audit"), FinalizationLedger(":memory:"),
                      memory)  # fmt: skip
    return harness.build(InMemorySaver()), memory


def start(graph, claim_id: str) -> dict:
    return graph.invoke({"claim_id": claim_id, "claim": {}},
                        {"configurable": {"thread_id": claim_id}})  # fmt: skip


DENY = HumanDecision(outcome="deny", payout=None, reason="reasons: excluded_driver", adjuster="a1")


def test_pause_resume_finalize_and_remember(tmp_path) -> None:
    graph, memory = setup(tmp_path, FakeLine(critic_passes=[False, False, False]))
    out = start(graph, "C1")
    assert "final" not in out
    queue = pending(graph)
    assert [cid for cid, _ in queue] == ["C1"]
    assert "checks_failed_after_retries" in queue[0][1]["route_reasons"]

    done = resume(graph, "C1", DENY)
    assert done["final"]["decided_by"] == "human" and done["final"]["outcome"] == "deny"
    assert pending(graph) == []
    (episode,) = memory.episodes()
    assert episode.corrected and episode.proposed_outcome == "approve"
    assert episode.human_reason == "reasons: excluded_driver"


def test_resuming_twice_is_refused(tmp_path) -> None:
    graph, memory = setup(tmp_path, FakeLine(critic_passes=[False, False, False]))
    start(graph, "C1")
    resume(graph, "C1", DENY)
    with pytest.raises(ValueError, match="not waiting"):
        resume(graph, "C1", DENY)
    assert len(memory) == 1


def test_queue_lists_only_waiting_claims(tmp_path) -> None:
    graph, memory = setup(tmp_path, FakeLine(critic_passes=[True, False, False, False]))
    start(graph, "AUTO")  # critic passes -> auto-decided
    start(graph, "WAIT")  # critic keeps failing -> human
    assert [cid for cid, _ in pending(graph)] == ["WAIT"]
    assert len(memory) == 0  # auto decisions are not episodes


def test_memory_from_review_feeds_the_next_similar_case(tmp_path) -> None:
    graph, memory = setup(tmp_path, FakeLine(critic_passes=[False, False, False]))
    start(graph, "C1")
    resume(graph, "C1", DENY)
    similar = memory.similar("case C2 collision", k=1)
    assert similar and similar[0].claim_id == "C1"


def test_pending_needs_a_checkpointer(tmp_path) -> None:
    harness = Harness(FakeLine(), CFG, AuditSink(tmp_path / "a"), FinalizationLedger(":memory:"))
    with pytest.raises(ValueError, match="checkpointer"):
        pending(harness.build())
