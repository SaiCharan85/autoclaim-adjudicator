"""Human review queue: find claims paused at human review and resume them with a decision.

A claim the router sends to a human pauses inside `interrupt()`; its state lives in the
checkpointer under its claim id (the thread id). Resuming feeds the adjuster's `HumanDecision`
back in: the graph finalizes it (idempotently: a second resume never decides twice) and the
feedback memory records the episode. Used by the oracle adjuster in evaluations and by the
console.
"""

from typing import Any

from langgraph.types import Command

from autoclaim.core.decision import HumanDecision


def _config(claim_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": claim_id}}


def pending(graph: Any) -> list[tuple[str, dict[str, Any]]]:
    """(claim id, review request) for every claim currently waiting for a human."""
    saver = graph.checkpointer
    if saver is None:
        raise ValueError("the graph has no checkpointer: paused claims cannot be found")
    threads = dict.fromkeys(t.config["configurable"]["thread_id"] for t in saver.list(None))
    out = []
    for claim_id in threads:
        snapshot = graph.get_state(_config(claim_id))
        if snapshot.interrupts:
            out.append((claim_id, snapshot.interrupts[0].value))
    return out


def resume(graph: Any, claim_id: str, decision: HumanDecision) -> dict[str, Any]:
    """Resume one paused claim with the adjuster's decision; returns the final graph state."""
    snapshot = graph.get_state(_config(claim_id))
    if not snapshot.interrupts:
        raise ValueError(f"claim {claim_id!r} is not waiting for human review")
    out: dict[str, Any] = graph.invoke(Command(resume=decision.model_dump()), _config(claim_id))
    return out
