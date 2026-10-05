"""Graph state: plain JSON-able values only (Pydantic models are dumped at node boundaries), so
checkpoints serialize safely and the audit trail is readable."""

import operator
from typing import Annotated, Any, TypedDict


def keep_first(current: str | None, new: str | None) -> str | None:
    """Reducer: the first failure wins (coverage and fraud run in parallel and may both fail)."""
    return current or new


class ClaimState(TypedDict, total=False):
    claim_id: str
    claim: dict[str, Any]  # the raw claim package (line-specific shape)
    guardrails: dict[str, Any]
    facts: dict[str, Any]  # intake output + deterministic derivations
    coverage: dict[str, Any]
    fraud: dict[str, Any]
    decision: dict[str, Any]
    critic: dict[str, Any]
    judge: dict[str, Any]
    issues: list[dict[str, Any]]  # critic/judge issues fed back to the adjudicator on retry
    retries: int
    route: dict[str, Any]
    final: dict[str, Any]
    human: dict[str, Any]
    failsafe: Annotated[str | None, keep_first]  # once set, later nodes skip; router -> human
    llm_calls: Annotated[int, operator.add]
    tokens: Annotated[int, operator.add]
    audit: Annotated[list[dict[str, Any]], operator.add]
