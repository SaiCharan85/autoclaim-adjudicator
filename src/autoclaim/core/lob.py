"""The seam between the shared harness and a line of business (auto today; flood, health later).

A line supplies its nodes; the core supplies the topology, routing, retries, audit, budgets,
fail-safe handling, idempotent finalization and human review.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from autoclaim.core.state import ClaimState
from autoclaim.llm.client import CallMeta


@dataclass
class NodeResult:
    update: dict[str, Any]  # state keys to write
    calls: list[CallMeta] = field(default_factory=list)  # LLM calls made (audit + budget)
    outputs: dict[str, Any] = field(default_factory=dict)  # compact audit summary


class LineOfBusiness(Protocol):
    name: str

    def guardrails(self, state: ClaimState) -> NodeResult: ...  # sets guardrails.rejected

    def intake(self, state: ClaimState) -> NodeResult: ...

    def coverage(self, state: ClaimState) -> NodeResult: ...

    def fraud(self, state: ClaimState) -> NodeResult: ...

    def adjudicate(self, state: ClaimState) -> NodeResult: ...

    def critic(self, state: ClaimState) -> NodeResult: ...  # sets critic (CheckResult)

    def judge(self, state: ClaimState) -> NodeResult: ...  # sets judge (CheckResult)

    def hard_escalations(self, state: ClaimState) -> list[str]: ...

    def fraud_score(self, state: ClaimState) -> float | None: ...

    def case_summary(self, state: ClaimState) -> dict[str, Any]: ...


class MemoryStore(Protocol):
    def remember(self, state: ClaimState) -> None: ...


class NoMemory:
    def remember(self, state: ClaimState) -> None:
        return None
