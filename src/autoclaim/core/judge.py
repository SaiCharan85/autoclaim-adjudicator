"""Judge: grades the REASONING behind a decision, never the decision itself.

The harness depends only on the small `Judge` interface below. JudgeKit (rubrics with severity
tiers, the LLM judge runner, panels) sits behind it through a thin adapter, so the judging
library can change without touching the graph.

Fail safe: if the judge cannot run (every model failed or is out of budget), `evaluate` raises
`JudgeUnavailableError`; the graph turns that into a failsafe and the claim goes to a human. A
missing judgment is never fed back to the adjudicator as if it were a "no" (that would spend
retries on a judge outage).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeVar

import judgekit
from pydantic import BaseModel

from autoclaim.core.decision import CheckResult, Issue
from autoclaim.llm.client import CallMeta, LLMClient
from autoclaim.llm.types import LLMError

T = TypeVar("T", bound=BaseModel)


class Judge(Protocol):
    def evaluate(
        self, case: str, exclude_families: frozenset[str]
    ) -> tuple[CheckResult, list[CallMeta]]: ...


class JudgeUnavailableError(RuntimeError):
    """No judgment could be made; route the claim to a human."""


def to_call_info(meta: CallMeta) -> judgekit.CallInfo:
    return judgekit.CallInfo(
        model=meta.model,
        family=meta.family,
        input_tokens=meta.prompt_tokens,
        output_tokens=meta.completion_tokens,
        latency_s=meta.latency_s,
        attempts=1 + meta.validation_retries,
        cached=meta.cached,
    )


class ClientLLM:
    """JudgeKit's `StructuredLLM` over our client (cache, budgets, fallback, validation retry).
    Keeps the `CallMeta` of each call so the graph can audit and budget it."""

    def __init__(self, client: LLMClient, role: str = "judge") -> None:
        self.client = client
        self.role = role
        self.metas: list[CallMeta] = []

    def complete(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        avoid_families: frozenset[str] = frozenset(),
    ) -> judgekit.StructuredResult[T]:
        try:
            res = self.client.structured(
                self.role, system, user, schema, exclude_families=avoid_families
            )
        except LLMError as exc:
            raise judgekit.LLMCallError(str(exc)) from exc
        self.metas.append(res.meta)
        return judgekit.StructuredResult(res.value, (to_call_info(res.meta),))


def to_check(verdict: judgekit.Verdict) -> CheckResult:
    """Pass/fail follows the rubric's severity tiers; every failed item (minor ones too) becomes
    an issue, so a retry gets the full feedback."""
    issues = [
        Issue(source="judge", code=r.item_id, detail=r.note or f"{r.severity} check failed")
        for r in verdict.failures
    ]
    return CheckResult(passed=verdict.passed, issues=issues)


class JudgeKitJudge:
    """Our `Judge` backed by any JudgeKit judge (one LLM judge or a panel)."""

    def __init__(self, judge: judgekit.Judge, llms: Sequence[ClientLLM]) -> None:
        self.judge = judge
        self.llms = tuple(llms)

    def evaluate(
        self, case: str, exclude_families: frozenset[str]
    ) -> tuple[CheckResult, list[CallMeta]]:
        for llm in self.llms:
            llm.metas.clear()
        verdict = self.judge.evaluate(case, avoid_families=exclude_families)
        if not verdict.available:
            raise JudgeUnavailableError(verdict.error)
        return to_check(verdict), [m for llm in self.llms for m in llm.metas]


def load_rubric(path: Path) -> judgekit.Rubric:
    return judgekit.load_rubric(path)


def build_judge(client: LLMClient, rubric: judgekit.Rubric, role: str = "judge") -> JudgeKitJudge:
    """The production judge: one LLM judge on the `role` model chain."""
    llm = ClientLLM(client, role)
    return JudgeKitJudge(judgekit.LLMJudge(llm, rubric, name=role), [llm])


def build_panel(
    client: LLMClient,
    rubric: judgekit.Rubric,
    roles: Sequence[str],
    rule: judgekit.PanelRule = judgekit.PanelRule.MAJORITY,
) -> JudgeKitJudge:
    """A multi-judge panel, one member per role (each role has its own model chain); members
    avoid each other's model families."""
    llms = [ClientLLM(client, r) for r in roles]
    members = [judgekit.LLMJudge(llm, rubric, name=llm.role) for llm in llms]
    return JudgeKitJudge(judgekit.JudgePanel(members, rubric, rule=rule, name="panel"), llms)
