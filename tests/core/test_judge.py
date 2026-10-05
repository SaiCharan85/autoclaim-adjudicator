from collections.abc import Callable

import judgekit
import pytest

from autoclaim.core.judge import (
    ClientLLM,
    JudgeKitJudge,
    JudgeUnavailableError,
    build_judge,
    build_panel,
    load_rubric,
    to_call_info,
    to_check,
)
from autoclaim.lines.auto.build import RUBRIC_PATH
from autoclaim.llm.client import CallMeta, Structured
from autoclaim.llm.types import LLMUnavailableError

RUBRIC = load_rubric(RUBRIC_PATH)
IDS = [i.id for i in RUBRIC.items]


def answers(no: tuple[str, ...] = (), note: str = "") -> judgekit.JudgeResponse:
    return judgekit.JudgeResponse(
        answers=[
            judgekit.Answer(id=i, answer="no" if i in no else "yes", note=note if i in no else "")
            for i in IDS
        ]
    )


class FakeClient:
    """Duck-typed LLMClient: `structured` answers per role, family per role."""

    def __init__(self, reply: Callable[[str], judgekit.JudgeResponse | Exception]) -> None:
        self.reply = reply
        self.calls: list[tuple[str, frozenset[str]]] = []

    def structured(self, role, instructions, user, schema, exclude_families=frozenset()):
        self.calls.append((role, exclude_families))
        out = self.reply(role)
        if isinstance(out, Exception):
            raise out
        family = f"fam_{role}"
        meta = CallMeta(role=role, model=f"m_{role}", family=family, prompt_tokens=100,
                        completion_tokens=20, latency_s=0.5, validation_retries=1)  # fmt: skip
        return Structured(schema.model_validate(out.model_dump()), meta)


# ---------------------------------------------------------------- rubric


def test_production_rubric_loads_with_severity_tiers() -> None:
    sev = {i.id: i.severity for i in RUBRIC.items}
    assert sev["facts_supported"] is judgekit.Severity.CRITICAL
    assert sev["material_facts_addressed"] is judgekit.Severity.MAJOR
    assert RUBRIC.instructions.startswith("You review the written reasoning")


# ---------------------------------------------------------------- mapping


def test_call_meta_maps_to_call_info() -> None:
    info = to_call_info(CallMeta(role="judge", model="m", family="f", prompt_tokens=10,
                                 completion_tokens=5, latency_s=1.5, cached=True,
                                 validation_retries=1))  # fmt: skip
    assert (info.model, info.family, info.input_tokens, info.output_tokens) == ("m", "f", 10, 5)
    assert (info.latency_s, info.attempts, info.cached) == (1.5, 2, True)


def test_to_check_lists_every_failure_as_judge_issue() -> None:
    verdict = judgekit.score(RUBRIC, answers(no=("facts_supported",), note="date invented"))
    check = to_check(verdict)
    assert not check.passed
    assert [(i.source, i.code, i.detail) for i in check.issues] == [
        ("judge", "facts_supported", "date invented")
    ]


def test_to_check_gives_a_detail_when_the_judge_left_no_note() -> None:
    check = to_check(judgekit.score(RUBRIC, answers(no=("numbers_consistent",))))
    assert check.issues[0].detail == "critical check failed"


# ---------------------------------------------------------------- single judge


def test_passing_judgment_returns_metas_for_audit_and_budget() -> None:
    client = FakeClient(lambda role: answers())
    check, metas = build_judge(client, RUBRIC).evaluate("case", frozenset({"adj_family"}))  # type: ignore[arg-type]
    assert check.passed and check.issues == []
    assert [m.model for m in metas] == ["m_judge"]
    assert client.calls == [("judge", frozenset({"adj_family"}))]  # family exclusion reaches client


def test_failing_judgment() -> None:
    judge = build_judge(FakeClient(lambda r: answers(no=("outcome_consistent",))), RUBRIC)  # type: ignore[arg-type]
    check, _ = judge.evaluate("case", frozenset())
    assert not check.passed and check.issues[0].code == "outcome_consistent"


def test_metas_reset_between_evaluations() -> None:
    judge = build_judge(FakeClient(lambda r: answers()), RUBRIC)  # type: ignore[arg-type]
    judge.evaluate("one", frozenset())
    _, metas = judge.evaluate("two", frozenset())
    assert len(metas) == 1


def test_judge_outage_raises_so_the_graph_routes_to_a_human() -> None:
    judge = build_judge(FakeClient(lambda r: LLMUnavailableError("all models failed")), RUBRIC)  # type: ignore[arg-type]
    with pytest.raises(JudgeUnavailableError, match="all models failed"):
        judge.evaluate("case", frozenset())


def test_client_llm_maps_llm_errors_only() -> None:
    llm = ClientLLM(FakeClient(lambda r: LLMUnavailableError("down")))  # type: ignore[arg-type]
    with pytest.raises(judgekit.LLMCallError):
        llm.complete("s", "u", judgekit.JudgeResponse)
    buggy = ClientLLM(FakeClient(lambda r: KeyError("bug")))  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        buggy.complete("s", "u", judgekit.JudgeResponse)


def test_minor_style_issues_do_not_exist_in_production_rubric() -> None:
    # every production item blocks: a "no" anywhere sends the decision back for a retry
    assert all(RUBRIC.fails_verdict(i.severity) for i in RUBRIC.items)


# ---------------------------------------------------------------- panel


def test_panel_members_use_their_roles_and_avoid_each_others_families() -> None:
    client = FakeClient(lambda role: answers())
    panel = build_panel(client, RUBRIC, ["judge", "judge_b", "judge_c"],  # type: ignore[arg-type]
                        rule=judgekit.PanelRule.MAJORITY)  # fmt: skip
    check, metas = panel.evaluate("case", frozenset({"adj"}))
    assert check.passed
    # majority of 3 decided after 2 clean passes (early stop saves the third request)
    assert [r for r, _ in client.calls] == ["judge", "judge_b"]
    assert client.calls[1][1] == frozenset({"adj", "fam_judge"})
    assert [m.role for m in metas] == ["judge", "judge_b"]


def test_panel_outage_raises() -> None:
    client = FakeClient(lambda role: LLMUnavailableError("down"))
    panel = build_panel(client, RUBRIC, ["judge", "judge_b"])  # type: ignore[arg-type]
    with pytest.raises(JudgeUnavailableError, match="judges available"):
        panel.evaluate("case", frozenset())


def test_judgekit_judge_wraps_any_judgekit_judge() -> None:
    llm = ClientLLM(FakeClient(lambda r: answers()))  # type: ignore[arg-type]
    wrapped = JudgeKitJudge(judgekit.LLMJudge(llm, RUBRIC), [llm])
    assert wrapped.evaluate("case", frozenset())[0].passed
