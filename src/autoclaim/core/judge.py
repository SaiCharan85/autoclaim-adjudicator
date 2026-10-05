"""Judge: grades the REASONING behind a decision, never the decision itself.

Labels evaluate outcomes; the judge evaluates explanations: is every statement supported by the
claim facts and the cited clause text, is anything material ignored, does the explanation agree
with the outcome? Yes/no questions from a YAML rubric (a yes always means "fine").

`Judge` is a protocol so the backend can be swapped: an API model from a different family than
the adjudicator (default), or a local fine-tuned small judge (Step 7.5).
"""

from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, Field

from autoclaim.core.decision import CheckResult, Issue
from autoclaim.llm.client import CallMeta, LLMClient


class RubricItem(BaseModel):
    id: str
    question: str  # phrased so that "yes" = passes


class Rubric(BaseModel):
    name: str
    version: str
    instructions: str
    items: list[RubricItem] = Field(min_length=1)


def load_rubric(path: Path) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Answer(BaseModel):
    id: str
    answer: Literal["yes", "no"]
    note: str = Field(default="", max_length=300, description="why, if no")


class Verdict(BaseModel):
    answers: list[Answer]


def to_check(rubric: Rubric, verdict: Verdict) -> CheckResult:
    """Every rubric item must be answered yes; unanswered items count as failures."""
    by_id = {a.id: a for a in verdict.answers}
    issues = []
    for item in rubric.items:
        ans = by_id.get(item.id)
        if ans is None:
            issues.append(Issue(source="judge", code=item.id, detail="not answered by the judge"))
        elif ans.answer == "no":
            issues.append(Issue(source="judge", code=item.id, detail=ans.note or item.question))
    return CheckResult(passed=not issues, issues=issues)


class Judge(Protocol):
    def evaluate(
        self, case: str, exclude_families: frozenset[str]
    ) -> tuple[CheckResult, list[CallMeta]]: ...


def rubric_prompt(rubric: Rubric) -> str:
    questions = "\n".join(f"- {i.id}: {i.question}" for i in rubric.items)
    return f"{rubric.instructions}\n\nAnswer every question yes or no:\n{questions}"


class LLMJudge:
    def __init__(self, client: LLMClient, rubric: Rubric, role: str = "judge") -> None:
        self.client = client
        self.rubric = rubric
        self.role = role
        self.instructions = rubric_prompt(rubric)

    def evaluate(
        self, case: str, exclude_families: frozenset[str]
    ) -> tuple[CheckResult, list[CallMeta]]:
        res = self.client.structured(
            self.role, self.instructions, case, Verdict, exclude_families=exclude_families
        )
        return to_check(self.rubric, res.value), [res.meta]
