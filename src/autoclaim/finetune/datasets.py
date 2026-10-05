"""Chat-format fine-tuning sets built without any LLM call.

- intake: claim narrative -> the true facts (what the statement says; a fact the story leaves out
  is null and listed in missing_info). Prompts are exactly the production intake prompts.
- judge: rendered judge case -> yes/no answers. Clean reference cases answer yes everywhere;
  planted copies answer no on the items the planted error violates (note = what was changed).

Leakage rules: examples come from the TRAIN period only for training and the VALIDATION period
only for validation (never the locked test period); a crash record used in training never
appears in validation. Every example carries its claim id and source record for audits.
"""

import json
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

import judgekit
import pandas as pd

from autoclaim.core.judge import load_rubric
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH
from autoclaim.lines.auto.claim import ClaimPackage
from autoclaim.lines.auto.facts import ClaimFacts
from autoclaim.lines.auto.line import INTAKE, guard
from autoclaim.llm.client import system_prompt

# Facts a narrative may leave out that intake extracts: (ClaimFacts field, missing_info name)
OMITTED_FIELDS = {
    "witness_count": ("witnesses", "witnesses"),
    "police_report_hours": ("police_report_hours", "police_report_timing"),
}


def chat(system: str, user: str, assistant: str, **meta: Any) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta,
    }


def period_of(loss_date: str | date, val_start: date, test_start: date) -> str:
    d = pd.Timestamp(loss_date).date()
    if d >= test_start:
        return "test"
    return "val" if d >= val_start else "train"


# ---------------------------------------------------------------- intake


def intake_target(row: pd.Series, leave_out: str | None) -> ClaimFacts:
    facts = je.true_facts(row)
    if leave_out in OMITTED_FIELDS:
        field, missing = OMITTED_FIELDS[leave_out]
        facts = facts.model_copy(update={field: None, "missing_info": [missing]})
    if facts.police_report_hours is not None:  # stories say "the next morning", not 14.2 hours
        facts = facts.model_copy(update={"police_report_hours": round(facts.police_report_hours)})
    return facts


def intake_examples(
    records: Iterable[dict[str, Any]], claims: pd.DataFrame
) -> list[dict[str, Any]]:
    """One example per narrative package (packages from data/sim/packages/dev.jsonl)."""
    system = system_prompt(INTAKE, ClaimFacts)
    by_id = claims.set_index("claim_id")
    out = []
    for rec in records:
        pkg = ClaimPackage.model_validate(rec["package"])
        row = pd.Series({**by_id.loc[pkg.claim_id].to_dict(), "claim_id": pkg.claim_id})
        g = guard(rec["package"])
        if g["rejected"]:
            continue
        user = (
            f"channel: {pkg.channel}\nreport_date: {pkg.report_date}\nstatement:\n{g['narrative']}"
        )
        target = intake_target(row, (rec.get("style") or {}).get("leave_out"))
        out.append(chat(system, user, target.model_dump_json(), claim_id=pkg.claim_id,
                        source_record=str(row["source_record"]),
                        loss_date=str(pd.Timestamp(row["loss_date"]).date())))  # fmt: skip
    return out


# ---------------------------------------------------------------- judge


def judge_answer(rubric: judgekit.Rubric, wrong: frozenset[str], note: str) -> str:
    answers = [
        judgekit.Answer(id=i, answer="no" if i in wrong else "yes", note=note if i in wrong else "")
        for i in rubric.ids
    ]
    return judgekit.JudgeResponse(answers=answers).model_dump_json()


def judge_examples(
    samples: Sequence[judgekit.Sample],
    error_types: Sequence[judgekit.ErrorType],
    seed: int,
    source_records: dict[str, str],
    loss_dates: dict[str, str],
) -> list[dict[str, Any]]:
    rubric = load_rubric(RUBRIC_PATH)
    system = system_prompt(judgekit.render_system(rubric), judgekit.JudgeResponse)
    suite = judgekit.plant(samples, error_types, seed=seed)
    out = []
    for s in suite.clean:
        out.append(chat(system, je.render(s), judge_answer(rubric, frozenset(), ""),
                        claim_id=s.id, error_type=None, source_record=source_records[s.id],
                        loss_date=loss_dates[s.id]))  # fmt: skip
    for p in suite.planted:
        note = p.description[:200]
        out.append(chat(system, je.render(p.sample),
                        judge_answer(rubric, p.expected_items, note), claim_id=p.sample_id,
                        error_type=p.error_type, source_record=source_records[p.sample_id],
                        loss_date=loss_dates[p.sample_id]))  # fmt: skip
    return out


# ---------------------------------------------------------------- leakage-safe split


def split(
    examples: Sequence[dict[str, Any]], val_start: date, test_start: date
) -> dict[str, list[dict[str, Any]]]:
    """Train-period examples train, validation-period examples validate; test-period examples are
    refused; a validation example sharing a crash record with training is dropped."""
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for ex in examples:
        period = period_of(ex["meta"]["loss_date"], val_start, test_start)
        if period == "test":
            raise ValueError(f"test-period claim {ex['meta']['claim_id']} in fine-tuning data")
        out[period].append(ex)
    used = {ex["meta"]["source_record"] for ex in out["train"]}
    out["val"] = [ex for ex in out["val"] if ex["meta"]["source_record"] not in used]
    return out


def to_jsonl(examples: Sequence[dict[str, Any]]) -> str:
    return "".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples)
