"""Chat-format fine-tuning sets built without any LLM call.

- intake: claim narrative -> the true facts (what the statement says; a fact the story leaves out
  is null and listed in missing_info). Prompts are exactly the production intake prompts.
- judge: rendered judge case -> yes/no answers. Clean reference cases answer yes everywhere;
  planted copies answer no on the items the planted error violates (note = what was changed).
- adjudicator: the exact production adjudicator prompt -> the true decision. The prompt is built
  by the real line code (retrieval, fraud models, prompt assembly); only the LLM steps before it
  are replaced by gold answers (true facts, a coverage analysis that agrees with the truth), so
  training prompts look like production prompts after a perfect intake and coverage step.

Leakage rules: examples come from the TRAIN period only for training and the VALIDATION period
only for validation (never the locked test period); a crash record used in training never
appears in validation. Every example carries its claim id and source record for audits.
"""

import dataclasses
import json
import random
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any, cast

import judgekit
import pandas as pd

from autoclaim.config import JurisdictionProfile
from autoclaim.core.decision import Decision, SelfCheck
from autoclaim.core.judge import load_rubric
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH
from autoclaim.lines.auto.claim import ClaimPackage
from autoclaim.lines.auto.facts import ClaimFacts, DerivedFacts, derive
from autoclaim.lines.auto.line import (
    ADJUDICATE,
    DENY_CODES,
    INTAKE,
    AutoLine,
    CoverageResult,
    FraudAssessment,
    ReasoningStep,
    guard,
)
from autoclaim.llm.client import CallMeta, LLMClient, Structured, system_prompt

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


# ---------------------------------------------------------------- adjudicator

EXCLUSION_CLAUSE = {"excluded_driver": "EXC-EXCLUDED-DRIVER",
                    "wear_and_tear_mechanical": "EXC-MECHANICAL-BREAKDOWN",
                    "business_use_exclusion": "EXC-COMMERCIAL-USE"}  # fmt: skip
CONDITION_CLAUSE = {"hit_and_run_report_condition": "COND-HIT-AND-RUN-POLICE-REPORT",
                    "late_notice_prejudice_review": "COND-LATE-NOTICE"}  # fmt: skip
FINDING = DENY_CODES | {"late_notice_prejudice_review": "notice came more than 30 days late"}
REMOVES_COVERAGE = frozenset(DENY_CODES) - {"below_deductible"}
NEEDS_REVIEW = frozenset({"late_notice_prejudice_review", "suspected_fraud_siu"})
ADJ_REASONS = je.SUPPORTED_REASONS | {"suspected_fraud_siu"}
FRAUD_SENTENCE = ("The fraud score of {score:.2f} is at or above the review threshold, so the "
                  "claim goes to the special investigations unit before any payment.")  # fmt: skip
FRAUD_SUMMARY = ("The fraud score is at or above the review threshold; the model's top factors "
                 "and any red flags are listed in the signals.")  # fmt: skip


class GoldClient:
    """Stands in for the LLM client while building prompts: answers the fraud role with a fixed
    summary and the adjudicator role with the gold decision, and records the adjudicator prompt."""

    def __init__(self) -> None:
        self.decision: Decision | None = None
        self.prompt: str | None = None

    def structured(self, role: str, instructions: str, user: str, schema: type[Any],
                   **_: Any) -> Structured[Any]:  # fmt: skip
        if role == "fraud":
            return Structured(FraudAssessment(summary=FRAUD_SUMMARY), CallMeta(role=role))
        if role == "adjudicator" and self.decision is not None:
            self.prompt = user
            return Structured(self.decision, CallMeta(role=role))
        raise ValueError(f"no gold answer for role {role!r}")


def gold_coverage(x: DerivedFacts, reasons: Sequence[str], cited: Sequence[str],
                  retrieved: Sequence[str]) -> CoverageResult:  # fmt: skip
    """A coverage analysis that agrees with the truth, citing only clauses retrieval returned."""
    rs = set(reasons)
    covered = "no" if rs & REMOVES_COVERAGE else "needs_review" if rs & NEEDS_REVIEW else "yes"
    clause_reason = {c: r for r, c in (EXCLUSION_CLAUSE | CONDITION_CLAUSE).items() if r in rs}
    steps = []
    for cid in cited:
        if cid not in retrieved:
            continue
        if cid in clause_reason:
            finding = f"Applies: {FINDING[clause_reason[cid]]}."
        elif cid in je.PART_CLAUSE.values():
            finding = f"The loss falls under {x.coverage_part} coverage."
        else:
            finding = "Sets the amount payable."
        steps.append(ReasoningStep(clause_id=cid, finding=finding))
    if not steps:
        steps = [ReasoningStep(clause_id=retrieved[0], finding="Reviewed; it does not change "
                               "coverage.")]  # fmt: skip
    return CoverageResult(
        coverage_part=x.coverage_part,
        covered=covered,  # type: ignore[arg-type]
        exclusions_triggered=[EXCLUSION_CLAUSE[r] for r in reasons if r in EXCLUSION_CLAUSE],
        conditions_unmet=[CONDITION_CLAUSE[r] for r in reasons if r in CONDITION_CLAUSE],
        reasoning=steps[:8],
    )


def adjudicator_example(row: pd.Series, line: AutoLine, jur: JurisdictionProfile,
                        fraud_review_score: float) -> dict[str, Any] | None:  # fmt: skip
    """One training example from one simulated claim, or None when the truth can't be learned
    from the prompt (fraud the score does not show) or no faithful reference exists."""
    outcome = str(row["gt_decision"])
    reasons = je.true_reasons(row)
    if not reasons or not set(reasons) <= ADJ_REASONS:
        return None
    facts, pkg = je.true_facts(row), je.package(row)
    x = derive(facts, pkg, jur)
    truth = je._truth(pkg, x)
    plain = [r for r in reasons if r != "suspected_fraud_siu"]
    if not all(truth[r] for r in plain):
        return None
    if outcome == "approve" and abs(x.payout - float(row["gt_payout"])) > 1.0:
        return None  # an inflated estimate: the derived payout is not the true one
    gold = GoldClient()
    lab = dataclasses.replace(line, client=cast(LLMClient, gold))
    state: dict[str, Any] = {
        "claim": pkg.model_dump(mode="json"),
        "guardrails": {"narrative": pkg.narrative},
        "facts": {"extracted": facts.model_dump(mode="json"), "derived": x.model_dump(mode="json")},
    }
    state |= lab.fraud(state).update  # type: ignore[arg-type]
    score = float(state["fraud"]["model_score"])
    if "suspected_fraud_siu" in reasons and score < fraud_review_score:
        if not plain:
            return None  # fraud the adjudicator cannot see: unlearnable from this prompt
        reasons = plain
    explanation, cited = je.explain(facts, x, pkg, outcome, reasons, jur)
    if "suspected_fraud_siu" in reasons:
        head, _, tail = explanation.rpartition(" We ")
        explanation = f"{head} {FRAUD_SENTENCE.format(score=score)} We {tail}".strip()
    hits = lab.retriever.search(lab.retrieval_query(facts, x), lab.top_k, lab.graph_hops,
                                max_expanded=lab.max_expanded)  # fmt: skip
    retrieved = [h.clause.id for h in hits]
    cov = gold_coverage(x, reasons, cited, retrieved)
    state["coverage"] = cov.model_dump(mode="json") | {"retrieved": retrieved,
                                                       "uncited_ids_dropped": []}  # fmt: skip
    gold.decision = Decision(
        outcome=outcome,  # type: ignore[arg-type]
        payout=x.payout if outcome == "approve" else None,
        reasons=list(reasons),
        cited_clauses=cited,
        explanation=explanation,
        confidence=0.9 if outcome != "escalate" else 0.8,
        self_check=SelfCheck(
            every_fact_from_claim=True,
            every_denial_cites_clause=True,
            numbers_from_tools=True,
            fraud_signals_considered=True,
        ),
    )
    lab.adjudicate(state)  # type: ignore[arg-type]
    assert gold.prompt is not None
    return chat(system_prompt(ADJUDICATE, Decision), gold.prompt,
                gold.decision.model_dump_json(), claim_id=str(row["claim_id"]),
                source_record=str(row["source_record"]), outcome=outcome,
                primary_reason=reasons[0],
                loss_date=str(pd.Timestamp(row["loss_date"]).date()))  # fmt: skip


def adjudicator_examples(claims: pd.DataFrame, start: str, end: str, n: int, line: AutoLine,
                         jur: JurisdictionProfile, fraud_review_score: float, seed: int,
                         approve_share: float = 0.35) -> list[dict[str, Any]]:  # fmt: skip
    """Up to `n` examples from loss dates in [start, end): `approve_share` approvals, the rest
    spread evenly over the primary deny/escalate reasons (rare ones are not swamped)."""
    pool = claims[(claims["loss_date"] >= start) & (claims["loss_date"] < end)]
    pool = pool.sample(frac=1.0, random_state=seed)
    n_approve = round(n * approve_share)
    other = sorted(ADJ_REASONS - {"covered_loss"})
    cap = {"covered_loss": n_approve} | dict.fromkeys(other, -(-(n - n_approve) // len(other)))
    buckets: dict[str, list[dict[str, Any]]] = {r: [] for r in cap}
    for _, row in pool.iterrows():
        if all(len(buckets[r]) >= cap[r] for r in cap):
            break
        primary = (je.true_reasons(row) or [""])[0]
        if primary not in cap or len(buckets[primary]) >= cap[primary]:
            continue
        ex = adjudicator_example(row, line, jur, fraud_review_score)
        if (
            ex is not None
            and len(buckets[ex["meta"]["primary_reason"]]) < cap[ex["meta"]["primary_reason"]]
        ):
            buckets[ex["meta"]["primary_reason"]].append(ex)
    out = [ex for b in buckets.values() for ex in b][:n]
    random.Random(seed).shuffle(out)
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
