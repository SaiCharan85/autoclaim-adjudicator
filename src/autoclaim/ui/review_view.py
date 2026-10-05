"""Pure helpers for the adjuster console (no Streamlit here, so they are unit-tested).

The console shows each waiting claim's review request (why it was escalated, what the harness
proposed, the deterministic case summary) and turns the adjuster's form into a `HumanDecision`.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autoclaim.core.decision import HumanDecision

OUTCOMES = ("approve", "deny", "escalate")


class FormError(ValueError):
    """The adjuster's form cannot become a decision (shown to the adjuster, nothing resumes)."""


def queue_rows(queue: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """One table row per waiting claim."""
    rows = []
    for claim_id, req in queue:
        prop = req.get("proposed_decision") or {}
        rows.append({
            "claim": claim_id,
            "why escalated": ", ".join(req.get("route_reasons", [])) or "-",
            "proposed": prop.get("outcome") or "none (failsafe)",
            "proposed payout": prop.get("payout"),
            "confidence": prop.get("confidence"),
        })  # fmt: skip
    return rows


def _money(x: Any) -> str:
    return "-" if x is None else f"${float(x):,.2f}"


def case_markdown(req: dict[str, Any]) -> str:
    """The review request as readable sections (deterministic, from the case summary)."""
    summary = req.get("case_summary") or {}
    prop = req.get("proposed_decision") or {}
    lines = [f"### {req['claim_id']}",
             f"**Why escalated:** {', '.join(req.get('route_reasons', [])) or '-'}"]  # fmt: skip
    if summary.get("failsafe"):
        lines.append(f"**Failsafe:** {summary['failsafe']}")
    if prop:
        lines += [
            "",
            f"**Harness proposal:** {prop.get('outcome')} {_money(prop.get('payout'))} "
            f"(reasons: {', '.join(prop.get('reasons', [])) or '-'}; "
            f"confidence {prop.get('confidence', '-')})",
            f"**Cited clauses:** {', '.join(prop.get('cited_clauses', [])) or '-'}",
            f"**Explanation:** {prop.get('explanation', '-')}",
        ]
    if "estimate" in summary:
        lines.append(f"**Repair estimate:** {_money(summary['estimate'])}")
    derived = (summary.get("facts") or {}).get("derived") or {}
    if derived:
        lines.append(
            f"**Deterministic:** {derived.get('coverage_part')} coverage "
            f"{'on' if derived.get('part_on_policy') else 'NOT on'} policy; deductible "
            f"{_money(derived.get('deductible'))}; approval would pay "
            f"{_money(derived.get('payout'))}; "
            f"driver {derived.get('driver_role')}; notice {derived.get('notice_days')} days"
            f"{' (late)' if derived.get('late_notice') else ''}"
        )
    fraud = summary.get("fraud")
    if fraud:
        flags = ", ".join(fraud.get("red_flags", [])) or "none"
        lines.append(f"**Fraud:** score {fraud.get('score', 0):.2f}; red flags: {flags}")
    for key, label in (("critic", "Critic"), ("judge", "Judge")):
        check = summary.get(key)
        if check and not check.get("passed"):
            issues = "; ".join(f"{i.get('code')}: {i.get('detail')}" for i in check["issues"])
            lines.append(f"**{label} issues:** {issues}")
    if summary.get("statement"):
        lines += ["", "**Claimant's statement:**", f"> {summary['statement']}"]
    return "\n".join(lines)


def decision_from_form(outcome: str, payout: float | None, reason: str, adjuster: str
                       ) -> HumanDecision:  # fmt: skip
    """Validate the adjuster's form. An approval needs a payout; every decision needs a reason
    (feedback memory learns from it) and the adjuster's name (audit)."""
    if outcome not in OUTCOMES:
        raise FormError(f"outcome must be one of {OUTCOMES}")
    if not reason.strip():
        raise FormError("give a reason: feedback memory learns from it")
    if not adjuster.strip():
        raise FormError("enter your name for the audit trail")
    if outcome == "approve" and payout is None:
        raise FormError("an approval needs a payout")
    try:
        return HumanDecision.model_validate({
            "outcome": outcome, "payout": payout if outcome == "approve" else None,
            "reason": reason.strip(), "adjuster": adjuster.strip(),
        })  # fmt: skip
    except ValidationError as exc:
        raise FormError(str(exc.errors()[0]["msg"])) from exc


def audit_rows(audit_dir: Path, claim_id: str) -> list[dict[str, Any]]:
    """The claim's audit trail (one row per node run), newest last."""
    path = audit_dir / f"{claim_id}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        rows.append({"time": e.get("ts", "")[11:19], "node": e.get("node"),
                     "status": e.get("status"), "model": e.get("model") or "-",
                     "tokens": (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0),
                     "outputs": json.dumps(e.get("outputs", {}))[:160]})  # fmt: skip
    return rows
