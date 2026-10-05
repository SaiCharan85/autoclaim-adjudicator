import pandas as pd
import pytest

from autoclaim.core.lob import NodeResult
from autoclaim.lines.auto import harness_eval as he

APPROVE = he.Truth(outcome="approve", payout=1650.0, reasons=("covered_loss",), traps=(),
                   is_fraud=False)  # fmt: skip
DENY_TRAP = he.Truth(outcome="deny", payout=None, reasons=("no_comprehensive_coverage",),
                     traps=("animal_strike",), is_fraud=False)  # fmt: skip
FRAUD = he.Truth(outcome="escalate", payout=None, reasons=("suspected_fraud_siu",), traps=(),
                 is_fraud=True)  # fmt: skip


def state(cid: str, by: str, outcome: str, payout: float | None = None,
          proposed: str | None = None, calls: int = 4) -> dict:  # fmt: skip
    return {"claim_id": cid, "final": {"decided_by": by, "outcome": outcome, "payout": payout},
            "decision": {"outcome": proposed or outcome}, "route": {"reasons": []},
            "llm_calls": calls, "tokens": 1000 * calls}  # fmt: skip


def s(cid: str, truth: he.Truth, by: str, outcome: str, payout: float | None = None, **kw):
    return he.score(state(cid, by, outcome, payout, **kw), truth, "full")


# ---------------------------------------------------------------- truth and scoring


def test_truth_of_row() -> None:
    row = pd.Series({"gt_decision": "approve", "gt_reasons": float("nan"), "gt_payout": 900.0,
                     "gt_traps": "animal_strike;late_notice", "gt_is_fraud": False})  # fmt: skip
    t = he.truth_of(row)
    assert (t.outcome, t.payout, t.reasons) == ("approve", 900.0, ("covered_loss",))
    assert t.traps == ("animal_strike", "late_notice") and not t.is_fraud
    deny = he.truth_of(row.copy().replace({"approve": "deny"}))
    assert deny.payout is None and deny.reasons == ()


def test_correct_auto_approval() -> None:
    x = s("A", APPROVE, "auto", "approve", 1650.0)
    assert x.auto and x.correct and not x.wrong_auto_approve and x.overpaid == 0.0


def test_trap_leak_is_a_wrong_auto_decision_on_a_trap_claim() -> None:
    leak = s("B", DENY_TRAP, "auto", "approve", 1650.0)
    assert leak.trap_leaked and leak.wrong_auto_approve and leak.overpaid == 1650.0
    caught = s("C", DENY_TRAP, "auto", "deny")
    assert not caught.trap_leaked
    escalated = s("D", DENY_TRAP, "human", "approve", 1650.0)  # adjuster error, not the harness
    assert not escalated.trap_leaked and escalated.overpaid == 0.0


def test_fraud_auto_paid() -> None:
    assert s("E", FRAUD, "auto", "approve", 2000.0).fraud_auto_paid
    assert not s("F", FRAUD, "human", "escalate").fraud_auto_paid


def test_overpayment_on_a_correct_approval() -> None:
    assert s("G", APPROVE, "auto", "approve", 1800.0).overpaid == pytest.approx(150.0)


# ---------------------------------------------------------------- metrics


SCORES = [
    s("1", APPROVE, "auto", "approve", 1650.0),
    s("2", DENY_TRAP, "auto", "approve", 1650.0),  # trap leaked, wrong approval
    s("3", DENY_TRAP, "human", "deny", proposed="approve"),
    s("4", FRAUD, "human", "escalate", calls=6),
]


def test_metrics_on_a_hand_built_set() -> None:
    assert he.metric(SCORES, "auto_rate") == 0.5
    assert he.metric(SCORES, "auto_accuracy") == 0.5
    assert he.metric(SCORES, "wrong_auto_approve_rate") == 0.25
    assert he.metric(SCORES, "trap_leak_rate") == 0.5
    assert he.metric(SCORES, "fraud_auto_paid_rate") == 0.0
    assert he.metric(SCORES, "proposal_accuracy") == 0.5  # #2 and #3 proposed approve wrongly
    assert he.metric(SCORES, "calls_per_claim") == pytest.approx(4.5)
    assert he.metric(SCORES, "overpaid_per_claim") == pytest.approx(1650.0 / 4)


def test_metrics_undefined_without_data() -> None:
    assert he.metric([], "auto_rate") is None
    assert he.metric([SCORES[0]], "trap_leak_rate") is None  # no trap claims
    with pytest.raises(ValueError, match="unknown metric"):
        he.metric(SCORES, "vibes")


def test_headline_metrics_are_fixed_and_known() -> None:
    for m in he.HEADLINE:
        he.metric(SCORES, m)  # every pre-registered metric is computable
    assert set(he.ALL_METRICS) >= he.LOWER_IS_BETTER


def test_summarize_gives_cis() -> None:
    est = he.summarize(SCORES * 5, n_boot=100)
    assert est["auto_rate"].value == 0.5
    assert est["auto_rate"].low is not None and est["auto_rate"].low <= 0.5


def test_paired_difference_uses_shared_claims_only() -> None:
    better = [
        s("1", APPROVE, "auto", "approve", 1650.0),
        s("2", DENY_TRAP, "auto", "deny"),
        s("3", DENY_TRAP, "auto", "deny"),
        s("9", APPROVE, "auto", "approve", 1650.0),
    ]
    d = he.paired(SCORES, better, "trap_leak_rate", n_boot=200)
    assert d.value == pytest.approx(-0.5) and d.n == 3


def test_markdown_table_has_every_metric_and_paired_section() -> None:
    md = he.markdown_table({"full": SCORES * 3, "few_shot": SCORES * 3}, n_boot=50)
    for m in he.ALL_METRICS:
        assert f"| {m} |" in md
    assert "Paired difference few_shot - full" in md and "| real gap? |" in md


# ---------------------------------------------------------------- ablations


class Line:
    name = "x"

    def judge(self, state):
        return NodeResult({"judge": {"passed": False, "issues": ["real"]}})

    def critic(self, state):
        return NodeResult({"critic": {"passed": False, "issues": ["real"]}})

    def intake(self, state):
        return "delegated"


def test_ablated_judge_always_passes_and_rest_delegates() -> None:
    a = he.AblatedLine(Line(), no_judge=True)
    assert a.judge({}).update["judge"]["passed"] is True and a.judge({}).calls == []
    assert a.critic({}).update["critic"]["passed"] is False  # untouched
    assert a.intake({}) == "delegated" and a.name == "x"


def test_ablated_critic() -> None:
    a = he.AblatedLine(Line(), no_critic=True)
    assert a.critic({}).update["critic"]["passed"] is True
    assert a.judge({}).update["judge"]["passed"] is False


@pytest.mark.parametrize(
    ("failsafe", "stop"),
    [("coverage: LLMUnavailableError: role 'coverage': every model failed: ['g: daily safety "
      "limit reached (0 requests left)']", True),
     ("judge: JudgeUnavailableError: every model failed: ['x: rate limited: quota']", True),
     ("adjudicator: LLMUnavailableError: every model failed: ['x: invalid output']", False),
     ("budget: claim used 15 calls", False), (None, False)],
)  # fmt: skip
def test_quota_exhaustion_is_not_scored_as_an_escalation(failsafe, stop) -> None:
    assert he.quota_exhausted({"failsafe": failsafe}) is stop
