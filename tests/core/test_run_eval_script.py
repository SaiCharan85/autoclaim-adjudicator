"""scripts/run_eval.py's claim loop on the fake line (no LLM): pause -> adjuster -> resume."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from test_graph import CFG, FakeLine

from autoclaim.core.audit import AuditSink
from autoclaim.core.decision import HumanDecision
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.graph import Harness
from autoclaim.lines.auto import harness_eval as he

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", SCRIPT)
assert _spec is not None and _spec.loader is not None
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


class Adjuster:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def decide(self, request) -> HumanDecision:
        self.asked.append(request["claim_id"])
        return HumanDecision(outcome="deny", payout=None, reason="reasons: x", adjuster="t")


def app_for(tmp_path, line: FakeLine):
    harness = Harness(line, CFG, AuditSink(tmp_path), FinalizationLedger(":memory:"))
    return SimpleNamespace(graph=harness.build(InMemorySaver()))


def test_escalated_claim_is_answered_and_finished(tmp_path) -> None:
    adj = Adjuster()
    out = run_eval.run_claim(app_for(tmp_path, FakeLine(critic_passes=[False] * 3)),
                             {"claim_id": "C1"}, adj)  # fmt: skip
    assert out["final"]["decided_by"] == "human" and adj.asked == ["C1"]


def test_auto_decided_claim_never_asks_the_adjuster(tmp_path) -> None:
    adj = Adjuster()
    out = run_eval.run_claim(app_for(tmp_path, FakeLine()), {"claim_id": "C2"}, adj)
    assert out["final"]["decided_by"] == "auto" and adj.asked == []


def test_scores_checkpoint_round_trip(tmp_path) -> None:
    truth = he.Truth(outcome="deny", payout=None, reasons=("x",), traps=(), is_fraud=False)
    out = run_eval.run_claim(app_for(tmp_path, FakeLine(critic_passes=[False] * 3)),
                             {"claim_id": "C3"}, Adjuster())  # fmt: skip
    path = tmp_path / "scores.jsonl"
    path.write_text(he.score(out, truth, "full").model_dump_json() + "\n", encoding="utf-8")
    (loaded,) = run_eval.load_scores(path)
    assert loaded.claim_id == "C3" and loaded.correct
    assert run_eval.load_scores(tmp_path / "missing.jsonl") == []
