"""Wire the flood harness: the SAME core graph as auto, with the flood line plugged in."""

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from autoclaim.config import CarrierConfig, load_carrier_config
from autoclaim.core.audit import AuditSink
from autoclaim.core.budget import BudgetConfig
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.graph import Harness, HarnessConfig
from autoclaim.core.judge import Judge
from autoclaim.core.router import RouterConfig
from autoclaim.lines.flood.line import FloodLine
from autoclaim.lines.flood.policy import POLICY_PATH
from autoclaim.llm.client import LLMClient
from autoclaim.retrieval.corpus import load_policy


def flood_harness_config(cfg: CarrierConfig) -> HarnessConfig:
    flood = (cfg.model_extra or {})["flood"]
    return HarnessConfig(
        max_retries=cfg.harness.max_retries,
        router=RouterConfig.model_validate(flood["router"]),
        budget=BudgetConfig.model_validate(flood["budgets"]),
    )


def build_flood(
    cfg: CarrierConfig | None = None,
    client: LLMClient | None = None,
    judge: Judge | None = None,
    state_dir: Path | None = None,
) -> tuple[Harness, Any]:
    """(harness, compiled graph). `client=None`: template explanations, zero LLM calls."""
    cfg = cfg or load_carrier_config()
    line = FloodLine(load_policy(POLICY_PATH), client=client, judge_impl=judge)
    ledger = FinalizationLedger(":memory:" if state_dir is None else str(state_dir / "ledger.db"))
    audit = AuditSink(None if state_dir is None else state_dir / "audit")
    harness = Harness(line, flood_harness_config(cfg), audit, ledger)
    return harness, harness.build(InMemorySaver())
