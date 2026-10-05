import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoclaim.config import ModelLimits, ModelsConfig, ProviderConfig, RoleConfig
from autoclaim.lines.auto import fraud_benchmark as fb
from autoclaim.lines.auto import fraud_llm
from autoclaim.llm.cache import LLMCache
from autoclaim.llm.client import LLMClient
from autoclaim.llm.types import ChatRequest, ChatResult
from autoclaim.llm.usage import UsageLedger
from autoclaim.ml import models
from autoclaim.ml.frame import Y

MODEL = "p:m"


class EchoProvider:
    """Scores each claim id in the prompt; claims mentioning `tools:` get a higher score."""

    name = "p"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResult:
        self.calls += 1
        user = request.messages[-1].content
        ids = re.findall(r"\[(C\d+)\]", user)
        p = 0.7 if "tools:" in user else 0.3
        scores = [{"id": i, "p": p, "why": "test"} for i in ids]
        return ChatResult(json.dumps({"scores": scores}), 50, 10, 0.0)


def _client(rpd: int = 100) -> tuple[LLMClient, EchoProvider]:
    cfg = ModelsConfig(
        safety_margin=1.0,
        day_reset_utc_offset_hours=0,
        timeout_s=5,
        cache_path=":memory:",
        usage_path=":memory:",
        providers={"p": ProviderConfig(base_url="http://p", api_key_env=None)},
        catalog={MODEL: ModelLimits(family="f", rpm=1000, rpd=rpd, tpm=10**7)},
        roles={"fraud": RoleConfig(chain=[MODEL], max_tokens=100)},
    )
    provider = EchoProvider()
    ledger = UsageLedger(":memory:", cfg.catalog, 1.0, 0, sleep=lambda s: None)
    return LLMClient(cfg, {"p": provider}, LLMCache(":memory:"), ledger), provider


@pytest.fixture(autouse=True)
def _few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 40)


def test_stratified_order_keeps_share_in_every_prefix() -> None:
    y = pd.Series([1] * 50 + [0] * 450, index=range(1000, 1500))
    order = fb.stratified_order(y, 200, 0.25, seed=0)
    assert len(order) == 200 and order.is_unique and set(order) <= set(y.index)
    for k in (20, 40, 100, 200):
        assert abs(y.loc[order[:k]].mean() - 0.25) <= 1 / k + 1e-9
    assert list(order) == list(fb.stratified_order(y, 200, 0.25, seed=0))  # reproducible
    assert list(order[:40]) != list(fb.stratified_order(y, 200, 0.25, seed=1)[:40])
    with pytest.raises(ValueError, match="not enough"):
        fb.stratified_order(y, 200, 0.5, seed=0)


def test_train_only_artifacts_and_signals(fraud_frame, legacy_spec, tmp_path: Path) -> None:
    frame = legacy_spec.prepare(fraud_frame)
    art = fb.load_or_train(frame, legacy_spec, 0, tmp_path / "m")
    assert art.dataset == legacy_spec.name and art.version.startswith("catboost-")
    again = fb.load_or_train(frame, legacy_spec, 0, tmp_path / "m")  # reused, not refit
    assert again.version == art.version
    _, val, _ = legacy_spec.split(frame)
    sig = fb.tool_signals(art, legacy_spec, fraud_frame.loc[val.index[:3]])
    assert len(sig) == 3 and all(0 <= s.model_score <= 1 for s in sig)


def test_prompts_estimate_and_run_arms() -> None:
    client, provider = _client()
    facts = [f"cause=c{i}" for i in range(25)]
    tools = [f"ml=0.{i}" for i in range(25)]
    prompts = {arm: fb.arm_prompts(arm, facts, tools, 10) for arm in ("llm", "hybrid")}
    assert len(prompts["llm"]) == 3 and "tools:" not in prompts["llm"][0]
    assert "tools:" in prompts["hybrid"][0]
    est = fb.estimate(client, MODEL, prompts)
    assert est["requests"] == 6 and est["tokens"] > 6 * 100

    sizes = [10, 10, 5]
    llm = fb.run_arm(client, "llm", MODEL, prompts["llm"], sizes)
    hyb = fb.run_arm(client, "hybrid", MODEL, prompts["hybrid"], sizes)
    assert llm.scores == [0.3] * 25 and hyb.scores == [0.7] * 25
    assert llm.requests == 3 and llm.stopped is None
    rerun = fb.run_arm(client, "llm", MODEL, prompts["llm"], sizes)
    assert rerun.requests == 0 and provider.calls == 6  # all from cache


def test_run_arm_stops_cleanly_when_budget_runs_out() -> None:
    client, _ = _client(rpd=2)
    prompts = fb.arm_prompts("llm", [f"f{i}" for i in range(30)], [""] * 30, 10)
    run = fb.run_arm(client, "llm", MODEL, prompts, [10, 10, 10])
    assert run.requests == 2 and run.stopped and "daily" in run.stopped
    assert np.isnan(run.scores[20:]).all() and not np.isnan(run.scores[:20]).any()


def test_summarize_pairs_and_drops_unscored() -> None:
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 150)
    good = y + rng.normal(0, 0.5, 200)
    bad = rng.random(200)
    partial = good.copy()
    partial[:5] = np.nan
    metrics, diffs = fb.summarize(
        y, {"ml": good, "llm": bad, "hybrid": partial}, 0.06, 0.05, 0, n_boot=200
    )
    assert metrics.loc["ml", "roc_auc"] > metrics.loc["llm", "roc_auc"]
    assert (metrics["n_scored"] == 195).all()
    assert list(diffs.index) == ["ml - llm", "ml - hybrid", "llm - hybrid"]
    assert diffs.loc["ml - llm", "delta_roc_auc"].startswith("+")


def test_benchmark_pool_and_rate(fraud_frame, legacy_spec) -> None:
    frame = legacy_spec.prepare(fraud_frame)
    _, val, test = legacy_spec.split(frame)
    assert fb.benchmark_pool(frame, legacy_spec, final=False).index.equals(val.index)
    assert fb.population_rate(frame, legacy_spec, final=True) == pytest.approx(test[Y].mean())


def test_instructions_cover_both_arms() -> None:
    assert set(fraud_llm.INSTRUCTIONS) == {"llm", "hybrid"}
