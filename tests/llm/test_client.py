import pytest
from llm_fakes import FakeClock, ScriptedProvider, models_cfg
from pydantic import BaseModel

from autoclaim.llm.cache import LLMCache
from autoclaim.llm.client import LLMClient, parse_json, system_prompt
from autoclaim.llm.types import LLMUnavailableError, ProviderError, RateLimitedError
from autoclaim.llm.usage import UsageLedger


class Out(BaseModel):
    label: str
    score: float


GOOD = '{"label": "ok", "score": 0.5}'


def _client(a: list, b: list | None = None, **limits) -> tuple[LLMClient, dict]:
    cfg = models_cfg(**limits)
    clock = FakeClock()
    providers = {"a": ScriptedProvider("a", a)}
    if b is not None:
        providers["b"] = ScriptedProvider("b", b)
    ledger = UsageLedger(":memory:", cfg.catalog, 1.0, 0, clock=clock, sleep=clock.sleep)
    return LLMClient(cfg, providers, LLMCache(":memory:"), ledger), providers


def _call(client: LLMClient, user: str = "claim", **kw):
    return client.structured("adjudicator", "Decide.", user, Out, **kw)


def test_happy_path_meta_and_request() -> None:
    client, prov = _client([GOOD])
    res = _call(client)
    assert res.value == Out(label="ok", score=0.5)
    assert res.meta.model == "a:m1" and res.meta.family == "fam1"
    assert res.meta.prompt_tokens == 100 and res.meta.cost_usd == 0.0 and not res.meta.cached
    req = prov["a"].requests[0]
    assert req.model == "m1" and req.reasoning_effort == "low" and req.max_tokens == 50
    assert req.messages[0].role == "system" and '"label"' in req.messages[0].content


@pytest.mark.parametrize(("otpm", "sent"), [(40, 40), (100, 50)])
def test_max_tokens_stays_under_the_output_per_minute_limit(otpm, sent) -> None:
    client, prov = _client([GOOD], otpm=otpm)
    _call(client)
    assert prov["a"].requests[0].max_tokens == sent


@pytest.mark.parametrize(("otpm", "sent"), [(None, 80), (70, 70)])
def test_thinking_models_get_their_allowance_on_top_still_under_otpm(otpm, sent) -> None:
    client, prov = _client([GOOD], thinking_tokens=30, otpm=otpm)
    _call(client)
    assert prov["a"].requests[0].max_tokens == sent


def test_second_identical_call_is_free() -> None:
    client, prov = _client([GOOD])
    _call(client)
    again = _call(client)
    assert again.meta.cached and len(prov["a"].requests) == 1
    assert client.ledger.today("a:m1") == (1, 120)


def test_validation_retry_once_with_error() -> None:
    client, prov = _client(['{"label": "ok"}', GOOD])
    res = _call(client)
    assert res.value.score == 0.5 and res.meta.validation_retries == 1
    retry = prov["a"].requests[1].messages
    assert retry[-2].role == "assistant" and "failed validation" in retry[-1].content


def test_falls_back_after_two_invalid_outputs() -> None:
    client, prov = _client(["nope", "still nope"], [GOOD])
    res = _call(client)
    assert res.meta.model == "b:m2"
    assert prov["b"].requests[0].reasoning_effort is None  # m2 does not accept efforts


def test_provider_error_falls_back() -> None:
    client, _ = _client([ProviderError("down")], [GOOD])
    res = _call(client)
    assert res.meta.model == "b:m2" and "down" in res.meta.errors[0]


def test_budget_exhausted_falls_back_then_unavailable() -> None:
    client, _ = _client([GOOD, GOOD], [GOOD], rpd=1)
    assert _call(client, "c1").meta.model == "a:m1"
    assert _call(client, "c2").meta.model == "b:m2"  # a is out of daily budget
    with pytest.raises(LLMUnavailableError, match="daily safety limit"):
        _call(client, "c3")


def test_short_rate_limit_waits_and_retries_same_model() -> None:
    client, prov = _client([RateLimitedError("429", retry_after_s=5), GOOD])
    assert _call(client).meta.model == "a:m1"
    assert len(prov["a"].requests) == 2


def test_long_rate_limit_moves_on() -> None:
    client, _ = _client([RateLimitedError("429", retry_after_s=600)], [GOOD])
    assert _call(client).meta.model == "b:m2"


def test_excluded_family_and_missing_provider_are_skipped() -> None:
    client, _ = _client([GOOD])  # provider b not configured
    with pytest.raises(LLMUnavailableError) as exc:
        _call(client, exclude_families=frozenset({"fam1"}))
    assert "excluded" in str(exc.value) and "not configured" in str(exc.value)


def test_parse_json_strips_fences_and_system_prompt_has_schema() -> None:
    assert parse_json(f"```json\n{GOOD}\n```", Out).label == "ok"
    sp = system_prompt("Be brief.", Out)
    assert sp.startswith("Be brief.") and '"score"' in sp


def test_estimate_tokens_includes_max_completion() -> None:
    client, _ = _client([])
    assert client.estimate_tokens("adjudicator", "x" * 40, "") == 11 + 50


def test_pinned_chain_never_falls_back() -> None:
    client, prov = _client([ProviderError("down")], [GOOD])
    with pytest.raises(LLMUnavailableError):
        _call(client, chain=["a:m1"])
    assert prov["b"].requests == []
    with pytest.raises(KeyError):
        _call(client, chain=["a:unknown"])


def test_parse_json_drops_an_inline_thought_block_with_json_drafts() -> None:
    reply = '<thought>* draft: `{"label": "draft"}`\n* final below</thought>' + GOOD
    assert parse_json(reply, Out).label == "ok"
    fenced = "<thought>x</thought>\n```json\n" + GOOD + "\n```"
    assert parse_json(fenced, Out).label == "ok"
