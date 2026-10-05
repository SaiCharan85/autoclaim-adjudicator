"""Fakes for LLM tests: scripted providers, a fake clock, a tiny model config."""

from autoclaim.config import ModelLimits, ModelsConfig, ProviderConfig, RoleConfig
from autoclaim.llm.types import ChatRequest, ChatResult, LLMError


class ScriptedProvider:
    """Returns (or raises) the scripted items in order and records every request."""

    def __init__(self, name: str, script: list[str | LLMError]) -> None:
        self.name = name
        self.script = list(script)
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, LLMError):
            raise item
        return ChatResult(text=item, prompt_tokens=100, completion_tokens=20, latency_s=0.01)


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def models_cfg(**limits: int) -> ModelsConfig:
    lim = {"rpm": 30, "rpd": 100, "tpm": 100_000, "tpd": 1_000_000} | limits
    return ModelsConfig(
        safety_margin=1.0,
        day_reset_utc_offset_hours=0,
        timeout_s=5,
        cache_path=":memory:",
        usage_path=":memory:",
        providers={
            "a": ProviderConfig(base_url="http://a", api_key_env=None),
            "b": ProviderConfig(base_url="http://b", api_key_env=None),
        },
        catalog={
            "a:m1": ModelLimits(family="fam1", efforts=["low"], **lim),
            "b:m2": ModelLimits(family="fam2", **lim),
        },
        roles={
            "adjudicator": RoleConfig(
                chain=["a:m1", "b:m2"], max_tokens=50, reasoning_effort="low"
            ),
        },
    )
