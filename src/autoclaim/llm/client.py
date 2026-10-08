"""Role-based structured LLM calls with fallback, exact-match cache and free-tier budgets.

For each model in the role's chain (config `models.roles`):
cache hit -> parse; else reserve budget (refuse = next model) -> call -> record usage -> cache ->
parse with Pydantic; on a validation error, retry once with the error appended.
Provider errors and budget refusals move on to the next model. When the chain is exhausted
LLMUnavailableError is raised, and the harness routes the claim to a human (fail safe).
"""

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from autoclaim.config import ModelsConfig
from autoclaim.llm.cache import LLMCache, cache_key
from autoclaim.llm.providers import ChatProvider, OpenAICompatProvider
from autoclaim.llm.types import (
    BudgetExceededError,
    ChatRequest,
    ChatResult,
    LLMUnavailableError,
    Message,
    ProviderError,
    RateLimitedError,
    TransientProviderError,
)
from autoclaim.llm.usage import UsageLedger, estimate_tokens
from autoclaim.paths import REPO_ROOT

T = TypeVar("T", bound=BaseModel)

MAX_RATE_LIMIT_WAIT_S = 30.0
# waits before retrying the same model after a 5xx or network error (Gemma 4 returns sporadic 500s)
TRANSIENT_RETRY_WAITS_S = (5.0, 15.0)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
# Some models (e.g. Gemma 4 on AI Studio) always think first, inline as <thought>...</thought>,
# which can itself contain JSON-like drafts: drop it before reading the answer.
_THOUGHT = re.compile(r"<thought>.*?</thought>", re.DOTALL)


@dataclass
class CallMeta:
    """Audit record for one structured call (cost is always $0: free tiers only)."""

    role: str
    model: str = ""
    family: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    validation_retries: int = 0
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Structured(Generic[T]):
    value: T
    meta: CallMeta


def compact_schema(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), separators=(",", ":"))


def system_prompt(instructions: str, schema: type[BaseModel]) -> str:
    """Static prefix (instructions + schema) first, so provider prefix caching can apply."""
    return f"{instructions}\nReply with one JSON object matching this JSON schema:\n" + (
        compact_schema(schema)
    )


def parse_json(text: str, schema: type[T]) -> T:
    """The answer object: thought blocks and code fences removed first."""
    return schema.model_validate_json(_FENCE.sub("", _THOUGHT.sub("", text).strip()))


class LLMClient:
    def __init__(
        self,
        cfg: ModelsConfig,
        providers: dict[str, ChatProvider],
        cache: LLMCache,
        ledger: UsageLedger,
    ) -> None:
        self.cfg = cfg
        self.providers = providers
        self.cache = cache
        self.ledger = ledger

    @classmethod
    def from_config(cls, cfg: ModelsConfig) -> "LLMClient":
        """Providers whose API key is missing are left out (their chain entries are skipped)."""
        providers: dict[str, ChatProvider] = {}
        for name, pc in cfg.providers.items():
            if pc.api_key_env and not os.environ.get(pc.api_key_env):
                continue
            providers[name] = OpenAICompatProvider.from_env(
                name, pc.base_url, pc.api_key_env, cfg.timeout_s
            )
        ledger = UsageLedger(
            REPO_ROOT / cfg.usage_path,
            cfg.catalog,
            cfg.safety_margin,
            cfg.day_reset_utc_offset_hours,
        )
        return cls(cfg, providers, LLMCache(REPO_ROOT / cfg.cache_path), ledger)

    def _request(self, role: str, key: str, messages: list[Message]) -> ChatRequest:
        rc = self.cfg.roles[role]
        effort = rc.reasoning_effort
        return ChatRequest(
            model=key.partition(":")[2],
            messages=tuple(messages),
            max_tokens=self._max_tokens(key, rc.max_tokens),
            reasoning_effort=effort if effort in self.cfg.catalog[key].efforts else None,
        )

    def _max_tokens(self, key: str, wanted: int) -> int:
        """The role's cap, lowered under the model's per-minute output limit (if it has one)."""
        limits = self.cfg.catalog[key]
        total = wanted + limits.thinking_tokens
        return (
            total if limits.otpm is None else min(total, int(limits.otpm * self.cfg.safety_margin))
        )

    def _chat_with_retries(
        self, provider: ChatProvider, key: str, request: ChatRequest, est: int
    ) -> ChatResult:
        """One short rate-limit wait, or up to two waits after a transient server error."""
        waits = list(TRANSIENT_RETRY_WAITS_S)
        while True:
            try:
                return provider.chat(request)
            except RateLimitedError as exc:
                wait = exc.retry_after_s
                if wait is None or wait > MAX_RATE_LIMIT_WAIT_S:
                    raise
                self.ledger.sleep(wait)
                self.ledger.reserve(key, est)
                return provider.chat(request)
            except TransientProviderError:
                if not waits:
                    raise
                self.ledger.sleep(waits.pop(0))
                self.ledger.reserve(key, est)

    def estimate_tokens(self, role: str, system: str, user: str) -> int:
        """Worst-case tokens for one call (prompt estimate + max completion), for dry runs."""
        return estimate_tokens(system + user) + self.cfg.roles[role].max_tokens

    def _call(self, key: str, request: ChatRequest, meta: CallMeta) -> ChatResult:
        ck = cache_key(key.partition(":")[0], request)
        hit = self.cache.get(ck)
        if hit is not None:
            meta.cached = True
            return hit
        provider = self.providers[key.partition(":")[0]]
        est = estimate_tokens("".join(m.content for m in request.messages)) + request.max_tokens
        self.ledger.reserve(key, est)
        result = self._chat_with_retries(provider, key, request, est)
        self.ledger.record(key, result.total_tokens or est)
        self.cache.put(ck, result)
        meta.cached = False
        return result

    def structured(
        self,
        role: str,
        instructions: str,
        user: str,
        schema: type[T],
        exclude_families: frozenset[str] = frozenset(),
        chain: Sequence[str] | None = None,
    ) -> Structured[T]:
        """`chain` overrides the role's fallback chain, e.g. one pinned model for a benchmark."""
        meta = CallMeta(role=role)
        system = system_prompt(instructions, schema)
        for key in chain if chain is not None else self.cfg.roles[role].chain:
            if key not in self.cfg.catalog:
                raise KeyError(f"{key!r} is not in the model catalog")
            family = self.cfg.catalog[key].family
            if family in exclude_families:
                meta.errors.append(f"{key}: skipped (family {family} excluded)")
                continue
            if key.partition(":")[0] not in self.providers:
                meta.errors.append(f"{key}: skipped (provider not configured)")
                continue
            messages = [Message(role="system", content=system), Message(role="user", content=user)]
            meta.model, meta.family = key, family
            for attempt in range(2):
                try:
                    result = self._call(key, self._request(role, key, messages), meta)
                except (BudgetExceededError, ProviderError) as exc:
                    meta.errors.append(f"{key}: {exc}")
                    break
                meta.prompt_tokens += result.prompt_tokens
                meta.completion_tokens += result.completion_tokens
                meta.latency_s += result.latency_s
                try:
                    return Structured(parse_json(result.text, schema), meta)
                except ValidationError as exc:
                    meta.errors.append(f"{key}: invalid output (attempt {attempt + 1})")
                    meta.validation_retries += attempt == 0
                    messages += [
                        Message(role="assistant", content=result.text),
                        Message(
                            role="user",
                            content="That JSON failed validation:\n"
                            f"{exc.errors(include_url=False)}\nReply with the corrected JSON only.",
                        ),
                    ]
        raise LLMUnavailableError(f"role {role!r}: every model failed: {meta.errors}")
