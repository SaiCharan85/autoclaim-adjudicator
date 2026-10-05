from autoclaim.llm.cache import LLMCache, cache_key
from autoclaim.llm.types import ChatRequest, ChatResult, Message


def _req(text: str = "hi", **kw) -> ChatRequest:
    return ChatRequest(
        model="m", messages=(Message(role="user", content=text),), max_tokens=10, **kw
    )


def test_key_is_exact_match_over_everything() -> None:
    base = cache_key("groq", _req())
    assert base == cache_key("groq", _req())
    assert base != cache_key("google", _req())
    assert base != cache_key("groq", _req("hi "))  # near-identical text is a different key
    assert base != cache_key("groq", _req(temperature=0.5))
    assert base != cache_key("groq", _req(reasoning_effort="low"))


def test_roundtrip_and_persistence(tmp_path) -> None:
    path = tmp_path / "c.sqlite3"
    cache = LLMCache(path)
    assert cache.get("k") is None
    cache.put("k", ChatResult(text="x", prompt_tokens=3, completion_tokens=2, latency_s=1.0))
    again = LLMCache(path).get("k")
    assert again is not None and again.text == "x" and again.total_tokens == 5
    assert again.latency_s == 0  # a cache hit costs no time
    assert len(LLMCache(path)) == 1
