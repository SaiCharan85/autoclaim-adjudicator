"""Per-claim budget caps: a claim that needs too many LLM calls or tokens goes to a human."""

from pydantic import BaseModel, Field


class BudgetConfig(BaseModel):
    max_llm_calls: int = Field(gt=0)
    max_tokens: int = Field(gt=0)


class BudgetTrippedError(Exception):
    pass


def check_budget(llm_calls: int, tokens: int, cfg: BudgetConfig) -> None:
    if llm_calls > cfg.max_llm_calls:
        raise BudgetTrippedError(f"llm_calls {llm_calls} > {cfg.max_llm_calls}")
    if tokens > cfg.max_tokens:
        raise BudgetTrippedError(f"tokens {tokens} > {cfg.max_tokens}")
