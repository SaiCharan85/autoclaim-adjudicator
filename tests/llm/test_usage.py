import pytest
from llm_fakes import FakeClock

from autoclaim.config import ModelLimits
from autoclaim.llm.types import BudgetExceededError
from autoclaim.llm.usage import UsageLedger, estimate_tokens


def _ledger(path=":memory:", clock=None, margin=1.0, **lim) -> UsageLedger:
    clock = clock or FakeClock()
    limits = {"m": ModelLimits(family="f", **({"rpm": 100, "rpd": 3, "tpd": 1000} | lim))}
    return UsageLedger(path, limits, margin, 0, clock=clock, sleep=clock.sleep)


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 101


def test_daily_request_cap_refuses() -> None:
    ledger = _ledger()
    for _ in range(3):
        ledger.reserve("m", 10)
        ledger.record("m", 10)
    with pytest.raises(BudgetExceededError, match="daily"):
        ledger.reserve("m", 10)


def test_daily_token_cap_and_margin() -> None:
    ledger = _ledger(margin=0.5)  # tpd 1000 -> 500 usable
    ledger.record("m", 450)
    with pytest.raises(BudgetExceededError):
        ledger.reserve("m", 100)


def test_counter_persists_and_resets_next_day(tmp_path) -> None:
    clock = FakeClock()
    path = tmp_path / "u.sqlite3"
    _ledger(path, clock).record("m", 10)
    assert _ledger(path, clock).today("m") == (1, 10)
    clock.t += 86_400
    assert _ledger(path, clock).today("m") == (0, 0)


def test_minute_window_waits_instead_of_failing() -> None:
    clock = FakeClock()
    ledger = _ledger(clock=clock, rpm=2, rpd=100, tpd=10**6)
    for _ in range(2):
        ledger.reserve("m", 1)
        ledger.record("m", 1)
    ledger.reserve("m", 1)  # third call in the same minute must wait ~60 s
    assert clock.slept and sum(clock.slept) >= 59


def test_single_call_over_tpm_refused() -> None:
    with pytest.raises(BudgetExceededError, match="TPM"):
        _ledger(tpm=50).reserve("m", 51)


def test_unlimited_model() -> None:
    clock = FakeClock()
    ledger = UsageLedger(":memory:", {"m": ModelLimits(family="f")}, 0.9, 0, clock, clock.sleep)
    for _ in range(50):
        ledger.reserve("m", 10**6)
        ledger.record("m", 10**6)
    assert clock.slept == []
