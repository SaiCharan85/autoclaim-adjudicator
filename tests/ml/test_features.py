import numpy as np
import pandas as pd
import pytest

from autoclaim.ml import features as feats


def _dates(**cols: list) -> pd.DataFrame:
    return pd.DataFrame(cols)


@pytest.mark.parametrize(
    ("month", "week", "m_claim", "w_claim", "expected"),
    [
        ("Jan", 1, "Jan", 1, 0.0),
        ("Jan", 1, "Mar", 2, 9.0),  # 2 months * 4 + 1 week
        ("Dec", 4, "Jan", 1, 1.0),  # year wrap: 1 month * 4 - 3 weeks
        ("Jan", 3, "Jan", 1, 0.0),  # claimed "earlier" in the same month: clipped to 0
    ],
)
def test_claim_lag_weeks(
    month: str, week: int, m_claim: str, w_claim: int, expected: float
) -> None:
    df = _dates(
        Month=[month], WeekOfMonth=[week], MonthClaimed=[m_claim], WeekOfMonthClaimed=[w_claim]
    )
    assert feats.claim_lag_weeks(df).iat[0] == expected


def test_claim_lag_unknown_claim_month_is_nan() -> None:
    df = _dates(Month=["Jan"], WeekOfMonth=[1], MonthClaimed=[np.nan], WeekOfMonthClaimed=[1])
    assert np.isnan(feats.claim_lag_weeks(df).iat[0])


def test_time_order_is_monotone_across_years() -> None:
    df = pd.DataFrame(
        {"Year": [1994, 1995, 1995], "Month": ["Dec", "Jan", "Jan"], "WeekOfMonth": [5, 1, 2]}
    )
    order = feats.time_order(df).tolist()
    assert order == sorted(order)
    assert len(set(order)) == 3


def test_clean_converts_sentinels_and_adds_lag(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "Age"] = 0
    valid_frame.loc[1, "MonthClaimed"] = "0"
    valid_frame.loc[1, "DayOfWeekClaimed"] = "0"
    out = feats.clean(valid_frame)
    assert np.isnan(out.loc[0, "Age"])
    assert pd.isna(out.loc[1, "MonthClaimed"])
    assert pd.isna(out.loc[1, "DayOfWeekClaimed"])
    assert np.isnan(out.loc[1, "claim_lag_weeks"])
    assert out["Age"].notna().sum() == len(out) - 1
    assert "claim_lag_weeks" in out


def test_clean_does_not_mutate_input(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "Age"] = 0
    before = valid_frame.copy()
    feats.clean(valid_frame)
    pd.testing.assert_frame_equal(valid_frame, before)


def test_feature_columns_drop_ids_target_and_exclusions(valid_frame: pd.DataFrame) -> None:
    cols = feats.feature_columns(feats.clean(valid_frame), exclude=["Sex"])
    for dropped in (*feats.NON_FEATURES, feats.TARGET, "Sex"):
        assert dropped not in cols
    assert "claim_lag_weeks" in cols
    assert "MaritalStatus" in cols
