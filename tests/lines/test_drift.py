import pandas as pd

from autoclaim.lines.auto import drift
from autoclaim.ml.frame import T


def test_era_rows_bounds() -> None:
    days = [(pd.Timestamp(d) - drift.EPOCH).days for d in ("2009-12-31", "2010-01-01", "2015-12-31",
                                                           "2016-01-01")]  # fmt: skip
    frame = pd.DataFrame({T: days})
    assert list(drift.era_rows(frame, (2010, 2015))[T]) == days[1:3]


def test_train_windows_precede_tested_eras() -> None:
    later = {
        name: [e for e, years in drift.ERAS.items() if years[0] > window[1]]
        for name, window in drift.TRAIN_WINDOWS.items()
    }
    assert later["2002-2009"] == ["2010-2015", "2016-2019", "2020-2023"]
    assert later["2002-2019 (all past)"] == ["2020-2023"]
    assert all(years[1] < 2024 for years in drift.ERAS.values())  # never the locked test window
