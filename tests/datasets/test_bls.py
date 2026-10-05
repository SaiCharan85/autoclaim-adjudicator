import pytest

from autoclaim.datasets import bls


def _poster(calls):
    def post(payload):
        calls.append(payload)
        lo, hi = int(payload["startyear"]), int(payload["endyear"])
        data = []
        for y in range(lo, hi + 1):
            for m in range(1, 13):
                value = "-" if (y == 2025 and m == 10) else str(100 + y - 2000 + m / 100)
                data.append({"year": str(y), "period": f"M{m:02d}", "value": value})
            data.append({"year": str(y), "period": "M13", "value": "999"})  # annual avg row
        series = [{"seriesID": sid, "data": data} for sid in payload["seriesid"]]
        return {"status": "REQUEST_SUCCEEDED", "Results": {"series": series}}

    return post


def test_annual_averages_windows_and_gaps(tmp_path) -> None:
    calls: list = []
    path = bls.download_cpi(tmp_path, 2002, 2025, poster=_poster(calls))
    assert [c["startyear"] for c in calls] == ["2002", "2012", "2022"]  # 10-year API windows
    table = bls.load_cpi(tmp_path)
    assert list(table.index) == list(range(2002, 2026))
    assert set(table.columns) == {"repair", "new_vehicles", "used_vehicles", "months_published"}
    assert table.loc[2002, "repair"] == pytest.approx(102 + 0.065)  # mean of M01..M12, no M13
    assert table.loc[2025, "months_published"] == 11 and path.exists()
    assert (tmp_path / "manifest.json").exists()


def test_api_failure_raises() -> None:
    with pytest.raises(RuntimeError, match="BLS API"):
        bls.annual_averages(2002, 2003, poster=lambda p: {"status": "REQUEST_NOT_PROCESSED"})
