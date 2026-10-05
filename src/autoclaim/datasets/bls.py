"""BLS Consumer Price Index series (public domain) -> annual averages, for era-correct amounts.

Uses the public API v1 (no key; at most 10 years per request and a small daily request quota), so
years are fetched in 10-year windows: 3 requests for 2002-2025.
"""

import json
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

API = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
SERIES = {
    "CUUR0000SETD": "repair",  # motor vehicle maintenance and repair
    "CUUR0000SETA01": "new_vehicles",
    "CUUR0000SETA02": "used_vehicles",
}
LICENSE = "Public domain (U.S. Bureau of Labor Statistics); citation requested"

Poster = Callable[[dict[str, object]], dict[str, object]]


def post(payload: dict[str, object]) -> dict[str, object]:
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "autoclaim-adjudicator/0.1 (research; free tier)",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out: dict[str, object] = json.loads(resp.read())
    return out


def annual_averages(
    start: int, end: int, series: Sequence[str] = tuple(SERIES), poster: Poster = post
) -> pd.DataFrame:
    """One row per year, one column per series (mean of the monthly values M01-M12)."""
    rows = []
    for lo in range(start, end + 1, 10):
        hi = min(lo + 9, end)
        res = poster({"seriesid": list(series), "startyear": str(lo), "endyear": str(hi)})
        if res.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS API: {res.get('status')} {res.get('message')}")
        for s in res["Results"]["series"]:  # type: ignore[index]
            for d in s["data"]:
                monthly = d["period"].startswith("M") and d["period"] != "M13"
                try:
                    value = float(d["value"])
                except ValueError:  # '-' = not published that month (e.g. a federal data gap)
                    continue
                if monthly:
                    rows.append({"series": s["seriesID"], "year": int(d["year"]), "value": value})
    frame = pd.DataFrame(rows)
    stats = frame.groupby(["year", "series"])["value"].agg(["mean", "count"])
    means: pd.DataFrame = stats["mean"].unstack("series").rename(columns=SERIES)
    counts: pd.DataFrame = stats["count"].unstack("series")
    means["months_published"] = counts.to_numpy().min(axis=1)
    return means.sort_index()


def download_cpi(dest: Path, start: int = 2002, end: int = 2025, poster: Poster = post) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    table = annual_averages(start, end, poster=poster).round(3)
    path = dest / "cpi_annual.csv"
    table.to_csv(path)
    manifest = {"source_key": "bls_cpi", "url": API, "series": SERIES, "licenses": [LICENSE],
                "years": [int(table.index.min()), int(table.index.max())],
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_cpi(dest: Path) -> pd.DataFrame:
    return pd.read_csv(dest / "cpi_annual.csv", index_col="year")
