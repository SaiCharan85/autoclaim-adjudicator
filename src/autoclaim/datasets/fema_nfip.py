"""FEMA NFIP redacted flood claims (OpenFEMA `FimaNfipClaims` v2) -> data/raw/fema_nfip/.

Real, recent US flood claims with what was actually paid and why a claim was not paid. Used by the
flood line (Step 8.5) to test deterministic flood coverage math against real outcomes.

Terms (https://www.fema.gov/about/openfema/terms-conditions, checked 2026-10-05): the disclaimer
below must accompany any product using the data; no re-identification; the data may not be used to
make determinations affecting anyone's rights or benefits. So: only state-level location is
fetched (no coordinates, census tract/block group, ZIP or county), and the data only replays
historical, already-closed claims for evaluation.
"""

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

API = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
DISCLAIMER = (
    "This product uses the Federal Emergency Management Agency's OpenFEMA API, but is not endorsed "
    "by FEMA. The Federal Government or FEMA cannot vouch for the data or analyses derived from "
    "these data after the data have been retrieved from the Agency's website(s)."
)
LICENSE = "OpenFEMA terms (U.S. government data): cite FEMA + disclaimer; no re-identification"
YEARS = (2022, 2025)
PAGE = 10_000
# Coverage-relevant fields only. Fine-grained location fields are deliberately NOT fetched.
COLUMNS = (
    "id", "dateOfLoss", "yearOfLoss", "state", "ratedFloodZone", "occupancyType",
    "primaryResidenceIndicator", "causeOfDamage", "floodEvent", "originalNBDate",
    "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
    "buildingDamageAmount", "contentsDamageAmount", "buildingDeductibleCode",
    "contentsDeductibleCode", "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
    "netBuildingPaymentAmount", "netContentsPaymentAmount", "nonPaymentReasonBuilding",
    "nonPaymentReasonContents", "buildingPropertyValue", "buildingReplacementCost",
    "replacementCostBasis", "waterDepth", "elevatedBuildingIndicator",
    "basementEnclosureCrawlspaceType", "numberOfFloorsInTheInsuredBuilding",
    "postFIRMConstructionIndicator",
)  # fmt: skip
FORBIDDEN = frozenset({"latitude", "longitude", "censusTract", "censusBlockGroupFips",
                       "reportedZipCode", "countyCode", "reportedCity"})  # fmt: skip

Getter = Callable[[str], dict[str, Any]]


def get(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "autoclaim-adjudicator/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out: dict[str, Any] = json.loads(resp.read())
    return out


def page_url(years: tuple[int, int], skip: int, top: int = PAGE) -> str:
    params = {
        "$filter": f"yearOfLoss ge {years[0]} and yearOfLoss le {years[1]}",
        "$select": ",".join(COLUMNS),
        "$orderby": "id",  # stable order, so paging never skips or repeats a row
        "$top": str(top),
        "$skip": str(skip),
        "$inlinecount": "allpages",
    }
    return f"{API}?{urllib.parse.urlencode(params, safe='$,')}"


def fetch(years: tuple[int, int] = YEARS, getter: Getter = get, page: int = PAGE
          ) -> pd.DataFrame:  # fmt: skip
    rows: list[dict[str, Any]] = []
    total: int | None = None
    skip = 0
    while total is None or skip < total:
        data = getter(page_url(years, skip, page))
        total = int(data["metadata"]["count"])
        batch = data["FimaNfipClaims"]
        if not batch:
            break
        rows += batch
        skip += len(batch)
    frame = pd.DataFrame(rows)
    leaked = FORBIDDEN & set(frame.columns)
    if leaked:
        raise ValueError(f"fine-grained location fields must not be stored: {sorted(leaked)}")
    return frame


def download(dest: Path, years: tuple[int, int] = YEARS, getter: Getter = get) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    frame = fetch(years, getter)
    path = dest / "claims.csv"
    frame.to_csv(path, index=False)
    manifest = {"source_key": "fema_nfip", "url": API, "licenses": [LICENSE],
                "disclaimer": DISCLAIMER, "years": list(years), "rows": len(frame),
                "columns": list(frame.columns),
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load(dest: Path, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(dest / "claims.csv", usecols=columns, low_memory=False,
                        dtype={"buildingDeductibleCode": str, "contentsDeductibleCode": str,
                               "causeOfDamage": str, "nonPaymentReasonBuilding": str,
                               "nonPaymentReasonContents": str})  # fmt: skip
    frame["dateOfLoss"] = pd.to_datetime(frame["dateOfLoss"], utc=True).dt.tz_localize(None)
    return frame
