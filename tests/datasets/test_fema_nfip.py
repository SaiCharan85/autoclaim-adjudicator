import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from autoclaim.datasets import fema_nfip


def fake_api(rows: list[dict], page_seen: list[int]):
    def get(url: str) -> dict:
        q = parse_qs(urlparse(url).query)
        skip, top = int(q["$skip"][0]), int(q["$top"][0])
        page_seen.append(skip)
        return {"metadata": {"count": len(rows)}, "FimaNfipClaims": rows[skip : skip + top]}

    return get


def test_page_url_selects_columns_filters_years_and_orders_stably() -> None:
    q = parse_qs(urlparse(fema_nfip.page_url((2022, 2025), 20, 10)).query)
    assert q["$filter"] == ["yearOfLoss ge 2022 and yearOfLoss le 2025"]
    assert q["$orderby"] == ["id"] and q["$skip"] == ["20"] and q["$top"] == ["10"]
    cols = set(q["$select"][0].split(","))
    assert "amountPaidOnBuildingClaim" in cols and not cols & fema_nfip.FORBIDDEN


def test_fetch_pages_until_the_count() -> None:
    rows = [{"id": str(i), "dateOfLoss": "2023-01-01"} for i in range(25)]
    seen: list[int] = []
    frame = fema_nfip.fetch((2022, 2025), fake_api(rows, seen), page=10)
    assert len(frame) == 25 and seen == [0, 10, 20]


def test_fetch_refuses_fine_grained_location() -> None:
    rows = [{"id": "1", "latitude": 35.5}]
    with pytest.raises(ValueError, match="location"):
        fema_nfip.fetch((2022, 2025), fake_api(rows, []), page=10)


def test_download_writes_csv_and_manifest_with_disclaimer(tmp_path: Path) -> None:
    rows = [{"id": "1", "dateOfLoss": "2023-09-23T00:00:00.000Z", "buildingDeductibleCode": "2",
             "contentsDeductibleCode": "2", "causeOfDamage": "1",
             "nonPaymentReasonBuilding": None, "nonPaymentReasonContents": "97"}]  # fmt: skip
    fema_nfip.download(tmp_path, (2023, 2023), fake_api(rows, []))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rows"] == 1 and manifest["disclaimer"] == fema_nfip.DISCLAIMER
    frame = fema_nfip.load(tmp_path)
    assert frame.loc[0, "buildingDeductibleCode"] == "2"  # codes stay text ("A", "02", ...)
    assert frame.loc[0, "nonPaymentReasonContents"] == "97"
    assert str(frame["dateOfLoss"].dtype).startswith("datetime64")


def test_disclaimer_is_fema_wording() -> None:
    assert fema_nfip.DISCLAIMER.startswith("This product uses the Federal Emergency Management")
