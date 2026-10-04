"""Checks against the real downloaded file. Skipped when data/ is absent (e.g. in CI)."""

import pytest

from autoclaim.datasets import vehicle_fraud
from autoclaim.datasets.download import manifest_is_valid, read_manifest
from autoclaim.datasets.sources import VEHICLE_FRAUD

CSV = vehicle_fraud.csv_path()
pytestmark = [
    pytest.mark.requires_data,
    pytest.mark.skipif(not CSV.exists(), reason="run scripts/download_data.py first"),
]


def test_manifest_hash_and_license() -> None:
    manifest = read_manifest(CSV.parent)
    assert manifest is not None
    assert manifest_is_valid(manifest, VEHICLE_FRAUD, CSV.parent)
    assert manifest.licenses == ["CC0-1.0"]


def test_real_file_matches_schema_with_only_known_warnings() -> None:
    df, report = vehicle_fraud.load_validated()
    assert df.shape == (15_420, 33)
    found = {(w.column, w.check, w.n_rows) for w in report.warnings}
    assert found == {
        ("Age", "sentinel", 320),
        ("DayOfWeekClaimed", "sentinel", 1),
        ("MonthClaimed", "sentinel", 1),
    }


def test_known_data_facts_documented_in_data_notes() -> None:
    df = vehicle_fraud.load_raw()
    assert df["FraudFound_P"].sum() == 923
    # Age == 0 always co-occurs with the "16 to 17" policyholder band
    assert set(df.loc[df["Age"] == 0, "AgeOfPolicyHolder"]) == {"16 to 17"}
    # PolicyType "Sedan - Liability" is always tagged VehicleCategory "Sport" (source inconsistency)
    assert set(df.loc[df["PolicyType"] == "Sedan - Liability", "VehicleCategory"]) == {"Sport"}
