from pathlib import Path

import pandas as pd
import pytest

from autoclaim.datasets import vehicle_fraud


def test_csv_path_uses_data_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOCLAIM_DATA_DIR", str(tmp_path))
    assert vehicle_fraud.csv_path() == tmp_path / "raw" / "vehicle_fraud" / "fraud_oracle.csv"


def test_load_raw_missing_file_gives_download_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"download_data\.py"):
        vehicle_fraud.load_raw(tmp_path / "nope.csv")


def test_load_validated_round_trip(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    path = tmp_path / "fraud_oracle.csv"
    valid_frame.to_csv(path, index=False)
    df, report = vehicle_fraud.load_validated(path)
    assert report.ok
    assert df.shape == valid_frame.shape


def test_load_validated_raises_on_schema_error(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    path = tmp_path / "fraud_oracle.csv"
    valid_frame.drop(columns=["FraudFound_P"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="FraudFound_P:missing_column"):
        vehicle_fraud.load_validated(path)
