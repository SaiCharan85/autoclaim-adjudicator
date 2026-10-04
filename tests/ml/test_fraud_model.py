from pathlib import Path

import pandas as pd
import pytest

from autoclaim.ml import fraud_model
from autoclaim.ml.features import clean


def test_metadata_records_split_and_exclusions(
    trained_artifacts, fraud_frame: pd.DataFrame
) -> None:
    md = trained_artifacts.metadata
    assert md["train_years"] == [1994, 1995]
    assert md["n_train"] == int(fraud_frame["Year"].isin([1994, 1995]).sum())
    assert md["n_test"] == int((fraud_frame["Year"] == 1996).sum())
    assert "Sex" not in trained_artifacts.features
    assert "MaritalStatus" not in trained_artifacts.features
    assert md["excluded_sensitive_features"] == ["Sex", "MaritalStatus"]
    assert {"catboost", "isolation_forest"} == set(md["test_metrics"])
    assert md["n_trees"] >= 1


def test_save_load_round_trip(trained_artifacts, fraud_frame: pd.DataFrame, tmp_path: Path) -> None:
    out = fraud_model.save(trained_artifacts, tmp_path / "fraud")
    assert trained_artifacts.version.startswith("catboost-")
    loaded = fraud_model.load(out)
    frame = clean(fraud_frame).head(25)
    pd.testing.assert_series_equal(
        loaded.predict_proba(frame), trained_artifacts.predict_proba(frame)
    )
    assert (loaded.anomaly.score(frame) == trained_artifacts.anomaly.score(frame)).all()
    assert loaded.version == trained_artifacts.version
    assert loaded.features == trained_artifacts.features


def test_load_missing_dir_gives_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"train_fraud.py"):
        fraud_model.load(tmp_path / "nothing")


def test_default_dir_respects_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trained_artifacts
) -> None:
    monkeypatch.setenv("AUTOCLAIM_MODELS_DIR", str(tmp_path))
    out = fraud_model.save(trained_artifacts)
    assert out == tmp_path / "fraud"
    assert fraud_model.load().version == trained_artifacts.version
