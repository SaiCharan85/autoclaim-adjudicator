from pathlib import Path

import pandas as pd
import pytest

from autoclaim.config import FraudModelConfig
from autoclaim.ml import fraud_model, models


def test_metadata_records_dataset_and_exclusions(
    trained_artifacts, fraud_std: pd.DataFrame
) -> None:
    md = trained_artifacts.metadata
    assert md["dataset"] == "legacy_1990s"
    assert md["real_labels"] is True
    assert "Sex" not in trained_artifacts.features
    assert md["excluded_sensitive_features"] == ["Sex", "MaritalStatus"]
    assert md["n_fit"] + md["n_test"] == len(fraud_std)  # fit = train + validation
    assert {"catboost", "isolation_forest"} == set(md["test_metrics"])
    assert md["n_trees"] >= 1
    assert "Make" in trained_artifacts.categorical


def test_test_metrics_only_when_final(
    legacy_spec,
    fraud_std: pd.DataFrame,
    fraud_cfg: FraudModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 40)
    art = fraud_model.train_final(fraud_std, legacy_spec, fraud_cfg, seed=0, final=False)
    assert art.metadata["test_metrics"] is None  # locked test set never touched by accident


def test_save_load_round_trip(trained_artifacts, fraud_std: pd.DataFrame, tmp_path: Path) -> None:
    out = fraud_model.save(trained_artifacts, tmp_path / "fraud")
    assert trained_artifacts.version.startswith("catboost-")
    loaded = fraud_model.load(out)
    frame = fraud_std.head(25)
    pd.testing.assert_series_equal(
        loaded.predict_proba(frame), trained_artifacts.predict_proba(frame)
    )
    assert (loaded.anomaly.score(frame) == trained_artifacts.anomaly.score(frame)).all()
    assert (loaded.version, loaded.dataset) == (trained_artifacts.version, "legacy_1990s")
    assert loaded.features == trained_artifacts.features
    assert loaded.categorical == trained_artifacts.categorical


def test_load_missing_dir_gives_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"train_fraud\.py"):
        fraud_model.load(tmp_path / "nothing")


def test_default_dir_respects_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trained_artifacts
) -> None:
    monkeypatch.setenv("AUTOCLAIM_MODELS_DIR", str(tmp_path))
    assert fraud_model.save(trained_artifacts) == tmp_path / "fraud"
    assert fraud_model.load().version == trained_artifacts.version
