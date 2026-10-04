from pathlib import Path

import pandas as pd
import pytest

from autoclaim.ml import fraud_model, models, train


@pytest.fixture(autouse=True)
def _few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 60)


@pytest.fixture
def csv(tmp_path: Path, fraud_frame: pd.DataFrame) -> Path:
    path = tmp_path / "fraud_oracle.csv"
    fraud_frame.to_csv(path, index=False)
    return path


def test_main_trains_saves_and_reports(tmp_path: Path, csv: Path) -> None:
    report, model_dir = tmp_path / "bench.md", tmp_path / "models"
    argv = ["--csv", str(csv), "--report", str(report), "--model-dir", str(model_dir)]
    assert train.main([*argv, "--models", "logistic"]) == 0
    text = report.read_text(encoding="utf-8")
    for heading in (
        "## Model comparison",
        "## Production artifacts",
        "## Global feature importance",
        "## Red-flag rules on labeled data",
    ):
        assert heading in text
    assert "catboost + Sex/MaritalStatus" in text  # fairness ablation row
    assert fraud_model.load(model_dir).features


def test_skip_benchmark_omits_comparison(tmp_path: Path, csv: Path) -> None:
    report = tmp_path / "bench.md"
    argv = ["--csv", str(csv), "--report", str(report), "--model-dir", str(tmp_path / "m")]
    assert train.main([*argv, "--skip-benchmark"]) == 0
    assert "## Model comparison" not in report.read_text(encoding="utf-8")


def test_unknown_model_rejected(csv: Path) -> None:
    with pytest.raises(SystemExit):
        train.main(["--csv", str(csv), "--models", "random_forest"])


def test_feature_importance_top_n(trained_artifacts) -> None:
    imp = train.feature_importance(trained_artifacts, top=5)
    assert len(imp) == 5
    assert imp["importance"].is_monotonic_decreasing
    assert imp["importance"].sum() <= 100.01
