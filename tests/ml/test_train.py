from dataclasses import replace
from pathlib import Path

import pytest

from autoclaim.ml import fraud_model, models, test_log, train
from fakes import make_fraud_frame


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch, legacy_spec, tmp_path: Path) -> None:
    monkeypatch.setattr(models, "MAX_TREES", 60)
    synthetic = replace(legacy_spec, load_raw=make_fraud_frame)  # no real data needed
    monkeypatch.setattr(train, "get_spec", lambda name, cfg: synthetic)
    monkeypatch.setattr(test_log, "DEFAULT_LOG", tmp_path / "test_set_log.md")


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--dataset", "legacy_1990s", "--report", str(tmp_path / "bench.md"),
        "--model-dir", str(tmp_path / "m"), *extra,
    ]  # fmt: skip


def test_validation_run_writes_report_without_touching_test(tmp_path: Path) -> None:
    assert train.main(_args(tmp_path, "--models", "logistic")) == 0
    text = (tmp_path / "bench.md").read_text(encoding="utf-8")
    for heading in (
        "## Leakage canaries",
        "## Model comparison: validation",
        "## Production artifacts",
        "## Global feature importance",
        "## Red-flag rules",
    ):
        assert heading in text
    assert "## Locked test set" not in text
    assert "not saved: reality-check dataset" in text  # legacy artifacts are never saved
    assert "catboost + Sex/MaritalStatus" in text  # fairness ablation row
    assert not (tmp_path / "test_set_log.md").exists()


def test_final_run_scores_test_and_logs_it(tmp_path: Path) -> None:
    assert train.main(_args(tmp_path, "--models", "logistic", "--final")) == 0
    assert "## Locked test set" in (tmp_path / "bench.md").read_text(encoding="utf-8")
    log = (tmp_path / "test_set_log.md").read_text(encoding="utf-8")
    assert "legacy_1990s" in log
    assert "test PR-AUC" in log


def test_skip_benchmark(tmp_path: Path) -> None:
    assert train.main(_args(tmp_path, "--skip-benchmark")) == 0
    assert "## Model comparison" not in (tmp_path / "bench.md").read_text(encoding="utf-8")


def test_non_production_dataset_is_not_saved(tmp_path: Path) -> None:
    assert train.main(_args(tmp_path, "--skip-benchmark")) == 0  # production is sim_us
    assert not (tmp_path / "m").exists()


def test_unknown_model_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        train.main(_args(tmp_path, "--models", "random_forest"))


def test_feature_importance_top_n(trained_artifacts) -> None:
    imp = train.feature_importance(trained_artifacts, top=5)
    assert len(imp) == 5
    assert imp["importance"].is_monotonic_decreasing
    assert imp["importance"].sum() <= 100.01


def test_saved_artifact_loads(trained_artifacts, tmp_path: Path) -> None:
    out = fraud_model.save(trained_artifacts, tmp_path)
    assert fraud_model.load(out).dataset == "legacy_1990s"
