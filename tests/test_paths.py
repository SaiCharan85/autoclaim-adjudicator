from pathlib import Path

import pytest

from autoclaim import paths


def test_repo_root_contains_pyproject() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").exists()


def test_data_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOCLAIM_DATA_DIR", raising=False)
    assert paths.data_dir() == paths.REPO_ROOT / "data"


def test_data_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOCLAIM_DATA_DIR", str(tmp_path))
    assert paths.raw_dir("x") == tmp_path / "raw" / "x"


def test_models_dir_default_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AUTOCLAIM_MODELS_DIR", raising=False)
    assert paths.models_dir() == paths.REPO_ROOT / "models"
    monkeypatch.setenv("AUTOCLAIM_MODELS_DIR", str(tmp_path))
    assert paths.models_dir() == tmp_path
