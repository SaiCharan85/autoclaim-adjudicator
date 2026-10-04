from pathlib import Path

from autoclaim.ml import test_log


def test_append_creates_header_then_rows(tmp_path: Path) -> None:
    path = tmp_path / "log.md"
    test_log.append("sim_us", "PR-AUC 0.3", path=path, version="abc123")
    test_log.append("sim_us", "PR-AUC 0.31", path=path, version="abc124+dirty")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Locked test-set log"
    rows = [line for line in lines if line.startswith("| 20")]
    assert len(rows) == 2
    assert "abc124+dirty" in rows[1]


def test_default_path_resolved_at_call_time(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(test_log, "DEFAULT_LOG", tmp_path / "x.md")
    test_log.append("d", "h", version="v")
    assert (tmp_path / "x.md").exists()


def test_code_version_reads_git() -> None:
    version = test_log.code_version()
    assert version == "unknown" or len(version.split("+")[0]) >= 7


def test_code_version_outside_a_repo(tmp_path: Path) -> None:
    assert test_log.code_version(tmp_path) == "unknown"
