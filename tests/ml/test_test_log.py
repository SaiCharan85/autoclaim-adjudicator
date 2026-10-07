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


def test_already_used_matches_dataset_and_headline_start(tmp_path) -> None:
    from autoclaim.ml import test_log

    log = tmp_path / "log.md"
    test_log.append("sim_us", "fraud LLM benchmark n=200: ml ROC-AUC 0.855", log, version="abc")
    assert test_log.already_used("sim_us", "fraud LLM benchmark n=200", log)
    assert not test_log.already_used("sim_us", "fraud LLM benchmark n=50", log)
    assert not test_log.already_used("legacy_1990s", "fraud LLM benchmark n=200", log)
    assert not test_log.already_used("sim_us", "x", tmp_path / "missing.md")


def test_guard_final_refuses_a_second_look_unless_asked(tmp_path) -> None:
    import pytest

    from autoclaim.ml import test_log

    log = tmp_path / "log.md"
    test_log.guard_final("flood", "auto_rate", rerun=False, path=log)  # first look: fine
    test_log.append("flood", "auto_rate 0.414", log, version="abc")
    with pytest.raises(SystemExit, match="second look"):
        test_log.guard_final("flood", "auto_rate", rerun=False, path=log)
    test_log.guard_final("flood", "auto_rate", rerun=True, path=log)  # explicit override


def test_the_real_log_blocks_the_fraud_benchmark_rerun() -> None:
    from autoclaim.ml import test_log

    assert test_log.already_used("sim_us", "fraud LLM benchmark n=200")
