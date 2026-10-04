"""Step 0 guardrails: package layout, config, secrets hygiene, and git safeguards."""

import importlib
import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

SUBPACKAGES = [
    "autoclaim.core",
    "autoclaim.lines",
    "autoclaim.lines.auto",
    "autoclaim.ml",
    "autoclaim.retrieval",
    "autoclaim.llm",
    "autoclaim.ui",
]


def test_package_version_matches_pyproject(repo_root: Path) -> None:
    import autoclaim

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert autoclaim.__version__ == pyproject["project"]["version"]


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_subpackages_import(module: str) -> None:
    importlib.import_module(module)


# ---------------------------------------------------------------- carrier config


@pytest.fixture(scope="module")
def carrier_config(repo_root: Path) -> dict:
    path = repo_root / "config" / "carrier_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_carrier_config_has_required_sections(carrier_config: dict) -> None:
    sections = ("carrier", "retrieval", "harness", "deductibles", "router", "budgets", "models")
    for section in sections:
        assert section in carrier_config, f"missing section: {section}"


def test_hnsw_params_are_positive_ints(carrier_config: dict) -> None:
    hnsw = carrier_config["retrieval"]["hnsw"]
    for key in ("M", "ef_construction", "ef_search"):
        assert isinstance(hnsw[key], int) and hnsw[key] > 0
    # efConstruction below M makes a poorly connected graph.
    assert hnsw["ef_construction"] >= hnsw["M"]


def test_harness_retry_and_seed(carrier_config: dict) -> None:
    harness = carrier_config["harness"]
    assert harness["max_retries"] == 2
    assert isinstance(harness["seed"], int)


def test_models_cover_every_role(carrier_config: dict) -> None:
    roles = {"intake", "coverage", "fraud", "adjudicator", "judge", "synthetic_generator"}
    assert roles <= set(carrier_config["models"])


# ---------------------------------------------------------------- secrets hygiene


def _env_example(repo_root: Path) -> dict[str, str]:
    pairs = {}
    for line in (repo_root / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key] = value
    return pairs


def test_env_example_lists_expected_variables(repo_root: Path) -> None:
    keys = set(_env_example(repo_root))
    assert {"KAGGLE_USERNAME", "KAGGLE_KEY"} <= keys


def test_env_example_secret_values_are_blank(repo_root: Path) -> None:
    secret_like = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)$")
    for key, value in _env_example(repo_root).items():
        if secret_like.search(key):
            assert value == "", f"{key} must be blank in .env.example"


@pytest.mark.parametrize("pattern", ["data/", "models/", ".env", ".cache/", "*.faiss"])
def test_gitignore_covers_sensitive_paths(repo_root: Path, pattern: str) -> None:
    lines = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert pattern in lines


def test_env_example_is_not_ignored(repo_root: Path) -> None:
    lines = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "!.env.example" in lines


# ---------------------------------------------------------------- git safeguards


def test_claude_settings_disable_attribution(repo_root: Path) -> None:
    settings = json.loads((repo_root / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["includeCoAuthoredBy"] is False
    assert settings["attribution"] == {"commit": "", "pr": ""}


@pytest.mark.parametrize("command", ["commit", "push", "rebase", "reset"])
def test_claude_settings_deny_history_rewriting_git(repo_root: Path, command: str) -> None:
    settings = json.loads((repo_root / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert f"Bash(git {command}:*)" in settings["permissions"]["deny"]
