"""Filesystem locations. Data lives outside git; override with AUTOCLAIM_DATA_DIR."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("AUTOCLAIM_DATA_DIR", REPO_ROOT / "data"))


def raw_dir(source_key: str) -> Path:
    return data_dir() / "raw" / source_key


def models_dir() -> Path:
    return Path(os.environ.get("AUTOCLAIM_MODELS_DIR", REPO_ROOT / "models"))
