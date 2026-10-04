"""Append-only log of every evaluation on a locked test set (docs/test_set_log.md).

Anyone can audit how often the test set was touched and with which code version. Running with
--final more than once is allowed but visible: that is the point.
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from autoclaim.paths import REPO_ROOT

DEFAULT_LOG = REPO_ROOT / "docs" / "test_set_log.md"
HEADER = (
    "# Locked test-set log\n\n"
    "Every evaluation on a locked test set is appended here automatically "
    "(`scripts/train_fraud.py --final`). Model and protocol decisions use validation data only.\n\n"
    "| when (UTC) | dataset | code | headline |\n|---|---|---|---|\n"
)


def code_version(repo: Path = REPO_ROOT) -> str:
    """Short commit hash, '+dirty' if the working tree has uncommitted changes (read-only git)."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}+dirty" if dirty else head


def append(
    dataset: str, headline: str, path: Path | None = None, version: str | None = None
) -> None:
    path = path or DEFAULT_LOG  # resolved at call time, so tests can redirect it
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER, encoding="utf-8")
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    row = f"| {when} | {dataset} | {version or code_version()} | {headline} |\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(row)
