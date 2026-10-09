"""Copy the Lucide icons the UI uses from the official GitHub release into assets/icons/lucide/.

Lucide (https://lucide.dev) is ISC-licensed; the license file is copied next to the icons. The icons
are committed, so this only needs re-running to change the set or the version.

Usage: uv run python scripts/fetch_icons.py
"""

import sys
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

from autoclaim.paths import REPO_ROOT

VERSION = "1.53.0"
RELEASE = (
    f"https://github.com/lucide-icons/lucide/releases/download/{VERSION}/lucide-icons-{VERSION}.zip"
)
LICENSE_URL = f"https://raw.githubusercontent.com/lucide-icons/lucide/{VERSION}/LICENSE"
ICONS = (
    "book-open", "calculator", "calendar", "camera", "car", "circle-check", "circle-x",
    "clipboard-list", "file-text", "image", "inbox", "info", "scale", "search", "shield-alert",
    "shield-check", "shuffle", "signpost", "sparkles", "triangle-alert", "user", "user-check",
    "banknote", "loader-circle",
)  # fmt: skip
DEST = REPO_ROOT / "assets" / "icons" / "lucide"


def extract(archive: Path, names: Iterable[str], dest: Path) -> list[str]:
    """Copy `<name>.svg` for each name out of the release zip; raises if one is missing."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        by_name = {Path(n).name: n for n in z.namelist() if n.endswith(".svg")}
        out = []
        for name in names:
            member = by_name.get(f"{name}.svg")
            if member is None:
                raise KeyError(f"icon {name!r} is not in Lucide {VERSION}")
            (dest / f"{name}.svg").write_bytes(z.read(member))
            out.append(name)
    return out


def main() -> int:
    cache = REPO_ROOT / ".cache" / "lucide" / f"lucide-icons-{VERSION}.zip"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RELEASE, cache)  # official release asset
    names = extract(cache, ICONS, DEST)
    with urllib.request.urlopen(LICENSE_URL, timeout=60) as resp:
        (DEST / "LICENSE").write_bytes(resp.read())
    print(f"{len(names)} Lucide {VERSION} icons -> {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
