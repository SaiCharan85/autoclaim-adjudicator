"""Download Kaggle datasets into data/raw/<key>/ with a hash manifest.

Idempotent: if the manifest exists and every file's sha256 still matches, nothing is downloaded.
The kaggle package authenticates on import, so it is imported lazily inside `default_api`.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from autoclaim.datasets.sources import SOURCES, DatasetSource
from autoclaim.paths import raw_dir

MANIFEST_NAME = "manifest.json"
KAGGLE_METADATA_NAME = "dataset-metadata.json"


class KaggleAuthError(RuntimeError):
    pass


class DatasetFileMissingError(RuntimeError):
    pass


class KaggleApiLike(Protocol):
    def dataset_download_files(
        self,
        dataset: str,
        path: str | None = ...,
        force: bool = ...,
        quiet: bool = ...,
        unzip: bool = ...,
    ) -> None: ...

    def dataset_metadata(self, dataset: str, path: str) -> str: ...


class Manifest(BaseModel):
    source_key: str
    kaggle_ref: str
    files: dict[str, str]  # filename -> sha256
    licenses: list[str]
    title: str | None = None
    downloaded_at: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(dest: Path) -> Manifest | None:
    path = dest / MANIFEST_NAME
    if not path.exists():
        return None
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def manifest_is_valid(manifest: Manifest, source: DatasetSource, dest: Path) -> bool:
    """True if the manifest covers every expected file and all hashes still match on disk."""
    if set(source.files) - set(manifest.files):
        return False
    return all(
        (dest / name).exists() and sha256_file(dest / name) == digest
        for name, digest in manifest.files.items()
    )


def parse_kaggle_metadata(path: Path) -> tuple[list[str], str | None]:
    """Extract license names and title from Kaggle's dataset-metadata.json.

    The kaggle 2.x API nests fields under "info"; older versions wrote them at the top level.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = raw.get("info", raw)
    licenses = [lic["name"] for lic in meta.get("licenses", []) if lic.get("name")]
    return licenses, meta.get("title")


def default_api() -> KaggleApiLike:
    """Authenticate with KAGGLE_API_TOKEN (or legacy KAGGLE_USERNAME/KAGGLE_KEY)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:  # the kaggle package calls exit(1) on missing credentials
        raise KaggleAuthError(
            "Kaggle credentials not found. Set KAGGLE_API_TOKEN in .env (see .env.example)."
        ) from exc
    return api  # type: ignore[no-any-return]


def download(
    source: DatasetSource,
    dest: Path,
    api_factory: Callable[[], KaggleApiLike] = default_api,
    force: bool = False,
) -> tuple[Manifest, bool]:
    """Download `source` into `dest`.

    Returns (manifest, downloaded); downloaded is False on a cache hit.
    """
    existing = read_manifest(dest)
    if not force and existing is not None and manifest_is_valid(existing, source, dest):
        return existing, False

    api = api_factory()
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(
        source.kaggle_ref, path=str(dest), force=True, quiet=True, unzip=True
    )
    for archive in dest.glob("*.zip"):
        archive.unlink()

    missing = [name for name in source.files if not (dest / name).exists()]
    if missing:
        raise DatasetFileMissingError(f"{source.kaggle_ref}: expected files not found: {missing}")

    meta_path = Path(api.dataset_metadata(source.kaggle_ref, str(dest)))
    licenses, title = parse_kaggle_metadata(meta_path)

    manifest = Manifest(
        source_key=source.key,
        kaggle_ref=source.kaggle_ref,
        files={name: sha256_file(dest / name) for name in source.files},
        licenses=licenses,
        title=title,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (dest / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest, True


def main(
    argv: Sequence[str] | None = None, api_factory: Callable[[], KaggleApiLike] = default_api
) -> int:
    parser = argparse.ArgumentParser(description="Download Kaggle datasets into data/raw/.")
    # no argparse `choices`: it rejects list defaults with nargs="*"
    parser.add_argument("sources", nargs="*", help=f"default: all ({', '.join(SOURCES)})")
    parser.add_argument("--force", action="store_true", help="re-download even if hashes match")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.sources) - set(SOURCES))
    if unknown:
        parser.error(f"unknown source(s) {unknown}; choose from {sorted(SOURCES)}")
    keys = args.sources or list(SOURCES)

    from dotenv import load_dotenv

    load_dotenv()
    for key in keys:
        source = SOURCES[key]
        manifest, downloaded = download(source, raw_dir(key), api_factory, force=args.force)
        status = "downloaded" if downloaded else "up to date"
        licenses = ", ".join(manifest.licenses) or "UNKNOWN (check the dataset page)"
        print(f"[{key}] {status}: {', '.join(manifest.files)} | license: {licenses}")
    return 0
