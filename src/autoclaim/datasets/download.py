"""Download datasets into data/raw/<key>/ with a hash manifest.

Two source kinds: Kaggle datasets and zip archives at public URLs (e.g. NHTSA CRSS).
Idempotent: if the manifest exists and every file's sha256 still matches, nothing is downloaded.
The kaggle package authenticates on import, so it is imported lazily inside `default_api`.
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from autoclaim.datasets.sources import SOURCES, URL_SOURCES, DatasetSource, UrlSource
from autoclaim.paths import raw_dir

MANIFEST_NAME = "manifest.json"
BLS_KEY = "bls_cpi"  # BLS price indexes (datasets/bls.py), fetched from the public API
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


Fetcher = Callable[[str, Path], None]  # (url, destination file) -> downloads it


class Manifest(BaseModel):
    source_key: str
    kaggle_ref: str | None = None
    url: str | None = None
    archive_sha256: str | None = None
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


def manifest_is_valid(manifest: Manifest, source: DatasetSource | UrlSource, dest: Path) -> bool:
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


def default_fetch(url: str, path: Path, timeout: float = 300) -> None:
    """Stream `url` to `path` (no third-party HTTP dependency)."""
    request = urllib.request.Request(url, headers={"User-Agent": "autoclaim-adjudicator/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as resp, path.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


def extract_members(archive: Path, names: Sequence[str], dest: Path) -> list[str]:
    """Extract members whose basename matches `names` (case-insensitive) as lowercase files.

    Returns the requested names that were not found.
    """
    wanted = {n.lower() for n in names}
    found: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name.lower()
            if base in wanted and not info.is_dir():
                with zf.open(info) as src, (dest / base).open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                found.add(base)
    return sorted(wanted - found)


def download_url(
    source: UrlSource, dest: Path, fetch: Fetcher = default_fetch, force: bool = False
) -> tuple[Manifest, bool]:
    """Fetch a zip archive, keep only `source.files`, delete the archive, write a manifest."""
    existing = read_manifest(dest)
    if not force and existing is not None and manifest_is_valid(existing, source, dest):
        return existing, False

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "_download.zip"
    try:
        fetch(source.url, archive)
        archive_digest = sha256_file(archive)
        missing = extract_members(archive, source.files, dest)
        extract_members(archive, source.optional_files, dest)  # absent ones are fine
    finally:
        archive.unlink(missing_ok=True)
    if missing:
        raise DatasetFileMissingError(f"{source.url}: expected files not found: {missing}")

    manifest = Manifest(
        source_key=source.key,
        url=source.url,
        archive_sha256=archive_digest,
        files={
            name: sha256_file(dest / name)
            for name in (*source.files, *(o.lower() for o in source.optional_files))
            if (dest / name).exists()
        },
        licenses=[source.license],
        title=source.description,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (dest / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest, True


def main(
    argv: Sequence[str] | None = None,
    api_factory: Callable[[], KaggleApiLike] = default_api,
    fetch: Fetcher = default_fetch,
) -> int:
    known = [*SOURCES, *URL_SOURCES, BLS_KEY]
    parser = argparse.ArgumentParser(description="Download datasets into data/raw/.")
    # no argparse `choices`: it rejects list defaults with nargs="*"
    parser.add_argument("sources", nargs="*", help=f"default: all ({', '.join(known)})")
    parser.add_argument("--force", action="store_true", help="re-download even if hashes match")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.sources) - set(known))
    if unknown:
        parser.error(f"unknown source(s) {unknown}; choose from {sorted(known)}")
    keys = args.sources or known

    from dotenv import load_dotenv

    load_dotenv()
    for key in keys:
        if key == BLS_KEY:
            from autoclaim.datasets import bls

            path = bls.download_cpi(raw_dir(key))
            print(f"[{key}] downloaded: {path.name} | license: {bls.LICENSE}")
            continue
        if key in URL_SOURCES:
            manifest, downloaded = download_url(
                URL_SOURCES[key], raw_dir(key), fetch, force=args.force
            )
        else:
            manifest, downloaded = download(
                SOURCES[key], raw_dir(key), api_factory, force=args.force
            )
        status = "downloaded" if downloaded else "up to date"
        licenses = ", ".join(manifest.licenses) or "UNKNOWN (check the dataset page)"
        print(f"[{key}] {status}: {', '.join(manifest.files)} | license: {licenses}")
    return 0
