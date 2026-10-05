import json
import zipfile
from pathlib import Path

import pytest

from autoclaim.datasets import download as dl
from autoclaim.datasets.sources import CRSS, CRSS_YEARS, DatasetSource, UrlSource

SOURCE = DatasetSource(
    key="toy", kaggle_ref="owner/toy", files=("a.csv", "b.csv"), description="toy"
)


class FakeKaggleApi:
    """Mimics kaggle.KaggleApi: writes files plus a zip, and a metadata JSON."""

    def __init__(self, files: dict[str, str] | None = None, licenses=("CC0-1.0",)) -> None:
        self.files = files if files is not None else {"a.csv": "x,y\n1,2\n", "b.csv": "z\n3\n"}
        self.licenses = licenses
        self.download_calls = 0

    def dataset_download_files(self, dataset, path=None, force=False, quiet=True, unzip=False):
        self.download_calls += 1
        dest = Path(path)
        for name, body in self.files.items():
            (dest / name).write_text(body, encoding="utf-8")
        (dest / "toy.zip").write_bytes(b"zip")

    def dataset_metadata(self, dataset, path):
        # real kaggle 2.x shape: fields nested under "info"
        meta = {"info": {"title": "Toy Data", "licenses": [{"name": n} for n in self.licenses]}}
        out = Path(path) / dl.KAGGLE_METADATA_NAME
        out.write_text(json.dumps(meta), encoding="utf-8")
        return str(out)


def test_download_writes_files_manifest_and_removes_zip(tmp_path: Path) -> None:
    api = FakeKaggleApi()
    manifest, downloaded = dl.download(SOURCE, tmp_path, lambda: api)
    assert downloaded
    assert set(manifest.files) == {"a.csv", "b.csv"}
    assert manifest.files["a.csv"] == dl.sha256_file(tmp_path / "a.csv")
    assert manifest.licenses == ["CC0-1.0"]
    assert manifest.title == "Toy Data"
    assert not list(tmp_path.glob("*.zip"))
    assert dl.read_manifest(tmp_path) == manifest


def test_second_call_is_cache_hit_without_auth(tmp_path: Path) -> None:
    dl.download(SOURCE, tmp_path, FakeKaggleApi)

    def must_not_be_called():
        raise AssertionError("API should not be constructed on a cache hit")

    _, downloaded = dl.download(SOURCE, tmp_path, must_not_be_called)
    assert not downloaded


def test_force_redownloads(tmp_path: Path) -> None:
    api = FakeKaggleApi()
    dl.download(SOURCE, tmp_path, lambda: api)
    _, downloaded = dl.download(SOURCE, tmp_path, lambda: api, force=True)
    assert downloaded
    assert api.download_calls == 2


def test_tampered_file_triggers_redownload(tmp_path: Path) -> None:
    api = FakeKaggleApi()
    dl.download(SOURCE, tmp_path, lambda: api)
    (tmp_path / "a.csv").write_text("tampered", encoding="utf-8")
    _, downloaded = dl.download(SOURCE, tmp_path, lambda: api)
    assert downloaded
    assert api.download_calls == 2


def test_deleted_file_triggers_redownload(tmp_path: Path) -> None:
    api = FakeKaggleApi()
    dl.download(SOURCE, tmp_path, lambda: api)
    (tmp_path / "b.csv").unlink()
    assert dl.download(SOURCE, tmp_path, lambda: api)[1]


def test_missing_expected_file_raises_and_writes_no_manifest(tmp_path: Path) -> None:
    api = FakeKaggleApi(files={"a.csv": "x\n"})
    with pytest.raises(dl.DatasetFileMissingError, match=r"b\.csv"):
        dl.download(SOURCE, tmp_path, lambda: api)
    assert dl.read_manifest(tmp_path) is None


def test_manifest_missing_a_source_file_is_invalid(tmp_path: Path) -> None:
    manifest, _ = dl.download(SOURCE, tmp_path, FakeKaggleApi)
    partial = manifest.model_copy(update={"files": {"a.csv": manifest.files["a.csv"]}})
    assert not dl.manifest_is_valid(partial, SOURCE, tmp_path)


def test_parse_metadata_handles_missing_licenses(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"licenses": [{"name": ""}, {}]}), encoding="utf-8")
    assert dl.parse_kaggle_metadata(path) == ([], None)


def test_parse_metadata_supports_flat_legacy_shape(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"title": "T", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    assert dl.parse_kaggle_metadata(path) == (["CC0-1.0"], "T")


def test_sha256_file_known_value(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert dl.sha256_file(path, chunk_size=1) == expected


def test_default_api_converts_sys_exit_to_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from kaggle.api import kaggle_api_extended

    def fake_authenticate(self):
        raise SystemExit(1)

    monkeypatch.setattr(kaggle_api_extended.KaggleApi, "authenticate", fake_authenticate)
    with pytest.raises(dl.KaggleAuthError, match="KAGGLE_API_TOKEN"):
        dl.default_api()


def test_main_downloads_then_reports_up_to_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AUTOCLAIM_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(dl.SOURCES, "toy", SOURCE)
    assert dl.main(["toy"], api_factory=FakeKaggleApi) == 0
    assert "[toy] downloaded" in capsys.readouterr().out
    assert dl.main(["toy"], api_factory=FakeKaggleApi) == 0
    assert "up to date" in capsys.readouterr().out
    assert (tmp_path / "raw" / "toy" / dl.MANIFEST_NAME).exists()


def test_main_defaults_to_all_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AUTOCLAIM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dl, "SOURCES", {"toy": SOURCE})
    monkeypatch.setattr(dl, "URL_SOURCES", {})
    assert dl.main([], api_factory=FakeKaggleApi) == 0
    assert "[toy]" in capsys.readouterr().out


def test_main_rejects_unknown_source() -> None:
    with pytest.raises(SystemExit):
        dl.main(["not_a_source"])


# ---------------------------------------------------------------- URL (zip) sources

URL_SOURCE = UrlSource(
    key="toy_url",
    url="https://example.invalid/toy.zip",
    files=("accident.csv", "vehicle.csv"),
    license="Public domain",
    description="toy",
)


def make_zip_fetcher(members: dict[str, str]):
    calls = []

    def fetch(url: str, path: Path) -> None:
        calls.append(url)
        with zipfile.ZipFile(path, "w") as zf:
            for name, body in members.items():
                zf.writestr(name, body)

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


GOOD_MEMBERS = {
    "TOY2024CSV/ACCIDENT.CSV": "CASENUM\n1\n",  # case differs from the requested name
    "TOY2024CSV/vehicle.csv": "CASENUM,VEH_NO\n1,1\n",
    "TOY2024CSV/person.csv": "unused\n",
}


def test_download_url_extracts_only_requested_files(tmp_path: Path) -> None:
    fetch = make_zip_fetcher(GOOD_MEMBERS)
    manifest, downloaded = dl.download_url(URL_SOURCE, tmp_path, fetch)
    assert downloaded
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "accident.csv",
        dl.MANIFEST_NAME,
        "vehicle.csv",
    ]  # person.csv skipped, archive deleted
    assert manifest.url == URL_SOURCE.url
    assert manifest.licenses == ["Public domain"]
    assert manifest.archive_sha256 is not None
    assert manifest.files["vehicle.csv"] == dl.sha256_file(tmp_path / "vehicle.csv")


def test_download_url_is_cached(tmp_path: Path) -> None:
    fetch = make_zip_fetcher(GOOD_MEMBERS)
    dl.download_url(URL_SOURCE, tmp_path, fetch)
    _, downloaded = dl.download_url(URL_SOURCE, tmp_path, fetch)
    assert not downloaded
    assert len(fetch.calls) == 1


def test_download_url_redownloads_tampered_file(tmp_path: Path) -> None:
    fetch = make_zip_fetcher(GOOD_MEMBERS)
    dl.download_url(URL_SOURCE, tmp_path, fetch)
    (tmp_path / "accident.csv").write_text("tampered", encoding="utf-8")
    assert dl.download_url(URL_SOURCE, tmp_path, fetch)[1]


def test_download_url_missing_member_raises_and_cleans_up(tmp_path: Path) -> None:
    fetch = make_zip_fetcher({"X/accident.csv": "a\n"})
    with pytest.raises(dl.DatasetFileMissingError, match=r"vehicle\.csv"):
        dl.download_url(URL_SOURCE, tmp_path, fetch)
    assert not (tmp_path / "_download.zip").exists()
    assert dl.read_manifest(tmp_path) is None


def test_failed_fetch_leaves_no_archive(tmp_path: Path) -> None:
    def broken(url: str, path: Path) -> None:
        path.write_bytes(b"partial")
        raise OSError("network down")

    with pytest.raises(OSError, match="network down"):
        dl.download_url(URL_SOURCE, tmp_path, broken)
    assert not (tmp_path / "_download.zip").exists()


def test_legacy_kaggle_manifest_still_loads(tmp_path: Path) -> None:
    (tmp_path / dl.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "source_key": "vehicle_fraud",
                "kaggle_ref": "o/d",
                "files": {},
                "licenses": ["CC0-1.0"],
                "downloaded_at": "2026-10-04T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    manifest = dl.read_manifest(tmp_path)
    assert manifest is not None
    assert manifest.url is None


def test_crss_registry_covers_recent_years() -> None:
    assert set(CRSS) == set(range(2016, 2025))  # 2022-24 production pool + 2016-21 drift study
    assert set(CRSS_YEARS) == {2022, 2023, 2024}
    for year, src in CRSS.items():
        assert src.url.endswith(f"/CRSS/{year}/CRSS{year}CSV.zip")
        assert "Public domain" in src.license
        assert "vehicle.csv" in src.files


def test_main_downloads_url_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AUTOCLAIM_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(dl.URL_SOURCES, "toy_url", URL_SOURCE)
    assert dl.main(["toy_url"], fetch=make_zip_fetcher(GOOD_MEMBERS)) == 0
    assert "[toy_url] downloaded" in capsys.readouterr().out
    assert (tmp_path / "raw" / "toy_url" / "vehicle.csv").exists()
