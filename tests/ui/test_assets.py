import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from autoclaim.ui import assets as a

ROOT = Path(__file__).resolve().parents[2]


def load(script: str):  # the scripts are not a package: load them by path
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fi, fd = load("fetch_icons"), load("fetch_demo_images")


# ---------------------------------------------------------------- committed assets


def test_every_icon_the_ui_uses_is_vendored_with_its_license() -> None:
    for name in fi.ICONS:
        assert (a.ICONS / f"{name}.svg").exists(), name
    assert "ISC" in (a.ICONS / "LICENSE").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["logo", "favicon", "empty_queue", "onboarding", "error"])
def test_self_made_illustrations_exist_and_use_the_brand_palette(name: str) -> None:
    svg = (a.BRAND / f"{name}.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "<script" not in svg
    colors = {c.lower() for c in __import__("re").findall(r"#[0-9a-fA-F]{6}", svg)}
    allowed = {v.lower() for v in a.PALETTE.values()} | {"#ffffff", "#e5e7eb", "#9ca3af",
                                                        "#fff7ed"}  # fmt: skip
    assert colors <= allowed, colors - allowed
    assert (a.BRAND / "favicon.png").exists()


def test_icon_is_inline_svg_with_size_color_and_stroke() -> None:
    svg = a.icon("car", 32, "#2563eb", 2.5)
    assert svg.startswith('<svg class="lc" aria-hidden="true"')
    assert 'width="32"' in svg and 'height="32"' in svg
    assert 'stroke="#2563eb"' in svg and 'stroke-width="2.5"' in svg and "\n" not in svg
    with pytest.raises(KeyError, match="fetch_icons"):
        a.icon("not-an-icon")


def test_illustration_is_an_inline_image() -> None:
    tag = a.illustration("error", 200, "oops")
    assert tag.startswith('<img src="data:image/svg+xml;base64,') and 'alt="oops"' in tag


def test_animations_are_pure_svg_css_and_respect_reduced_motion() -> None:
    assert "@keyframes" in a.ANIMATION_CSS and "prefers-reduced-motion" in a.ANIMATION_CSS
    for markup in (a.loader_html("Working <now>", "x"), a.success_tick(), a.review_alert(),
                   a.deny_mark()):  # fmt: skip
        assert "<svg" in markup and "<script" not in markup
    assert "Working &lt;now&gt;" in a.loader_html("Working <now>")  # escaped
    assert 'role="status"' in a.loader_html("x")


# ---------------------------------------------------------------- demo photos and credits

PHOTO = {"id": "unsplash-1", "provider": "Unsplash", "file": None, "url": "https://img/1.jpg",
         "alt": "dented car", "photographer": "Ana <B>", "photographer_url": "https://u/ana",
         "source_url": "https://unsplash.com/", "photo_page": "https://u/p/1"}  # fmt: skip


def test_demo_photos_keep_hotlinks_and_existing_files_only(tmp_path: Path) -> None:
    (tmp_path / "pexels-2.jpg").write_bytes(b"\xff\xd8jpg")
    rows = [PHOTO, {**PHOTO, "id": "p2", "url": None, "file": "pexels-2.jpg"},
            {**PHOTO, "id": "p3", "url": None, "file": "missing.jpg"}]  # fmt: skip
    (tmp_path / "credits.json").write_text(json.dumps(rows))
    got = a.demo_photos(tmp_path)
    assert [p["id"] for p in got] == ["unsplash-1", "p2"]
    assert a.photo_src(got[0], tmp_path) == "https://img/1.jpg"  # Unsplash: hotlinked
    assert a.photo_src(got[1], tmp_path).startswith("data:image/jpg;base64,")
    assert a.demo_photos(tmp_path / "none") == []


def test_photo_choice_is_stable_and_credit_is_escaped() -> None:
    photos = [PHOTO, {**PHOTO, "id": "b"}]
    assert a.photo_for(7, photos) == a.photo_for(7, photos) and a.photo_for(1, []) is None
    credit = a.credit_html(PHOTO)
    assert credit.startswith('Photo by <a href="https://u/ana"') and "Ana &lt;B&gt;" in credit
    assert credit.endswith(">Unsplash</a>")


# ---------------------------------------------------------------- the fetch scripts


def test_extract_copies_named_icons_and_refuses_unknown(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("icons/car.svg", "<svg>car</svg>")
        z.writestr("icons/car.json", "{}")
    archive = tmp_path / "lucide.zip"
    archive.write_bytes(buf.getvalue())
    assert fi.extract(archive, ["car"], tmp_path / "out") == ["car"]
    assert (tmp_path / "out" / "car.svg").read_text() == "<svg>car</svg>"
    with pytest.raises(KeyError, match="nope"):
        fi.extract(archive, ["nope"], tmp_path / "out")


def unsplash_result(i: int) -> dict:
    return {"id": f"u{i}", "alt_description": "car damage",
            "urls": {"regular": f"https://images.unsplash.com/{i}"},
            "links": {"html": f"https://unsplash.com/photos/u{i}",
                      "download_location": f"https://api.unsplash.com/photos/u{i}/download"},
            "user": {"name": f"P{i}",
                     "links": {"html": f"https://unsplash.com/@p{i}"}}}  # fmt: skip


def test_unsplash_hotlinks_pings_downloads_and_tags_credits(tmp_path: Path) -> None:
    pings: list[str] = []
    headers_seen: list[dict] = []

    def get(url: str, headers: dict) -> dict:
        headers_seen.append(headers)
        assert url.startswith("https://api.unsplash.com/search/photos?") and "per_page=2" in url
        return {"results": [unsplash_result(1), unsplash_result(2)]}

    def ping(url: str, headers: dict) -> bytes:
        pings.append(url)
        return b"{}"

    rows = fd.fetch("unsplash", "KEY", 2, tmp_path, get, ping)
    assert pings == [r["links"]["download_location"] for r in (unsplash_result(1),
                                                                unsplash_result(2))]  # fmt: skip
    assert headers_seen[0]["Authorization"] == "Client-ID KEY"
    assert rows[0]["file"] is None and rows[0]["url"] == "https://images.unsplash.com/1"
    assert "utm_source=autoclaim_adjudicator&utm_medium=referral" in rows[0]["photographer_url"]
    assert not list(tmp_path.glob("*.jpg"))  # nothing re-hosted
    assert json.loads((tmp_path / "credits.json").read_text()) == rows


def test_pexels_downloads_the_files_with_credits(tmp_path: Path) -> None:
    photo = {"id": 9, "url": "https://www.pexels.com/photo/9", "photographer": "Lee",
             "photographer_url": "https://www.pexels.com/@lee", "alt": "crash",
             "src": {"large": "https://images.pexels.com/9.jpg"}}  # fmt: skip
    rows = fd.fetch("pexels", "KEY", 1, tmp_path, lambda u, h: {"photos": [photo]},
                    lambda u, h: b"JPEG")  # fmt: skip
    assert rows[0]["file"] == "pexels-9.jpg" and (tmp_path / "pexels-9.jpg").read_bytes() == b"JPEG"
    assert rows[0]["provider"] == "Pexels" and rows[0]["photographer"] == "Lee"


def test_cli_dry_run_and_missing_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    assert fd.main(["--dry-run", "--n", "5"]) == 0
    assert "~6 requests" in capsys.readouterr().out
    monkeypatch.setattr(fd, "load_dotenv", lambda *_: None)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    assert fd.main(["--provider", "pexels"]) == 1
