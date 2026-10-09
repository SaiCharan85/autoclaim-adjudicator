"""Demo car-damage photos for the UI from the official Unsplash or Pexels API.

Each photo's credit (photographer + links) goes to assets/demo_claims/credits.json and is shown
under the photo in the UI, as both providers' guidelines ask.

- Unsplash (UNSPLASH_ACCESS_KEY): its API guidelines require hotlinking the returned image URLs
  and triggering the photo's download endpoint, so only the links and credits are stored; the UI
  loads the image from Unsplash. Credit links carry utm_source / utm_medium=referral.
- Pexels (PEXELS_API_KEY): the photos are downloaded into the folder.

Usage:
  uv run python scripts/fetch_demo_images.py [--provider unsplash|pexels] [--n 20] [--dry-run]
The folder is git-ignored; re-running regenerates it.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from autoclaim.paths import REPO_ROOT

DEST = REPO_ROOT / "assets" / "demo_claims"
APP = "autoclaim_adjudicator"  # utm_source for Unsplash credit links
QUERY = "car accident damage"
KEYS = {"unsplash": "UNSPLASH_ACCESS_KEY", "pexels": "PEXELS_API_KEY"}

GetJson = Callable[[str, dict[str, str]], Any]
GetBytes = Callable[[str, dict[str, str]], bytes]


def get_bytes(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": APP, **headers})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: bytes = resp.read()
    return data


def get_json(url: str, headers: dict[str, str]) -> Any:
    return json.loads(get_bytes(url, headers).decode("utf-8"))


def utm(url: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}utm_source={APP}&utm_medium=referral"


def unsplash(key: str, n: int, get: GetJson, ping: GetBytes) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Client-ID {key}", "Accept-Version": "v1"}
    q = urllib.parse.urlencode({"query": QUERY, "per_page": n, "orientation": "landscape"})
    results = get(f"https://api.unsplash.com/search/photos?{q}", headers)["results"]
    rows = []
    for r in results[:n]:
        ping(r["links"]["download_location"], headers)  # required when a photo is used
        rows.append({"id": f"unsplash-{r['id']}", "provider": "Unsplash", "file": None,
                     "url": r["urls"]["regular"], "alt": r.get("alt_description") or "",
                     "photographer": r["user"]["name"],
                     "photographer_url": utm(r["user"]["links"]["html"]),
                     "source_url": utm("https://unsplash.com/"),
                     "photo_page": utm(r["links"]["html"])})  # fmt: skip
    return rows


def pexels(key: str, n: int, get: GetJson, download: GetBytes, dest: Path) -> list[dict[str, Any]]:
    headers = {"Authorization": key}
    q = urllib.parse.urlencode({"query": QUERY, "per_page": n, "orientation": "landscape"})
    photos = get(f"https://api.pexels.com/v1/search?{q}", headers)["photos"]
    rows = []
    for p in photos[:n]:
        name = f"pexels-{p['id']}.jpg"
        (dest / name).write_bytes(download(p["src"]["large"], {}))
        rows.append({"id": f"pexels-{p['id']}", "provider": "Pexels", "file": name,
                     "url": None, "alt": p.get("alt") or "", "photographer": p["photographer"],
                     "photographer_url": p["photographer_url"], "source_url": "https://www.pexels.com",
                     "photo_page": p["url"]})  # fmt: skip
    return rows


def fetch(provider: str, key: str, n: int, dest: Path, get: GetJson = get_json,
          raw: GetBytes = get_bytes) -> list[dict[str, Any]]:  # fmt: skip
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("pexels-*.jpg"):
        old.unlink()
    rows = unsplash(key, n, get, raw) if provider == "unsplash" else pexels(key, n, get, raw, dest)
    (dest / "credits.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--provider", choices=list(KEYS), default="unsplash")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    requests = 1 + args.n  # one search + one ping (Unsplash) or download (Pexels) per photo
    print(f"{args.provider}: ~{requests} requests for {args.n} photos of {QUERY!r} -> {DEST}")
    if args.dry_run:
        return 0
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get(KEYS[args.provider])
    if not key:
        print(f"set {KEYS[args.provider]} in .env (free key; see the docstring)")
        return 1
    rows = fetch(args.provider, key, args.n, DEST)
    print(f"{len(rows)} photos; credits -> {DEST / 'credits.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
