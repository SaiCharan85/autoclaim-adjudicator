"""NHTSA vehicle-safety complaints (public domain) -> real crash stories to try the harness on.

Owners describe, in their own words (or as told to NHTSA staff), what happened to their car. Only
complaints that report a crash are kept. These are not insurance claims: they are used as
real-world statements of loss, never for training or evaluation (no truth labels exist).

API: https://api.nhtsa.gov/complaints/complaintsByVehicle?make=&model=&modelYear= (no key).
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

API = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
LICENSE = "Public domain (U.S. Government work, NHTSA Office of Defects Investigation)"
FILE = "complaints.jsonl"
# popular US vehicles (make, model, body class), model years 2021-2022
VEHICLES = (
    ("Toyota", "Camry", "car"), ("Toyota", "RAV4", "suv"), ("Toyota", "Corolla", "car"),
    ("Honda", "Civic", "car"), ("Honda", "CR-V", "suv"), ("Honda", "Accord", "car"),
    ("Ford", "F-150 SUPER CREW", "pickup"), ("Ford", "Escape", "suv"), ("Ford", "Explorer", "suv"),
    ("Chevrolet", "Silverado 1500", "pickup"), ("Chevrolet", "Equinox", "suv"),
    ("Nissan", "Altima", "car"), ("Nissan", "Rogue", "suv"), ("Hyundai", "Elantra", "car"),
    ("Hyundai", "Tucson", "suv"), ("Tesla", "Model 3", "car"), ("Tesla", "Model Y", "suv"),
    ("Jeep", "Grand Cherokee", "suv"), ("Subaru", "Outback", "suv"), ("Kia", "Forte", "car"),
)  # fmt: skip
YEARS = (2021, 2022)
MIN_CHARS, MAX_CHARS = 200, 2500

JsonFetcher = Callable[[str], dict[str, Any]]


class Complaint(BaseModel):
    odi_number: int
    make: str
    model: str
    year: int
    body_class: str
    date_of_incident: date
    date_filed: date
    injuries: int
    components: str
    summary: str


def _date(s: str | None) -> date | None:
    try:
        return datetime.strptime(s or "", "%m/%d/%Y").date()
    except ValueError:
        return None


def parse(results: Iterable[dict[str, Any]], make: str, model: str, year: int,
          body_class: str) -> list[Complaint]:  # fmt: skip
    """Crash complaints with a usable story and an incident date (bad rows are skipped)."""
    out = []
    for r in results:
        text = " ".join(str(r.get("summary") or "").split())
        incident, filed = _date(r.get("dateOfIncident")), _date(r.get("dateComplaintFiled"))
        if not r.get("crash") or incident is None or filed is None:
            continue
        if incident.year < year - 1 or incident > filed:  # mistyped dates in the source
            continue
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            continue
        out.append(Complaint(odi_number=int(r["odiNumber"]), make=make, model=model, year=year,
                             body_class=body_class, date_of_incident=incident, date_filed=filed,
                             injuries=int(r.get("numberOfInjuries") or 0),
                             components=str(r.get("components") or ""), summary=text))  # fmt: skip
    return out


def get_json(url: str, timeout: float = 60) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "autoclaim-adjudicator"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8", errors="replace"))
    return data


def url_for(make: str, model: str, year: int) -> str:
    return f"{API}?" + urllib.parse.urlencode({"make": make, "model": model, "modelYear": year})


def download(dest: Path, fetch: JsonFetcher = get_json, pause_s: float = 0.3,
             vehicles: Sequence[tuple[str, str, str]] = VEHICLES,
             years: Sequence[int] = YEARS) -> Path:  # fmt: skip
    """One request per vehicle and year (40 by default), politely spaced; deduplicated."""
    dest.mkdir(parents=True, exist_ok=True)
    seen: dict[int, Complaint] = {}
    for make, model, body in vehicles:
        for year in years:
            try:
                data = fetch(url_for(make, model, year))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                print(f"  skipped {year} {make} {model}: {exc}")  # one bad vehicle never stops it
                continue
            for c in parse(data.get("results") or [], make, model, year, body):
                seen.setdefault(c.odi_number, c)
            time.sleep(pause_s)
    rows = sorted(seen.values(), key=lambda c: (c.date_of_incident, c.odi_number), reverse=True)
    path = dest / FILE
    path.write_text("".join(c.model_dump_json() + "\n" for c in rows), encoding="utf-8")
    meta = {"source": API, "license": LICENSE, "rows": len(rows),
            "downloaded_at": datetime.now(UTC).isoformat()}  # fmt: skip
    (dest / "manifest.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path


def load(path: Path) -> list[Complaint]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Complaint.model_validate_json(x) for x in lines if x.strip()]
