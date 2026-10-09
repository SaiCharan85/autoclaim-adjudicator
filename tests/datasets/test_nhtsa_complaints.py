import json
from pathlib import Path

from autoclaim.datasets import nhtsa_complaints as nc

STORY = "I was driving home on the highway when the car ahead braked hard. " * 5


def raw(**over: object) -> dict:
    r = {"odiNumber": 11600001, "crash": True, "fire": False, "numberOfInjuries": 1,
         "dateOfIncident": "09/18/2024", "dateComplaintFiled": "12/30/2024",
         "components": "SERVICE BRAKES", "summary": STORY}  # fmt: skip
    return r | over


def test_parse_keeps_crash_stories_with_dates_and_normalizes_whitespace() -> None:
    rows = [raw(), raw(odiNumber=2, crash=False), raw(odiNumber=3, dateOfIncident=None),
            raw(odiNumber=4, summary="too short"), raw(odiNumber=5, summary="x" * 3000),
            raw(odiNumber=6, summary="  spaced\n\n" + STORY),
            raw(odiNumber=8, dateOfIncident="09/02/1981")]  # fmt: skip
    out = nc.parse(rows, "Honda", "Civic", 2021, "car")
    assert [c.odi_number for c in out] == [11600001, 6]
    c = out[0]
    assert (c.make, c.model, c.year, c.body_class, c.injuries) == ("Honda", "Civic", 2021, "car", 1)
    assert c.date_of_incident.isoformat() == "2024-09-18" and "\n" not in out[1].summary


def test_download_dedupes_across_vehicles_and_writes_a_manifest(tmp_path: Path) -> None:
    urls: list[str] = []

    def fetch(url: str) -> dict:
        urls.append(url)
        return {"results": [raw(), raw(odiNumber=7, dateOfIncident="11/02/2024")]}

    path = nc.download(tmp_path, fetch, pause_s=0, vehicles=[("Honda", "Civic", "car"),
                       ("Ford", "F-150", "pickup")], years=[2021])  # fmt: skip
    rows = nc.load(path)
    assert len(urls) == 2 and "model=F-150" in urls[1] and "modelYear=2021" in urls[0]
    assert [c.odi_number for c in rows] == [7, 11600001]  # newest incident first, no duplicates
    assert json.loads((tmp_path / "manifest.json").read_text())["rows"] == 2


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert nc.load(tmp_path / "nope.jsonl") == []


def test_a_failing_vehicle_is_skipped(tmp_path: Path) -> None:
    import urllib.error

    def fetch(url: str) -> dict:
        if "Ford" in url:
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)  # type: ignore[arg-type]
        return {"results": [raw()]}

    path = nc.download(tmp_path, fetch, pause_s=0, vehicles=[("Ford", "F-150", "pickup"),
                       ("Honda", "Civic", "car")], years=[2021])  # fmt: skip
    assert [c.make for c in nc.load(path)] == ["Honda"]
