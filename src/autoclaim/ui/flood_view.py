"""Logic behind the "Flood claim" page (pure, tested; the page only lays it out).

A flood claim is structured, the way the NFIP records it: the policy (limits, deductibles, zone),
the loss (date, cause) and the adjuster's documented damage. The flood line decides with code only.
"""

import hashlib
import re
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from autoclaim.lines.flood.claim import DEDUCTIBLE_CODES, FLOOD_CAUSES, JUDGMENT_CAUSES, FloodClaim
from autoclaim.ui.try_claim_view import Source, details_from_text

# deductible amount -> FEMA code (the form asks for dollars, the claim stores the code)
DEDUCTIBLE_TO_CODE = {amount: code for code, amount in DEDUCTIBLE_CODES.items()}
DEDUCTIBLES = sorted(DEDUCTIBLE_TO_CODE)
CAUSES = {**FLOOD_CAUSES, **JUDGMENT_CAUSES}  # code -> plain English
ZONES = ["AE", "A", "VE", "X", "B", "C", "D"]  # FEMA flood zones (A/V = high risk)


class FloodForm(BaseModel):
    policyholder: str = ""
    address: str = ""
    policy_start: date | None = None
    flood_zone: str = "AE"
    primary_residence: bool = True
    building_limit: float = Field(default=250_000.0, ge=0)
    contents_limit: float = Field(default=100_000.0, ge=0)
    building_deductible: float = 2_000.0
    contents_deductible: float = 2_000.0
    date_of_loss: date | None = None
    cause_code: str = "4"
    flood_event: str = ""
    water_depth_inches: float | None = None
    building_damage: float | None = Field(default=None, ge=0)
    contents_damage: float | None = Field(default=None, ge=0)


SAMPLE = FloodForm(
    policyholder="Priya Raman", address="1415 Bayou Bend Court, Houston, TX 77007",
    policy_start=date(2024, 6, 1), flood_zone="AE", primary_residence=True,
    building_limit=250_000.0, contents_limit=100_000.0, building_deductible=2_000.0,
    contents_deductible=2_000.0, date_of_loss=date(2026, 5, 18), cause_code="4",
    flood_event="Flash flood, May 2026", water_depth_inches=14.0, building_damage=38_500.0,
    contents_damage=12_200.0,
)  # fmt: skip
REQUIRED = {"policyholder": "policyholder name", "date_of_loss": "date of loss",
            "policy_start": "policy start"}  # fmt: skip


def missing_fields(form: FloodForm) -> list[str]:
    out = [lab for k, lab in REQUIRED.items() if getattr(form, k) in (None, "")]
    if form.building_damage is None and form.contents_damage is None:
        out.append("building or contents damage")
    return out


def claim_id(form: FloodForm) -> str:
    """Same details -> same id (never decided twice); any change -> a new claim."""
    return "FLD-" + hashlib.sha256(form.model_dump_json().encode()).hexdigest()[:10].upper()


def to_claim(form: FloodForm) -> FloodClaim:
    if form.date_of_loss is None:
        raise ValueError("date of loss is required")
    return FloodClaim(
        claim_id=claim_id(form), date_of_loss=form.date_of_loss, policy_start=form.policy_start,
        building_limit=form.building_limit, contents_limit=form.contents_limit,
        building_damage=form.building_damage, contents_damage=form.contents_damage,
        building_deductible_code=DEDUCTIBLE_TO_CODE.get(form.building_deductible),
        contents_deductible_code=DEDUCTIBLE_TO_CODE.get(form.contents_deductible),
        cause_codes=[form.cause_code], flood_event=form.flood_event or None,
        flood_zone=form.flood_zone, primary_residence=form.primary_residence,
        water_depth=form.water_depth_inches,
    )  # fmt: skip


def _money(x: Any) -> str:
    return f"${float(x):,.2f}" if isinstance(x, int | float) else "-"


def _part_step(name: str, p: dict[str, Any]) -> dict[str, str]:
    title = f"{name.capitalize()} coverage"
    if not p.get("covered"):
        return {"title": title, "status": "bad", "detail": f"The policy has no {name} coverage."}
    if p.get("damage") is None:
        return {"title": title, "status": "ok", "detail": f"No {name} damage claimed."}
    return {"title": title, "status": "ok",
            "detail": f"{_money(p['damage'])} damage - {_money(p.get('deductible'))} deductible "
                      f"= {_money(p['payout'])} (within the limit)."}  # fmt: skip


def flood_trace(out: dict[str, Any]) -> list[dict[str, str]]:
    """How the flood claim was decided, step by step (status: ok / warn / bad)."""
    cov = out.get("coverage") or {}
    steps = []
    flood = cov.get("cause") not in (None, "unclear")
    steps.append({"title": "Checked the cause", "status": "ok" if flood else "warn",
                  "detail": f"Cause: {cov.get('cause', 'unknown')}."
                  + ("" if flood else " Not clearly a flood: an adjuster decides.")})  # fmt: skip
    before = bool(cov.get("before_inception"))
    dates = ("The loss is dated before the policy start: an adjuster checks the policy history."
             if before else "The loss falls within the policy.")  # fmt: skip
    steps.append({"title": "Checked the policy dates", "status": "warn" if before else "ok",
                  "detail": dates})  # fmt: skip
    for part in ("building", "contents"):
        if cov.get(part):
            steps.append(_part_step(part, cov[part]))
    if cov:
        steps.append({"title": "Total", "status": "ok",
                      "detail": f"Building + contents = {_money(cov.get('total'))}."})  # fmt: skip
    for j in cov.get("judgment_needed") or []:
        steps.append({"title": "Needs judgment", "status": "warn", "detail": j.replace("_", " ")})
    if "final" in out:
        steps.append(
            {
                "title": "Decided automatically",
                "status": "ok",
                "detail": "Clear-cut and within the $50,000 automatic authority.",
            }
        )
    elif "__interrupt__" in out:
        why = ", ".join(out["__interrupt__"][0].value.get("route_reasons") or [])
        steps.append(
            {
                "title": "Sent to an adjuster",
                "status": "warn",
                "detail": f"Why: {why.replace('_', ' ') or 'needs judgment'}.",
            }
        )
    return steps


def flood_summary(out: dict[str, Any]) -> dict[str, Any]:
    """What the page shows: who decided, the decision and why (same shape as the car page)."""
    decision = out.get("decision") or {}
    base = {"explanation": decision.get("explanation"), "proposed": decision.get("outcome")}
    if "final" in out:
        f = out["final"]
        return base | {"status": "decided", "outcome": f.get("outcome"), "payout": f.get("payout"),
                       "reasons": f.get("reasons") or [], "route_reasons": []}  # fmt: skip
    req = out["__interrupt__"][0].value
    prop = req.get("proposed_decision") or {}
    return base | {"status": "human_review", "outcome": None, "payout": prop.get("payout"),
                   "reasons": prop.get("reasons") or [],
                   "route_reasons": list(req.get("route_reasons") or [])}  # fmt: skip


# ---------------------------------------------------------------- real FEMA claims as stories


class FloodRecord(BaseModel):
    """One real FEMA NFIP claim, coverage fields only (no location finer than the state)."""

    id: str
    date_of_loss: date
    state: str
    flood_zone: str
    flood_event: str
    cause_code: str
    water_depth_inches: float | None
    building_limit: float
    contents_limit: float
    building_deductible: float | None
    contents_deductible: float | None
    building_damage: float | None
    contents_damage: float | None
    policy_start: date | None
    primary_residence: bool


def _opt_float(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def records_from_frame(raw: pd.DataFrame, n: int = 400, seed: int = 42) -> list[FloodRecord]:
    """A browsable sample of real claims with documented building damage, spread over events."""
    df = raw[pd.to_numeric(raw["buildingDamageAmount"], errors="coerce") > 0].copy()
    df["cause"] = df["causeOfDamage"].astype(str).str.replace(".0", "", regex=False).str[:1]
    df = df[df["cause"].isin(list(CAUSES))]
    df = df.sample(frac=1.0, random_state=seed).groupby("floodEvent", dropna=False).head(25)
    out = []
    for _, r in df.head(n).iterrows():
        start = pd.to_datetime(str(r.get("originalNBDate") or ""), errors="coerce")
        b_code = str(r.get("buildingDeductibleCode") or "").replace(".0", "")
        c_code = str(r.get("contentsDeductibleCode") or "").replace(".0", "")
        out.append(FloodRecord(
            id=str(r["id"]), date_of_loss=pd.Timestamp(r["dateOfLoss"]).date(),
            state=str(r.get("state") or "?"), flood_zone=str(r.get("ratedFloodZone") or "X"),
            flood_event=str(r.get("floodEvent") or "No named event"), cause_code=str(r["cause"]),
            water_depth_inches=_opt_float(r.get("waterDepth")),
            building_limit=_opt_float(r.get("totalBuildingInsuranceCoverage")) or 0.0,
            contents_limit=_opt_float(r.get("totalContentsInsuranceCoverage")) or 0.0,
            building_deductible=DEDUCTIBLE_CODES.get(b_code),
            contents_deductible=DEDUCTIBLE_CODES.get(c_code),
            building_damage=_opt_float(r.get("buildingDamageAmount")),
            contents_damage=_opt_float(r.get("contentsDamageAmount")),
            policy_start=None if pd.isna(start) else start.date(),
            primary_residence=bool(r.get("primaryResidenceIndicator")),
        ))  # fmt: skip
    return sorted(out, key=lambda x: x.date_of_loss, reverse=True)


def record_summary(r: FloodRecord) -> str:
    """The record in plain English: what the claim card shows in place of a story."""
    depth = f", water {r.water_depth_inches:g} inches deep" if r.water_depth_inches else ""
    contents = (f" and ${r.contents_damage:,.0f} to contents"
                if r.contents_damage else "")  # fmt: skip
    return (f"{r.flood_event} ({r.date_of_loss:%b %d, %Y}), {r.state}, flood zone {r.flood_zone}: "
            f"{CAUSES.get(r.cause_code, 'flood')}{depth}. The adjuster documented "
            f"${r.building_damage or 0:,.0f} of damage to the building{contents}.")  # fmt: skip


def form_from_record(r: FloodRecord) -> FloodForm:
    """Every detail from the record except the policyholder (FEMA redacts names)."""

    def ded(v: float | None) -> float:
        return v if v in DEDUCTIBLE_TO_CODE else 2_000.0

    return FloodForm(
        policy_start=r.policy_start, flood_zone=r.flood_zone,
        primary_residence=r.primary_residence, building_limit=r.building_limit,
        contents_limit=r.contents_limit, building_deductible=ded(r.building_deductible),
        contents_deductible=ded(r.contents_deductible), date_of_loss=r.date_of_loss,
        cause_code=r.cause_code, flood_event=r.flood_event,
        water_depth_inches=r.water_depth_inches, building_damage=r.building_damage,
        contents_damage=r.contents_damage,
    )  # fmt: skip


# ---------------------------------------------------------------- reading a flood story


class FloodStoryDetails(BaseModel):
    """What a policyholder's flood story says (null = it does not say)."""

    date_of_loss: date | None = None
    cause_code: str | None = None
    water_depth_inches: float | None = None
    building_damage: float | None = None
    contents_damage: float | None = None
    policyholder: str | None = None
    address: str | None = None
    flood_event: str | None = None


EXTRACT_FLOOD = """From a homeowner's account of a flood, copy only what it states: the date of
the flood; the cause as one code (1 storm surge or tidal water, 2 river, creek, bayou, stream or
lake overflow, 4 heavy rain, flash flood or snowmelt accumulation, 9 mudslide or earth movement);
the water depth inside in inches (feet x 12); the building damage and the contents (belongings)
damage in dollars; the writer's full name; the property address; and the named storm or event.
Use null for anything not stated. Never guess a date, name or amount."""

_CAUSE_WORDS = (("1", r"storm surge|tidal|high tide|king tide"),
                ("2", r"river|creek|bayou|stream|lake|levee"),
                ("4", r"flash flood|heavy rain|rainfall|downpour|snowmelt|storm drain"),
                ("9", r"mudslide|mudflow|landslide|earth movement"))  # fmt: skip
_DEPTH = re.compile(r"(\d+(?:\.\d+)?)\s*(inches|inch|in\b|feet|foot|ft\b)", re.IGNORECASE)


def flood_details_from_text(story: str, today: date) -> FloodStoryDetails:
    """Code-only reading: date, cause words, water depth and a single dollar amount."""
    low = " ".join(story.lower().split())
    cause = next((code for code, words in _CAUSE_WORDS if re.search(words, low)), None)
    depth = None
    m = _DEPTH.search(story)
    if m:
        depth = float(m[1]) * (12 if m[2].lower().startswith(("f",)) else 1)
    car = details_from_text(story, today)  # dates and amounts (shared with the car page)
    return FloodStoryDetails(date_of_loss=car.loss_date, cause_code=cause, water_depth_inches=depth,
                             building_damage=car.estimate_amount)  # fmt: skip


def _checked(values: dict[str, Any], today: date) -> dict[str, Any]:
    """Drop impossible answers (future dates, unknown causes, non-positive amounts, one-word
    names) so the other reading can fill the gap."""
    v = dict(values)
    if v.get("date_of_loss") is not None and v["date_of_loss"] > today:
        v["date_of_loss"] = None
    if v.get("cause_code") not in CAUSES:
        v["cause_code"] = None
    for k in ("building_damage", "contents_damage", "water_depth_inches"):
        if v.get(k) is not None and v[k] <= 0:
            v[k] = None
    name = str(v.get("policyholder") or "").strip()
    if len([w for w in name.split() if w.isalpha()]) < 2:
        v["policyholder"] = None
    return v


def read_flood_story(story: str, today: date, client: Any | None = None) -> FloodStoryDetails:
    """Model reading where available, code reading for its gaps, then sanity checks."""
    text = flood_details_from_text(story, today)
    llm = None
    if client is not None:
        try:
            llm = client.structured("intake", EXTRACT_FLOOD,
                                    f"report_date: {today}\nstatement:\n{story}",
                                    FloodStoryDetails).value  # fmt: skip
        except Exception:  # quota or provider outage: the code reading still applies
            llm = None
    llm_ok = _checked(llm.model_dump(), today) if llm is not None else {}
    text_ok = _checked(text.model_dump(), today)
    merged = {k: llm_ok.get(k) if llm_ok.get(k) not in (None, "") else text_ok.get(k)
              for k in FloodStoryDetails.model_fields}  # fmt: skip
    return FloodStoryDetails.model_validate(merged)


def form_from_story(d: FloodStoryDetails) -> FloodForm:
    """Story details where given; the policy part stays a sample (a story never contains it)."""
    return FloodForm(policyholder=d.policyholder or "", address=d.address or "",
                     date_of_loss=d.date_of_loss, cause_code=d.cause_code or "4",
                     flood_event=d.flood_event or "", water_depth_inches=d.water_depth_inches,
                     building_damage=d.building_damage,
                     contents_damage=d.contents_damage)  # fmt: skip


FLOOD_FIELDS = ("policyholder", "policy_start", "date_of_loss", "cause_code",
                "water_depth_inches", "building_damage", "contents_damage", "building_limit",
                "contents_limit", "building_deductible", "contents_deductible",
                "flood_zone")  # fmt: skip
_POLICY = ("building_limit", "contents_limit", "building_deductible", "contents_deductible",
           "flood_zone")  # fmt: skip


def flood_sources(record: FloodRecord | None, d: FloodStoryDetails | None) -> dict[str, Source]:
    """Where each flood detail comes from: the FEMA record, the story, a sample policy value, or
    nothing (the user enters it)."""
    if record is not None:
        out: dict[str, Source] = dict.fromkeys(FLOOD_FIELDS, "fema")
        out["policyholder"] = "needed"  # FEMA redacts names
        if record.water_depth_inches is None:
            out["water_depth_inches"] = "needed"
        if record.policy_start is None:
            out["policy_start"] = "needed"
        return out
    d = d or FloodStoryDetails()
    out = {}
    for f in FLOOD_FIELDS:
        if f in _POLICY:
            out[f] = "sample"
        elif f == "policy_start":
            out[f] = "needed"
        else:
            out[f] = "story" if getattr(d, f, None) not in (None, "") else "needed"
    if out["cause_code"] == "needed":
        out["cause_code"] = "estimated"  # defaults to heavy rain: check it
    return out
