"""Logic behind the "Try a claim" page (pure, tested; the Streamlit page only lays it out).

A real first notice of loss has two parts: what the policyholder says (the story) and what the
insurer already knows or asks on the form (who the policyholder is, the policy, the car, the loss
date, the repair estimate). The page takes the story verbatim from a real NHTSA complaint, or
pasted text, and the rest from the form. The story is never edited.
"""

import hashlib
import re
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from autoclaim.datasets.nhtsa_complaints import Complaint
from autoclaim.lines.auto.claim import Appraisal, ClaimPackage, PolicyRecord

Coverage = Literal["liability_only", "collision", "collision_comprehensive"]
# rough actual cash value by vehicle age and type, only as a form default (the user can change it)
NEW_PRICE = {"car": 30_000.0, "suv": 38_000.0, "pickup": 50_000.0}


class ClaimForm(BaseModel):
    policyholder: str = ""  # the name on the policy: never in the story, entered on the form
    listed_drivers: list[str] = []
    excluded_drivers: list[str] = []
    policy_state: str = "TX"
    policy_start_date: date
    coverage: Coverage = "collision_comprehensive"
    collision_deductible: float = 500.0
    comprehensive_deductible: float = 250.0
    rideshare_endorsement: bool = False
    vehicle_year: int
    vehicle_make: str
    vehicle_model: str
    body_class: str = "car"
    vehicle_acv: float = Field(gt=0)
    lienholder: str | None = None
    loss_date: date
    report_date: date
    estimate_amount: float = Field(gt=0)
    channel: Literal["phone", "web", "email", "app"] = "web"
    # the independent appraisal, when there is one: fraud triage then uses the stage-2 model
    appraised_amount: float | None = Field(default=None, gt=0)
    appraiser_prior_damage: bool = False


def acv_default(year: int, body: str, today: date) -> float:
    """~15% depreciation per year from a typical new price, floored at 15% of it."""
    new = NEW_PRICE.get(body, NEW_PRICE["car"])
    age = max(0, today.year - year)
    return round(max(new * 0.85**age, new * 0.15), -2)


def form_from_complaint(c: Complaint, today: date) -> ClaimForm:
    """Form defaults for a complaint: its car and loss date; reported to the insurer two days later
    (the NHTSA filing date is when the owner wrote to the regulator, not to the insurer)."""
    return ClaimForm(
        policy_start_date=date(c.date_of_incident.year - 1, 1, 15),
        vehicle_year=c.year,
        vehicle_make=c.make,
        vehicle_model=c.model,
        body_class=c.body_class,
        vehicle_acv=acv_default(c.year, c.body_class, c.date_of_incident),
        loss_date=c.date_of_incident,
        report_date=min(c.date_of_incident + timedelta(days=2), today),
        estimate_amount=4500.0,
    )


def statement(story: str, form: ClaimForm) -> str:
    """The form's facts as a short header (as an FNOL form records them), then the story as is."""
    v = f"{form.vehicle_year} {form.vehicle_make} {form.vehicle_model}"
    return (f"Reported by the policyholder, {form.policyholder}. Date of loss: {form.loss_date}. "
            f"Insured vehicle: {v}.\nStatement:\n{story.strip()}")  # fmt: skip


def claim_id(story: str, form: ClaimForm) -> str:
    """Same story and form -> same id (re-running returns the first decision, never pays twice);
    any change -> a new claim."""
    digest = hashlib.sha256((story.strip() + form.model_dump_json()).encode()).hexdigest()
    return f"TRY-{digest[:10].upper()}"


def build_package(story: str, form: ClaimForm) -> dict[str, Any]:
    policy = PolicyRecord.model_validate(
        {
            "policy_id": "POL-TRY",
            "policy_state": form.policy_state,
            "policy_start_date": form.policy_start_date,
            "coverage": form.coverage,
            "collision_deductible": None
            if form.coverage == "liability_only"
            else form.collision_deductible,
            "comprehensive_deductible": form.comprehensive_deductible
            if form.coverage == "collision_comprehensive"
            else None,
            "rideshare_endorsement": form.rideshare_endorsement,
            "named_insured": form.policyholder,
            "listed_drivers": form.listed_drivers,
            "excluded_drivers": form.excluded_drivers,
            "vehicle": {
                "year": form.vehicle_year,
                "make": form.vehicle_make,
                "model": form.vehicle_model,
                "body_class": form.body_class,
                "actual_cash_value": form.vehicle_acv,
                "adas": None,
                "lienholder": form.lienholder,
            },
            "prior_claims_3y": 0,
            "address_change_days": None,
        }
    )
    pkg = ClaimPackage(
        claim_id=claim_id(story, form),
        channel=form.channel,
        report_date=form.report_date,
        estimate_amount=form.estimate_amount,
        narrative=statement(story, form),
        policy=policy,
        appraisal=None if form.appraised_amount is None else Appraisal(
            appraised_amount=form.appraised_amount, prior_damage=form.appraiser_prior_damage),
    )  # fmt: skip
    return pkg.model_dump(mode="json")


def summarize(out: dict[str, Any]) -> dict[str, Any]:
    """What the page shows after a run: who decided, the decision, why, and what it cost."""
    decision = out.get("decision") or {}
    facts = (out.get("facts") or {}).get("derived") or {}
    base = {"explanation": decision.get("explanation"), "proposed": decision.get("outcome"),
            "coverage_part": facts.get("coverage_part"), "deductible": facts.get("deductible"),
            "llm_calls": out.get("llm_calls"), "tokens": out.get("tokens")}  # fmt: skip
    if "final" in out:
        f = out["final"]
        return base | {"status": "decided", "decided_by": f.get("decided_by"),
                       "outcome": f.get("outcome"), "payout": f.get("payout"),
                       "reasons": f.get("reasons") or [], "route_reasons": []}  # fmt: skip
    req = out["__interrupt__"][0].value
    prop = req.get("proposed_decision") or {}
    return base | {"status": "human_review", "decided_by": None, "outcome": None,
                   "payout": prop.get("payout"), "reasons": prop.get("reasons") or [],
                   "route_reasons": list(req.get("route_reasons") or [])}  # fmt: skip


ROLE_TEXT = {"named_insured": "the policyholder", "listed_driver": "a listed driver",
             "permissive_unlisted": "someone driving with permission",
             "excluded_driver": "a driver excluded by name", "unknown": "not clear"}  # fmt: skip


def _money(x: Any) -> str:
    return f"${float(x):,.2f}" if isinstance(x, int | float) else "-"


def trace(out: dict[str, Any], fraud_review_score: float) -> list[dict[str, str]]:
    """How the claim was decided, step by step, in plain English (status: ok / warn / bad)."""
    ext = (out.get("facts") or {}).get("extracted") or {}
    x = (out.get("facts") or {}).get("derived") or {}
    steps = []
    cause = str(ext.get("cause") or "unknown")
    steps.append({"title": "Read the story", "status": "ok" if cause != "unknown" else "warn",
                  "detail": f"Cause: {cause.replace('_', ' ')}; driver: "
                            f"{ext.get('driver_name') or 'not stated'}."})  # fmt: skip
    role = str(x.get("driver_role") or "unknown")
    steps.append({"title": "Checked the driver",
                  "status": {"excluded_driver": "bad", "unknown": "warn"}.get(role, "ok"),
                  "detail": f"On the policy as {ROLE_TEXT.get(role, role)}."})  # fmt: skip
    part = str(x.get("coverage_part") or "unknown")
    on = bool(x.get("part_on_policy"))
    carries = "carries" if on else "does not carry"
    steps.append({"title": "Matched the coverage", "status": "ok" if on else "bad",
                  "detail": f"A {part} loss; the policy {carries} {part} coverage."})  # fmt: skip
    sig = (out.get("fraud") or {}).get("signals") or {}
    score = (out.get("fraud") or {}).get("model_score")
    if sig.get("two_stage_referral") is not None:
        refer = bool(sig["two_stage_referral"])
        steps.append(
            {
                "title": "Screened for fraud",
                "status": "warn" if refer else "ok",
                "detail": f"With the independent appraisal: first-notice score "
                f"{score:.2f}, post-appraisal score {sig['stage2_score']:.2f} "
                f"({'referred to the fraud team' if refer else 'cleared'}).",
            }
        )
    elif isinstance(score, int | float):
        high = score >= fraud_review_score
        steps.append({"title": "Screened for fraud", "status": "warn" if high else "ok",
                      "detail": f"Fraud score {score:.2f} "
                                f"({'at or above' if high else 'below'} the review line "
                                f"{fraud_review_score:.2f})."})  # fmt: skip
    if x and not on:
        steps.append(
            {
                "title": "Worked out the payout",
                "status": "bad",
                "detail": "Nothing is payable: the policy does not cover this loss.",
            }
        )
    elif x:
        how = "car's value (total loss)" if x.get("total_loss") else "repair estimate"
        steps.append({"title": "Worked out the payout", "status": "ok",
                      "detail": f"{_money(x.get('gross_loss'))} {how} - "
                                f"{_money(x.get('deductible') or 0)} deductible = "
                                f"{_money(x.get('payout'))}."})  # fmt: skip
    critic, judge = out.get("critic") or {}, out.get("judge") or {}
    if critic or judge:
        ok = bool(critic.get("passed")) and bool(judge.get("passed"))
        retries = int(out.get("retries") or 0)
        rules = "passed" if critic.get("passed") else "flagged issues"
        reviewer = "agreed" if judge.get("passed") else "raised issues"
        redo = f" (after {retries} revision{'s' * (retries > 1)})" if retries else ""
        detail = f"Rule checks {rules}; independent reviewer {reviewer}{redo}."
        steps.append({"title": "Double-checked the decision", "status": "ok" if ok else "warn",
                      "detail": detail})  # fmt: skip
    if "final" in out:
        steps.append(
            {
                "title": "Decided automatically",
                "status": "ok",
                "detail": "Clear-cut and within the automatic authority limit.",
            }
        )
    elif "__interrupt__" in out:
        reasons = ", ".join(out["__interrupt__"][0].value.get("route_reasons") or [])
        steps.append(
            {
                "title": "Sent to a human adjuster",
                "status": "warn",
                "detail": f"Why: {reasons.replace('_', ' ') or 'needs judgment'}.",
            }
        )
    return steps


def verdict(summary: dict[str, Any]) -> tuple[str, str, str]:
    """(headline, sub-line, tone) for the result banner; tone is ok / bad / warn."""
    if summary["status"] != "decided":
        return ("Sent to an adjuster", "A person decides this one. Open the Adjuster console "
                "page to review it.", "warn")  # fmt: skip
    if summary["outcome"] == "approve":
        return (f"Approved: insurer pays {_money(summary['payout'])}",
                "Paid to the policyholder or the repair shop.", "ok")  # fmt: skip
    if summary["outcome"] == "deny":
        return ("Denied", "The policy does not cover this loss.", "bad")
    return ("Escalated", "Needs a person before any payment.", "warn")


def _norm(text: str) -> str:
    return " ".join(text.lower().replace(chr(0x2019), "'").split())  # curly apostrophes as straight


def match_complaint(text: str, complaints: list[Complaint]) -> Complaint | None:
    """The downloaded NHTSA story a pasted text comes from (whole or a 120+ character part), so the
    form gets that story's car and loss date instead of generic defaults."""
    t = _norm(text)
    if len(t) < 120:
        return None
    for c in complaints:
        s = _norm(c.summary)
        if t == s or t in s or (len(s) >= 120 and s in t):
            return c
    return None


def generic_form(today: date) -> ClaimForm:
    """Defaults for a story the page doesn't recognize."""
    return ClaimForm(policy_start_date=date(today.year - 1, 1, 15),
                     vehicle_year=2021, vehicle_make="Toyota", vehicle_model="Camry",
                     vehicle_acv=21_000.0, loss_date=today, report_date=today,
                     estimate_amount=4500.0)  # fmt: skip


REQUIRED = {"policyholder": "policyholder name", "estimate_amount": "repair estimate",
            "loss_date": "date of loss", "report_date": "reported date",
            "vehicle_year": "vehicle year", "vehicle_make": "make", "vehicle_model": "model",
            "vehicle_acv": "car's value"}  # fmt: skip


def missing_fields(values: dict[str, Any]) -> list[str]:
    """The claim details still empty (step 2 starts empty for a story the page doesn't know)."""
    return [label for key, label in REQUIRED.items()
            if values.get(key) is None or str(values.get(key)).strip() == ""]  # fmt: skip


# ---------------------------------------------------------------- claim details from a story

Body = Literal["car", "suv", "pickup", "van"]


class StoryDetails(BaseModel):
    """What a story says about the claim details (null = the story does not say)."""

    vehicle_year: int | None = None
    vehicle_make: str | None = None
    vehicle_model: str | None = None
    body_class: Body | None = None
    loss_date: date | None = None
    estimate_amount: float | None = None
    policyholder: str | None = None


EXTRACT = """From a policyholder's account of a car crash, copy only what it states about the
insured vehicle and the loss: model year, make and model (e.g. "my 2023 CX 5" -> 2023, Mazda, CX-5;
"toyota corporate" names the make), body type (car, suv, pickup or van) when the model makes it
clear, the date of the loss, a repair estimate or damage amount in dollars, and the writer's own
name if they give it. Use null for anything not stated. Never guess a date, a name or an amount.
Resolve relative dates ("yesterday") against the report date."""

MAKES = {"acura": "Acura", "audi": "Audi", "bmw": "BMW", "buick": "Buick", "cadillac": "Cadillac",
         "chevrolet": "Chevrolet", "chevy": "Chevrolet", "chrysler": "Chrysler", "dodge": "Dodge",
         "ford": "Ford", "gmc": "GMC", "honda": "Honda", "hyundai": "Hyundai", "jeep": "Jeep",
         "kia": "Kia", "lexus": "Lexus", "mazda": "Mazda", "mercedes": "Mercedes-Benz",
         "nissan": "Nissan", "ram": "Ram", "subaru": "Subaru", "tesla": "Tesla",
         "toyota": "Toyota", "volkswagen": "Volkswagen", "vw": "Volkswagen",
         "volvo": "Volvo"}  # fmt: skip
# model words -> (make, model, body): common US models a story may name without the make
MODELS = {"camry": ("Toyota", "Camry", "car"), "corolla": ("Toyota", "Corolla", "car"),
          "rav4": ("Toyota", "RAV4", "suv"), "tacoma": ("Toyota", "Tacoma", "pickup"),
          "tundra": ("Toyota", "Tundra", "pickup"), "highlander": ("Toyota", "Highlander", "suv"),
          "civic": ("Honda", "Civic", "car"), "accord": ("Honda", "Accord", "car"),
          "cr-v": ("Honda", "CR-V", "suv"), "crv": ("Honda", "CR-V", "suv"),
          "pilot": ("Honda", "Pilot", "suv"), "f-150": ("Ford", "F-150", "pickup"),
          "f150": ("Ford", "F-150", "pickup"), "escape": ("Ford", "Escape", "suv"),
          "explorer": ("Ford", "Explorer", "suv"),
          "silverado": ("Chevrolet", "Silverado", "pickup"),
          "equinox": ("Chevrolet", "Equinox", "suv"), "malibu": ("Chevrolet", "Malibu", "car"),
          "altima": ("Nissan", "Altima", "car"), "rogue": ("Nissan", "Rogue", "suv"),
          "sentra": ("Nissan", "Sentra", "car"), "elantra": ("Hyundai", "Elantra", "car"),
          "tucson": ("Hyundai", "Tucson", "suv"), "model 3": ("Tesla", "Model 3", "car"),
          "model y": ("Tesla", "Model Y", "suv"),
          "grand cherokee": ("Jeep", "Grand Cherokee", "suv"),
          "wrangler": ("Jeep", "Wrangler", "suv"), "outback": ("Subaru", "Outback", "suv"),
          "forester": ("Subaru", "Forester", "suv"), "cx-5": ("Mazda", "CX-5", "suv"),
          "cx 5": ("Mazda", "CX-5", "suv"), "cx5": ("Mazda", "CX-5", "suv"),
          "jetta": ("Volkswagen", "Jetta", "car"), "forte": ("Kia", "Forte", "car"),
          "sierra": ("GMC", "Sierra", "pickup")}  # fmt: skip
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}  # fmt: skip
_YEAR = r"(?:19[9]\d|20[0-4]\d)"
_DATE_NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DATE_TXT = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
                       re.IGNORECASE)  # fmt: skip
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d\d)?(?!\s*k)", re.IGNORECASE)


def _valid_date(y: int, m: int, d: int, today: date) -> date | None:
    try:
        out = date(y, m, d)
    except ValueError:
        return None
    return out if date(1990, 1, 1) <= out <= today else None


def details_from_text(story: str, today: date) -> StoryDetails:
    """Code-only reading (no model): car from known makes/models, dates, dollar amounts."""
    low = " ".join(story.lower().split())
    make = model = body = None
    for key in sorted(MODELS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", low):
            make, model, body = MODELS[key]
            break
    if make is None:
        hit = next((v for k, v in MAKES.items() if re.search(rf"\b{k}\b", low)), None)
        make = hit
    year_hit = re.search(rf"\b({_YEAR})\s+(?:[a-z-]+\s+)?(?:{'|'.join(map(re.escape, MAKES))}|"
                         rf"{'|'.join(map(re.escape, MODELS))})", low)  # fmt: skip
    loss = None
    for m in _DATE_NUM.finditer(story):
        loss = loss or _valid_date(int(m[3]), int(m[1]), int(m[2]), today)
    for m in _DATE_TXT.finditer(story):
        loss = loss or _valid_date(int(m[3]), MONTHS[m[1].lower()], int(m[2]), today)
    amounts = [float(m[1].replace(",", "")) for m in _MONEY.finditer(story)]
    return StoryDetails(vehicle_year=int(year_hit[1]) if year_hit else None, vehicle_make=make,
                        vehicle_model=model, body_class=body, loss_date=loss,  # type: ignore[arg-type]
                        estimate_amount=max(amounts) if amounts else None)  # fmt: skip


def details_from_llm(client: Any, story: str, today: date) -> StoryDetails | None:
    """The cheap intake model reads the story (cached; None if no model is available)."""
    try:
        res = client.structured("intake", EXTRACT, f"report_date: {today}\nstatement:\n{story}",
                                StoryDetails)  # fmt: skip
    except Exception:  # quota or provider outage: the code reading still applies
        return None
    return res.value  # type: ignore[no-any-return]


def read_story(story: str, today: date, client: Any | None = None) -> StoryDetails:
    """Model reading where available, code reading for anything it left empty; then sanity checks
    (no future dates or impossible years, positive amounts only)."""
    text = details_from_text(story, today)
    llm = details_from_llm(client, story, today) if client is not None else None
    merged = {k: (getattr(llm, k) if llm is not None and getattr(llm, k) not in (None, "")
                  else getattr(text, k)) for k in StoryDetails.model_fields}  # fmt: skip
    if merged["vehicle_year"] is not None and not 1990 <= merged["vehicle_year"] <= today.year + 1:
        merged["vehicle_year"] = None
    if merged["loss_date"] is not None and merged["loss_date"] > today:
        merged["loss_date"] = None
    if merged["estimate_amount"] is not None and merged["estimate_amount"] <= 0:
        merged["estimate_amount"] = None
    name = str(merged["policyholder"] or "").strip()
    if len([w for w in name.split() if w.isalpha()]) < 2:  # a sign-off like "- toy" is not a name
        merged["policyholder"] = None
    return StoryDetails.model_validate(merged)


def defaults_from_details(d: StoryDetails, today: date) -> dict[str, Any]:
    """Step-2 starting values from what the story says; None where it says nothing."""
    body = d.body_class or "car"
    return {"policyholder": d.policyholder or "", "vehicle_year": d.vehicle_year,
            "vehicle_make": d.vehicle_make or "", "vehicle_model": d.vehicle_model or "",
            "body_class": body,
            "vehicle_acv": acv_default(d.vehicle_year, body, d.loss_date or today)
            if d.vehicle_year else None,
            "loss_date": d.loss_date,
            "report_date": min(d.loss_date + timedelta(days=2), today) if d.loss_date else None,
            "estimate_amount": d.estimate_amount}  # fmt: skip


FIELD_LABELS = {"vehicle_year": "year", "vehicle_make": "make", "vehicle_model": "model",
                "loss_date": "date of loss", "estimate_amount": "repair estimate",
                "policyholder": "policyholder name"}  # fmt: skip


def found_and_missing(d: StoryDetails) -> tuple[list[str], list[str]]:
    """Which details the story gave and which the user still has to enter."""
    found = [lab for k, lab in FIELD_LABELS.items() if getattr(d, k) not in (None, "")]
    return found, [lab for k, lab in FIELD_LABELS.items() if getattr(d, k) in (None, "")]


# ---------------------------------------------------------------- autofill status per field

Source = Literal["record", "story", "estimated", "sample", "needed"]
SOURCE_TAG = {
    "record": ":green[● from the NHTSA record]",
    "story": ":green[● from the story]",
    "estimated": ":blue[● estimated, check it]",
    "sample": ":gray[● sample policy]",
    "needed": ":orange[● please enter]",
}
AUTOFILL_FIELDS = ("estimate_amount", "loss_date", "report_date", "vehicle_year", "vehicle_make",
                   "vehicle_model", "body_class", "vehicle_acv", "policyholder")  # fmt: skip
_RECORD = ("loss_date", "vehicle_year", "vehicle_make", "vehicle_model", "body_class")


def field_sources(known: bool, details: StoryDetails | None) -> dict[str, Source]:
    """Where each claim detail's starting value comes from, like an autofill form shows it: the
    NHTSA record, the story, an estimate to check, or nothing (the user enters it)."""
    if known:
        return {f: ("record" if f in _RECORD else "estimated"
                    if f in ("report_date", "vehicle_acv") else "needed")
                for f in AUTOFILL_FIELDS}  # fmt: skip
    d = details or StoryDetails()
    out: dict[str, Source] = {}
    for f in AUTOFILL_FIELDS:
        if f == "report_date":
            out[f] = "estimated" if d.loss_date else "needed"
        elif f == "vehicle_acv":
            out[f] = "estimated" if d.vehicle_year else "needed"
        elif f == "body_class":
            out[f] = "story" if d.body_class else "estimated"
        else:
            out[f] = "story" if getattr(d, f) not in (None, "") else "needed"
    return out


def tagged(label: str, source: Source) -> str:
    """A field label with its autofill tag (Streamlit colours the tag)."""
    return f"{label} {SOURCE_TAG[source]}"


def autofill_progress(sources: dict[str, Source]) -> tuple[int, int]:
    """(details filled automatically, details in total)."""
    return sum(s != "needed" for s in sources.values()), len(sources)
