"""Try a claim: story -> claim details -> decision, on a real crash story or your own text.

Run:  uv run streamlit run src/autoclaim/ui/console.py   (this page is in the sidebar)
Stories: uv run python scripts/download_data.py nhtsa_complaints
Demo photos (optional): uv run python scripts/fetch_demo_images.py

Each run makes ~4-6 free-tier LLM calls (cached: the same story and details never pay twice). A
claim the harness can't decide alone waits on the Adjuster console page.
"""

import html
import random
from datetime import date
from typing import Any

import streamlit as st
from pydantic import ValidationError

from autoclaim.datasets import nhtsa_complaints as nc
from autoclaim.paths import data_dir, raw_dir
from autoclaim.ui.assets import (
    ANIMATION_CSS,
    BRAND,
    credit_html,
    demo_photos,
    deny_mark,
    icon,
    illustration,
    loader_html,
    photo_for,
    photo_src,
    review_alert,
    success_tick,
)
from autoclaim.ui.review_view import audit_rows
from autoclaim.ui.shared import harness, llm_client
from autoclaim.ui.try_claim_view import (
    ClaimForm,
    StoryDetails,
    autofill_progress,
    build_package,
    defaults_from_details,
    field_sources,
    form_from_complaint,
    generic_form,
    match_complaint,
    missing_fields,
    read_story,
    summarize,
    tagged,
    trace,
    verdict,
)

BODIES = ["car", "suv", "pickup", "van"]
COVERAGES = {
    "Collision + comprehensive": "collision_comprehensive",
    "Collision only": "collision",
    "Liability only (no own-car cover)": "liability_only",
}
STEP_ICON = {"ok": "circle-check", "warn": "triangle-alert", "bad": "circle-x"}
CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1180px;}
.hero {background: linear-gradient(120deg,#1e3a8a,#2563eb 60%,#3b82f6); color: #fff;
       border-radius: 16px; padding: 22px 28px; margin-bottom: 14px; display: flex; gap: 18px;
       align-items: center;}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: 6px 0 0; opacity: .9;}
.badge {background: rgba(255,255,255,.16); border-radius: 14px; padding: 12px; display: flex;}
.stepper {display: flex; gap: 10px; margin: 4px 0 18px;}
.stp {flex: 1; display: flex; gap: 10px; align-items: center; background: #fff;
      border-radius: 12px; padding: 10px 14px; color: #6b7280;
      box-shadow: 0 1px 2px rgba(0,0,0,.05);}
.stp .n {flex: 0 0 28px; height: 28px; border-radius: 50%; background: #e5e7eb; color: #6b7280;
         display: flex; align-items: center; justify-content: center; font-weight: 700;}
.stp.active {color: #1e3a8a; outline: 2px solid #2563eb;}
.stp.active .n {background: #2563eb; color: #fff;}
.stp.done {color: #047857;} .stp.done .n {background: #10b981; color: #fff;}
.sec {font-size: 1.15rem; font-weight: 700; color: #1f2937; margin: 6px 0 8px;}
.story {background: #fff; border-left: 5px solid #2563eb; border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.06); font-size: 1.02rem;
        line-height: 1.55;}
.chips {margin: 10px 0 4px;}
.chip {display: inline-flex; gap: 6px; align-items: center; background: #eef2ff; color: #1e3a8a;
       border-radius: 999px; padding: 4px 11px; margin: 0 6px 6px 0; font-size: .85rem;}
.photo {background: #fff; border-radius: 12px; padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.06);}
.photo img {width: 100%; height: 210px; object-fit: cover; border-radius: 8px; display: block;}
.photo .credit {font-size: .78rem; color: #6b7280; padding: 6px 4px 0;}
.verdict {border-radius: 16px; padding: 18px 24px; margin: 8px 0 16px; color: #fff;
          display: flex; gap: 16px; align-items: center;}
.verdict h2 {margin: 0; color: #fff; font-size: 1.6rem;}
.verdict p {margin: 4px 0 0; opacity: .92;}
.tone-ok {background: linear-gradient(120deg,#047857,#10b981);}
.tone-bad {background: linear-gradient(120deg,#b91c1c,#ef4444);}
.tone-warn {background: linear-gradient(120deg,#b45309,#f59e0b);}
.step {display: flex; gap: 14px; align-items: flex-start; background: #fff; border-radius: 12px;
       padding: 12px 16px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,.05);}
.dot {flex: 0 0 32px; height: 32px; border-radius: 50%; display: flex; align-items: center;
      justify-content: center;}
.dot-ok {background: #10b981;} .dot-warn {background: #f59e0b;} .dot-bad {background: #ef4444;}
.step b {display: block;} .step span {color: #4b5563; font-size: .93rem;}
.card {background: #fff; border-radius: 14px; padding: 18px 22px; color: #374151;
       box-shadow: 0 1px 2px rgba(0,0,0,.05);}
.center {text-align: center;}
</style>
"""


def names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def chips(items: list[tuple[str, str]]) -> str:
    """Pill labels, each with a Lucide icon: [(icon name, text), ...]."""
    spans = "".join(f'<span class="chip">{icon(i, 14)}{html.escape(t)}</span>' for i, t in items)
    return f'<div class="chips">{spans}</div>'


def show(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section(name: str, title: str, color: str = "currentColor") -> None:
    show(f'<div class="sec">{icon(name, 20, color)} {html.escape(title)}</div>')


st.set_page_config(page_title="Try a claim · autoclaim", page_icon=str(BRAND / "favicon.png"),
                   layout="wide")  # fmt: skip
st.logo(str(BRAND / "logo.svg"), icon_image=str(BRAND / "favicon.svg"))
show(CSS + ANIMATION_CSS)
show(f'<div class="hero"><div class="badge">{icon("car", 34, "#ffffff")}</div><div>'
     "<h1>Try a claim</h1><p>Start with the story: a real crash from NHTSA's public complaint "
     "database (owners' own words) or your own. Then confirm the claim details and see how the "
     "harness decides it, step by step.</p></div></div>")  # fmt: skip

complaints = nc.load(raw_dir("nhtsa_complaints") / nc.FILE)
by_id = {c.odi_number: c for c in complaints}
photos = demo_photos()
today = date.today()
ss = st.session_state
ss.setdefault("story", None)  # the claim story, once chosen
ss.setdefault("story_odi", None)  # its NHTSA complaint number, if it is one
ss.setdefault("result", None)
ss.setdefault("story_details", None)  # what an unrecognized story says (claim details)


def set_story(text: str, odi: int | None, details: dict[str, object] | None = None) -> None:
    ss.story, ss.story_odi, ss.result, ss.story_details = text, odi, None, details
    ss.autofill_cleared = False


def clear_story() -> None:
    ss.story, ss.story_odi, ss.result, ss.story_details = None, None, None, None
    ss.autofill_cleared = False


def stepper(active: int) -> None:
    cells = []
    for i, label in enumerate(["Story", "Claim details", "Decision"], 1):
        cls = "done" if i < active else "active" if i == active else ""
        mark = icon("circle-check", 16, "#ffffff", 3) if i < active else str(i)
        cells.append(f'<div class="stp {cls}"><span class="n">{mark}</span>{label}</div>')
    show(f'<div class="stepper">{"".join(cells)}</div>')


stepper(1 if ss.story is None else 3 if ss.result else 2)

# ---------------------------------------------------------------- 1. story
if ss.story is None:
    section("file-text", "1. The claim story")
    source = st.radio("Where does the story come from?",
                      ["Real NHTSA story", "Write or paste my own"], horizontal=True)  # fmt: skip
    if source == "Real NHTSA story":
        if not complaints:
            st.warning("No stories downloaded yet: "
                       "`uv run python scripts/download_data.py nhtsa_complaints`")  # fmt: skip
            st.stop()
        f1, f2 = st.columns([2, 1])
        query = f1.text_input("Search the stories", placeholder="deer, rear ended, hydrant")
        make = f2.selectbox("Make", ["All makes", *sorted({c.make for c in complaints})])
        pool = [c for c in complaints if make in ("All makes", c.make)
                and query.lower() in c.summary.lower()]  # fmt: skip
        if not pool:
            st.info("No stories match. Try another word.")
            st.stop()
        labels = {f"{c.year} {c.make} {c.model} · {c.date_of_incident:%b %d, %Y} · "
                  f"{c.summary[:70]}…": c for c in pool}  # fmt: skip
        keys = list(labels)
        if st.button("Random story"):
            ss["story_pick"] = random.choice(keys)
        if ss.get("story_pick") not in keys:
            ss["story_pick"] = keys[0]
        pick = labels[st.selectbox(f"{len(pool)} stories", keys, key="story_pick")]
        show(f'<div class="story">“{html.escape(pick.summary)}”</div>')
        st.button("Use this story", type="primary", on_click=set_story,
                  args=(pick.summary, pick.odi_number))  # fmt: skip
    else:
        text = st.text_area("What happened, in the policyholder's words", height=220,
                            placeholder="I was backing out of my driveway when…")  # fmt: skip
        if st.button("Use this story", type="primary"):
            if len(text.strip()) < 30:
                st.error("The story needs at least 30 characters.")
            else:
                match = match_complaint(text, complaints)
                details = None
                if match is None:
                    with st.spinner("Reading the story for the claim details…"):
                        details = read_story(text, today, llm_client()).model_dump(mode="json")
                set_story(text.strip(), match.odi_number if match else None, details)
                st.rerun()
    show(f'<div class="card center" style="margin-top:14px">{illustration("onboarding", 420)}'
         "<p><b>Story, then details, then decision.</b> The harness reads the story, checks the "
         "driver and the policy, screens for fraud, works out the payout and double-checks "
         "itself before anything is paid.</p></div>")  # fmt: skip
    st.stop()

known = by_id.get(ss.story_odi) if ss.story_odi else None
h1, h2 = st.columns([5, 1])
with h1:
    section("circle-check", "1. The claim story", "#10b981")
with h2:
    st.button("Change story", on_click=clear_story, use_container_width=True)
photo = photo_for(known.odi_number, photos) if known else None
story_col, photo_col = st.columns([3, 2], gap="medium") if photo else (st.container(), None)
with story_col:
    show(f'<div class="story">“{html.escape(ss.story)}”</div>')
    if known:
        show(chips([("car", f"{known.year} {known.make} {known.model}"),
                    ("calendar", f"Loss {known.date_of_incident:%b %d, %Y}"),
                    ("user", f"{known.injuries} injured"),
                    ("file-text", f"NHTSA #{known.odi_number}")]))  # fmt: skip
if photo is not None and photo_col is not None:
    with photo_col:
        show(f'<div class="photo"><img src="{photo_src(photo)}" '
             f'alt="{html.escape(photo["alt"] or "car damage")}">'
             f'<div class="credit">{icon("camera", 12)} Demo photo, not from this claim. '
             f"{credit_html(photo)}</div></div>")  # fmt: skip

# ---------------------------------------------------------------- 2. claim details
section("clipboard-list", "2. Claim details")
ss.setdefault("autofill_cleared", False)
d = form_from_complaint(known, today) if known else None
story_details = StoryDetails.model_validate(ss.story_details) if ss.story_details else None
if d is not None:
    start_vals: dict[str, object] = {
        "estimate_amount": None, "loss_date": d.loss_date, "report_date": d.report_date,
        "vehicle_year": d.vehicle_year, "vehicle_make": d.vehicle_make,
        "vehicle_model": d.vehicle_model, "body_class": d.body_class,
        "vehicle_acv": d.vehicle_acv, "policyholder": ""}  # fmt: skip
else:
    start_vals = defaults_from_details(story_details or StoryDetails(), today)
sources = field_sources(known is not None, story_details)
if ss.autofill_cleared:
    start_vals = {k: ("" if isinstance(v, str) else None) for k, v in start_vals.items()}
    start_vals["body_class"] = "car"
    sources = {k: "needed" for k in sources}
filled, total = autofill_progress(sources)
a1, a2, a3 = st.columns([4, 1, 1])
with a1:
    origin = f"NHTSA complaint #{known.odi_number}" if known else "the story"
    st.progress(filled / total, text=f"Autofilled {filled} of {total} details from {origin} · "
                                     f"{total - filled} for you to enter")  # fmt: skip
with a2:
    if st.button("Autofill again", use_container_width=True, disabled=known is not None,
                 help="Read the story again (new stories only)"):  # fmt: skip
        with st.spinner("Reading the story…"):
            ss.story_details = read_story(ss.story, today, llm_client()).model_dump(mode="json")
        ss.autofill_cleared = False
        st.rerun()
with a3:
    if st.button("Clear autofill", use_container_width=True):
        ss.autofill_cleared = not ss.autofill_cleared
        st.rerun()
st.caption("Every field can be edited. Green came from the source, blue is an estimate to check, "
           "orange is yours to enter. The policy is a sample: a story never contains it (an "
           "insurer looks it up by policy number).")  # fmt: skip
base = d or generic_form(today)


def start(field: str) -> Any:
    return start_vals.get(field)


with st.form("claim", border=True):
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        show(f"<b>{icon('banknote', 16)} Loss</b>")
        estimate = st.number_input(tagged("Repair estimate ($)", sources["estimate_amount"]),
                                   1.0, 500_000.0, start("estimate_amount"), step=100.0,
                                   placeholder="from the body shop")  # fmt: skip
        loss = st.date_input(tagged("Date of loss", sources["loss_date"]), start("loss_date"))
        report = st.date_input(tagged("Reported to insurer", sources["report_date"]),
                               start("report_date"),
                               help="Over 30 days after the loss needs an adjuster")  # fmt: skip
        channel = st.selectbox("Reported by", ["web", "phone", "email", "app"])
        appraisal = st.number_input(
            "Independent appraisal ($) :gray[● optional]",
            1.0,
            500_000.0,
            None,
            step=100.0,
            placeholder="if an appraiser inspected it",
            help="The insurer's own appraiser's figure. When given, the "
            "stronger post-appraisal fraud model screens the claim",
        )
        prior = st.checkbox("Appraiser found prior damage")
    with c2:
        show(f"<b>{icon('car', 16)} Vehicle</b>")
        year = st.number_input(tagged("Year", sources["vehicle_year"]), 1990, today.year + 1,
                               start("vehicle_year"), placeholder="e.g. 2021")  # fmt: skip
        vmake = st.text_input(tagged("Make", sources["vehicle_make"]),
                              start("vehicle_make") or "", placeholder="e.g. Toyota")  # fmt: skip
        vmodel = st.text_input(tagged("Model", sources["vehicle_model"]),
                               start("vehicle_model") or "", placeholder="e.g. RAV4")  # fmt: skip
        body = st.selectbox(tagged("Body", sources["body_class"]), BODIES,
                            index=BODIES.index(start("body_class") or "car"))  # fmt: skip
        acv = st.number_input(tagged("Car's current value ($)", sources["vehicle_acv"]), 500.0,
                              500_000.0, start("vehicle_acv"), step=500.0,
                              placeholder="actual cash value",
                              help="The most a total loss pays")  # fmt: skip
    with c3:
        show(f"<b>{icon('shield-check', 16)} Policy</b>")
        holder = st.text_input(tagged("Policyholder (name on the policy)", sources["policyholder"]),
                               start("policyholder") or "", placeholder="e.g. Maria Lopez",
                               help="Who the policy belongs to. If the story says 'I was "
                               "driving', this person is the driver")  # fmt: skip
        coverage = COVERAGES[st.selectbox(tagged("Coverage", "sample"), list(COVERAGES))]
        p1, p2 = st.columns(2)
        coll = p1.number_input(tagged("Collision deductible", "sample"), 0.0, 5000.0,
                               base.collision_deductible, step=50.0)  # fmt: skip
        comp = p2.number_input(tagged("Comprehensive deductible", "sample"), 0.0, 5000.0,
                               base.comprehensive_deductible, step=50.0)  # fmt: skip
        listed = st.text_input("Other listed drivers", "", placeholder="comma-separated")
        excluded = st.text_input("Excluded drivers", "", placeholder="comma-separated")
        p3, p4 = st.columns(2)
        state = p3.text_input(tagged("State", "sample"), base.policy_state)
        start_date = p4.date_input(tagged("Policy start", "sample"), base.policy_start_date)
        rideshare = st.checkbox("Rideshare endorsement")
    go = st.form_submit_button("3. Decide this claim", type="primary", use_container_width=True)

if go:
    values = {"policyholder": holder.strip(), "listed_drivers": names(listed),
              "excluded_drivers": names(excluded), "policy_state": state,
              "policy_start_date": start_date, "coverage": coverage, "collision_deductible": coll,
              "comprehensive_deductible": comp, "rideshare_endorsement": rideshare,
              "vehicle_year": year, "vehicle_make": vmake.strip(),
              "vehicle_model": vmodel.strip(), "body_class": body, "vehicle_acv": acv,
              "loss_date": loss, "report_date": report, "estimate_amount": estimate,
              "channel": channel, "appraised_amount": appraisal,
              "appraiser_prior_damage": prior}  # fmt: skip
    gaps = missing_fields(values)
    if gaps:
        st.error("Fill in: " + ", ".join(gaps) + ".")
        st.stop()
    try:
        form = ClaimForm.model_validate(values)
    except ValidationError as exc:
        err = exc.errors()[0]
        st.error(f"Check the details: {err['loc'][0]}: {err['msg']}")
        st.stop()
    pkg = build_package(ss.story, form)
    working = st.empty()
    working.markdown(loader_html("Deciding the claim…", "Reading the story, checking the "
                                 "policy, screening for fraud, double-checking"),
                     unsafe_allow_html=True)  # fmt: skip
    try:
        app = harness()
        config = {"configurable": {"thread_id": pkg["claim_id"]}}
        out = app.graph.invoke({"claim_id": pkg["claim_id"], "claim": pkg}, config)
    except Exception as exc:  # any failure is shown, never a half-decided claim
        working.empty()
        show(f'<div class="card center">{illustration("error", 300)}<h4>Something went wrong'
             f"</h4><p>Nothing was paid or decided. {html.escape(type(exc).__name__)}: "
             f"{html.escape(str(exc)[:200])}</p></div>")  # fmt: skip
        st.stop()
    working.empty()
    ss.result = {"out": out, "pkg": pkg, "review": app.line.fraud_review_score}
    st.rerun()

# ---------------------------------------------------------------- 3. decision
if ss.result:
    out, pkg = ss.result["out"], ss.result["pkg"]
    section("scale", "3. Decision")
    r = summarize(out)
    head, sub, tone = verdict(r)
    mark = {"ok": success_tick(), "warn": review_alert(), "bad": deny_mark()}[tone]
    show(f'<div class="verdict tone-{tone}">{mark}<div><h2>{html.escape(head)}</h2>'
         f"<p>{html.escape(sub)}</p></div></div>")  # fmt: skip
    steps_col, info_col = st.columns([3, 2], gap="large")
    with steps_col:
        show(f"<b>{icon('signpost', 18)} How it was decided</b>")
        for s in trace(out, ss.result["review"]):
            show(f'<div class="step"><div class="dot dot-{s["status"]}">'
                 f"{icon(STEP_ICON[s['status']], 18, '#ffffff', 2.6)}</div><div>"
                 f'<b>{html.escape(s["title"])}</b><span>{html.escape(s["detail"])}</span>'
                 "</div></div>")  # fmt: skip
    with info_col:
        show(f"<b>{icon('file-text', 18)} Letter to the policyholder</b>")
        show(f'<div class="card">{html.escape(r["explanation"] or "-")}</div>')
        show(chips([("scale", f"Reasons: {', '.join(r['reasons']) or '-'}"),
                    ("sparkles", f"{r['llm_calls'] or 0} LLM calls"),
                    ("calculator", f"{r['tokens'] or 0:,} tokens"),
                    ("banknote", "Cost $0")]))  # fmt: skip
    with st.expander("Audit trail: every step, model and token count"):
        st.dataframe(audit_rows(data_dir() / "audit", pkg["claim_id"]), hide_index=True,
                     use_container_width=True)  # fmt: skip
    with st.expander(f"Raw claim sent to the harness ({pkg['claim_id']})"):
        st.json(pkg)
