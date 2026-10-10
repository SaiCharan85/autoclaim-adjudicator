"""Flood claim: story -> claim details -> decision, on a real FEMA NFIP claim or your own story.

Run:  uv run streamlit run src/autoclaim/ui/app.py   (Flood section in the sidebar)

The flood line runs on the same core harness as auto and decides with code only. Reading your own
story into the form makes one cached call on the intake model (with a code fallback).
"""

import html
import random
from datetime import date
from typing import Any

import streamlit as st
from pydantic import ValidationError

from autoclaim.datasets import fema_nfip
from autoclaim.lines.flood.build import build_flood
from autoclaim.paths import REPO_ROOT, raw_dir
from autoclaim.ui.assets import (
    ANIMATION_CSS,
    BRAND,
    PAGE_CSS,
    deny_mark,
    icon,
    illustration,
    review_alert,
    success_tick,
)
from autoclaim.ui.flood_view import (
    CAUSES,
    DEDUCTIBLES,
    ZONES,
    FloodForm,
    FloodRecord,
    FloodStoryDetails,
    flood_sources,
    flood_summary,
    flood_trace,
    form_from_record,
    form_from_story,
    missing_fields,
    read_flood_story,
    record_summary,
    records_from_frame,
    to_claim,
)
from autoclaim.ui.shared import llm_client
from autoclaim.ui.try_claim_view import autofill_progress, tagged, verdict

STEP_ICON = {"ok": "circle-check", "warn": "triangle-alert", "bad": "circle-x"}


def show(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section(name: str, title: str, color: str = "currentColor") -> None:
    show(f'<div class="sec">{icon(name, 20, color)} {html.escape(title)}</div>')


@st.cache_resource
def flood_graph() -> Any:
    state = REPO_ROOT / ".cache" / "flood_ui"
    state.mkdir(parents=True, exist_ok=True)
    return build_flood(state_dir=state)[1]  # template explanations: no LLM calls


@st.cache_data
def fema_records() -> list[FloodRecord]:
    path = raw_dir("fema_nfip")
    if not (path / "claims.csv").exists():
        return []
    return records_from_frame(fema_nfip.load(path))


st.set_page_config(page_title="Flood claim · autoclaim", page_icon=str(BRAND / "favicon.png"),
                   layout="wide")  # fmt: skip
show(PAGE_CSS + ANIMATION_CSS)
show(f'<div class="hero"><div class="badge">{icon("shield-check", 34, "#ffffff")}</div><div>'
     "<h1>Flood claim</h1><p>Start with the claim: a real FEMA flood insurance claim or your own "
     "story. Then confirm the details and see how the flood line decides it: building and "
     "contents coverage, each with its own limit and deductible.</p></div></div>")  # fmt: skip

records = fema_records()
by_id = {r.id: r for r in records}
today = date.today()
ss = st.session_state
for key in ("f_story", "f_record", "f_details", "f_result"):
    ss.setdefault(key, None)
ss.setdefault("f_cleared", False)


def set_claim(story: str, record: str | None, details: dict[str, Any] | None) -> None:
    ss.f_story, ss.f_record, ss.f_details, ss.f_result, ss.f_cleared = (
        story, record, details, None, False)  # fmt: skip


def clear_claim() -> None:
    ss.f_story = ss.f_record = ss.f_details = ss.f_result = None
    ss.f_cleared = False


def stepper(active: int) -> None:
    cells = []
    for i, label in enumerate(["Story", "Claim details", "Decision"], 1):
        cls = "done" if i < active else "active" if i == active else ""
        mark = icon("circle-check", 16, "#ffffff", 3) if i < active else str(i)
        cells.append(f'<div class="stp {cls}"><span class="n">{mark}</span>{label}</div>')
    show(f'<div class="stepper">{"".join(cells)}</div>')


stepper(1 if ss.f_story is None else 3 if ss.f_result else 2)

# ---------------------------------------------------------------- 1. story
if ss.f_story is None:
    section("file-text", "1. The flood claim")
    source = st.radio("Where does the claim come from?",
                      ["Real FEMA claim", "Write or paste my own"], horizontal=True)  # fmt: skip
    if source == "Real FEMA claim":
        if not records:
            st.warning("No FEMA claims downloaded yet: "
                       "`uv run python scripts/download_data.py fema_nfip`")  # fmt: skip
            st.stop()
        f1, f2 = st.columns([2, 1])
        query = f1.text_input("Search", placeholder="Helene, Ian, FL, rainfall, river")
        states = ["All states", *sorted({r.state for r in records})]
        state = f2.selectbox("State", states)
        pool = [r for r in records if state in ("All states", r.state)
                and query.lower() in record_summary(r).lower()]  # fmt: skip
        if not pool:
            st.info("No claims match. Try another word.")
            st.stop()
        labels = {f"{r.date_of_loss:%b %d, %Y} · {r.flood_event} · {r.state} · "
                  f"${r.building_damage or 0:,.0f}": r for r in pool}  # fmt: skip
        keys = list(labels)
        if st.button("Random claim"):
            ss["flood_pick"] = random.choice(keys)
        if ss.get("flood_pick") not in keys:
            ss["flood_pick"] = keys[0]
        pick = labels[st.selectbox(f"{len(pool)} real claims", keys, key="flood_pick")]
        show(f'<div class="story">{html.escape(record_summary(pick))}</div>')
        st.caption(fema_nfip.DISCLAIMER)
        st.button("Use this claim", type="primary", on_click=set_claim,
                  args=(record_summary(pick), pick.id, None))  # fmt: skip
    else:
        text = st.text_area("What happened, in the homeowner's words", height=220,
                            placeholder="On May 18 a flash flood filled our house…")  # fmt: skip
        if st.button("Use this story", type="primary"):
            if len(text.strip()) < 30:
                st.error("The story needs at least 30 characters.")
            else:
                with st.spinner("Reading the story for the claim details…"):
                    found = read_flood_story(text, today, llm_client())
                set_claim(text.strip(), None, found.model_dump(mode="json"))
                st.rerun()
    show(f'<div class="card center" style="margin-top:14px">{illustration("onboarding", 420)}'
         "<p><b>Claim, then details, then decision.</b> The flood line checks the cause and the "
         "policy dates, applies each deductible and limit, and sends anything that needs "
         "judgment, or more than $50,000, to an adjuster.</p></div>")  # fmt: skip
    st.stop()

record = by_id.get(ss.f_record) if ss.f_record else None
h1, h2 = st.columns([5, 1])
with h1:
    section("circle-check", "1. The flood claim", "#10b981")
with h2:
    st.button("Change claim", on_click=clear_claim, width="stretch")
show(f'<div class="story">{"" if record else "“"}{html.escape(ss.f_story)}'
     f'{"" if record else "”"}</div>')  # fmt: skip
if record:
    st.caption(f"FEMA NFIP claim, redacted (no names or exact location). {fema_nfip.DISCLAIMER}")

# ---------------------------------------------------------------- 2. claim details
section("clipboard-list", "2. Claim details")
details = FloodStoryDetails.model_validate(ss.f_details) if ss.f_details else None
start = form_from_record(record) if record else form_from_story(details or FloodStoryDetails())
sources = flood_sources(record, details)
if ss.f_cleared:
    start = FloodForm(building_limit=start.building_limit, contents_limit=start.contents_limit)
    sources = {k: ("sample" if v == "sample" else "needed") for k, v in sources.items()}
filled, total = autofill_progress(sources)
a1, a2, a3 = st.columns([4, 1, 1])
with a1:
    origin = "the FEMA record" if record else "the story"
    st.progress(filled / total, text=f"Autofilled {filled} of {total} details from {origin} · "
                                     f"{total - filled} for you to enter")  # fmt: skip
with a2:
    if st.button("Autofill again", width="stretch", disabled=record is not None,
                 help="Read the story again (own stories only)"):  # fmt: skip
        with st.spinner("Reading the story…"):
            ss.f_details = read_flood_story(ss.f_story, today, llm_client()).model_dump(mode="json")
        ss.f_cleared = False
        st.rerun()
with a3:
    if st.button("Clear autofill", width="stretch"):
        ss.f_cleared = not ss.f_cleared
        st.rerun()
st.caption("Every field can be edited. Green came from the source, blue is an estimate to check, "
           "grey is a sample policy value, orange is yours to enter.")  # fmt: skip
zones = ZONES if start.flood_zone in ZONES else [start.flood_zone, *ZONES]
causes = list(CAUSES)

with st.form("flood", border=True):
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        show(f"<b>{icon('shield-check', 16)} Policy</b>")
        holder = st.text_input(tagged("Policyholder", sources["policyholder"]), start.policyholder,
                               placeholder="e.g. Priya Raman")  # fmt: skip
        address = st.text_input("Property address", start.address,
                                placeholder="street, city, state")  # fmt: skip
        pstart = st.date_input(tagged("Policy start", sources["policy_start"]), start.policy_start)
        zone = st.selectbox(tagged("Flood zone", sources["flood_zone"]), zones,
                            index=zones.index(start.flood_zone),
                            help="A and V zones: high risk; X, B, C: moderate or low")  # fmt: skip
        primary = st.checkbox("Primary residence", start.primary_residence)
    with c2:
        show(f"<b>{icon('banknote', 16)} Limits and deductibles</b>")
        b_lim = st.number_input(tagged("Building limit ($)", sources["building_limit"]), 0.0,
                                500_000.0, start.building_limit, step=5_000.0)  # fmt: skip
        c_lim = st.number_input(tagged("Contents limit ($)", sources["contents_limit"]), 0.0,
                                500_000.0, start.contents_limit, step=5_000.0)  # fmt: skip
        b_ded = st.selectbox(tagged("Building deductible ($)", sources["building_deductible"]),
                             DEDUCTIBLES, index=DEDUCTIBLES.index(start.building_deductible),
                             format_func=lambda x: f"{x:,.0f}")  # fmt: skip
        c_ded = st.selectbox(tagged("Contents deductible ($)", sources["contents_deductible"]),
                             DEDUCTIBLES, index=DEDUCTIBLES.index(start.contents_deductible),
                             format_func=lambda x: f"{x:,.0f}")  # fmt: skip
    with c3:
        show(f"<b>{icon('calendar', 16)} The flood and the damage</b>")
        loss = st.date_input(tagged("Date of loss", sources["date_of_loss"]), start.date_of_loss)
        cause = st.selectbox(tagged("Cause", sources["cause_code"]), causes,
                             index=causes.index(start.cause_code),
                             format_func=lambda k: CAUSES[k])  # fmt: skip
        depth = st.number_input(tagged("Water depth inside (inches)",
                                       sources["water_depth_inches"]), 0.0, 400.0,
                                start.water_depth_inches, step=1.0)  # fmt: skip
        b_dmg = st.number_input(tagged("Building damage ($)", sources["building_damage"]), 0.0,
                                5_000_000.0, start.building_damage, step=500.0,
                                placeholder="adjuster's documented damage")  # fmt: skip
        c_dmg = st.number_input(tagged("Contents damage ($)", sources["contents_damage"]), 0.0,
                                5_000_000.0, start.contents_damage, step=500.0,
                                placeholder="adjuster's documented damage")  # fmt: skip
    go = st.form_submit_button("3. Decide this flood claim", type="primary", width="stretch")

if go:
    try:
        form = FloodForm(policyholder=holder.strip(), address=address.strip(),
                         policy_start=pstart, flood_zone=zone, primary_residence=primary,
                         building_limit=b_lim, contents_limit=c_lim, building_deductible=b_ded,
                         contents_deductible=c_ded, date_of_loss=loss, cause_code=cause,
                         flood_event=start.flood_event, water_depth_inches=depth,
                         building_damage=b_dmg, contents_damage=c_dmg)  # fmt: skip
    except ValidationError as exc:
        err = exc.errors()[0]
        st.error(f"Check the details: {err['loc'][0]}: {err['msg']}")
        st.stop()
    gaps = missing_fields(form)
    if gaps:
        st.error("Fill in: " + ", ".join(gaps) + ".")
        st.stop()
    claim = to_claim(form)
    out = flood_graph().invoke({"claim_id": claim.claim_id, "claim": claim.model_dump(mode="json")},
                               {"configurable": {"thread_id": claim.claim_id}})  # fmt: skip
    ss.f_result = {"out": out, "claim_id": claim.claim_id}
    st.rerun()

# ---------------------------------------------------------------- 3. decision
if ss.f_result:
    out = ss.f_result["out"]
    section("scale", "3. Decision")
    r = flood_summary(out)
    head, sub, tone = verdict(r)
    if r["status"] == "decided" and r["outcome"] == "deny":
        sub = "Nothing is payable under this policy for this loss."
    if r["status"] != "decided":
        sub = "A person decides this one: " + ", ".join(r["route_reasons"]).replace("_", " ") + "."
    mark = {"ok": success_tick(), "warn": review_alert(), "bad": deny_mark()}[tone]
    show(f'<div class="verdict tone-{tone}">{mark}<div><h2>{html.escape(head)}</h2>'
         f"<p>{html.escape(sub)}</p></div></div>")  # fmt: skip
    left, right = st.columns([3, 2], gap="large")
    with left:
        show(f"<b>{icon('signpost', 18)} How it was decided</b>")
        for s in flood_trace(out):
            show(f'<div class="step"><div class="dot dot-{s["status"]}">'
                 f"{icon(STEP_ICON[s['status']], 18, '#ffffff', 2.6)}</div><div>"
                 f'<b>{html.escape(s["title"])}</b><span>{html.escape(s["detail"])}</span>'
                 "</div></div>")  # fmt: skip
    with right:
        show(f"<b>{icon('file-text', 18)} Letter to the policyholder</b>")
        show(f'<div class="card">{html.escape(r["explanation"] or "-")}</div>')
        st.caption(f"Claim {ss.f_result['claim_id']} · decided by code · 0 LLM calls · $0.")
    with st.expander("Raw claim sent to the flood line"):
        st.json(out.get("claim") or {})
