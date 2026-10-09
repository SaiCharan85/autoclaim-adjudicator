"""Adjuster console: review claims the harness escalated and resume them with a decision.

Run:  uv run streamlit run src/autoclaim/ui/console.py

Submitting a decision resumes the paused claim (finalized exactly once, even if submitted twice)
and stores the episode in feedback memory. No LLM call is made by anything on this page.
"""

import streamlit as st

from autoclaim.core.review import pending, resume
from autoclaim.paths import data_dir
from autoclaim.ui.assets import ANIMATION_CSS, BRAND, illustration
from autoclaim.ui.review_view import (
    OUTCOMES,
    FormError,
    audit_rows,
    case_markdown,
    decision_from_form,
    queue_rows,
)
from autoclaim.ui.shared import harness


def main() -> None:
    st.set_page_config(page_title="Adjuster console · autoclaim",
                       page_icon=str(BRAND / "favicon.png"), layout="wide")  # fmt: skip
    st.logo(str(BRAND / "logo.svg"), icon_image=str(BRAND / "favicon.svg"))
    st.markdown(ANIMATION_CSS, unsafe_allow_html=True)
    st.title("Adjuster console")
    st.caption("Claims the harness could not decide alone. Your decision is final and is "
               "remembered as feedback for similar future claims.")  # fmt: skip
    app = harness()
    queue = pending(app.graph)
    st.sidebar.metric("Waiting for review", len(queue))
    if st.sidebar.button("Refresh"):
        st.rerun()
    if not queue:
        st.markdown(f'<div style="text-align:center;padding:24px">'
                    f'{illustration("empty_queue", 300, "No claims waiting")}'
                    "<h4>All caught up</h4><p>No claims are waiting for a person. Try one on the "
                    "<b>Try a claim</b> page.</p></div>", unsafe_allow_html=True)  # fmt: skip
        return

    st.dataframe(queue_rows(queue), hide_index=True, use_container_width=True)
    by_id = dict(queue)
    claim_id = st.selectbox("Claim", list(by_id))
    req = by_id[claim_id]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(case_markdown(req))
        with st.expander("Full case summary (JSON)"):
            st.json(req.get("case_summary") or {})
    with right:
        prop = req.get("proposed_decision") or {}
        default = OUTCOMES.index(prop["outcome"]) if prop.get("outcome") in OUTCOMES else 0
        with st.form(f"decide-{claim_id}"):
            outcome = st.radio("Decision", OUTCOMES, index=default, horizontal=True)
            payout = st.number_input("Payout (approvals only)", min_value=0.0,
                                     value=float(prop.get("payout") or 0.0), step=50.0)  # fmt: skip
            reason = st.text_area("Reason (remembered as feedback)")
            adjuster = st.text_input("Your name")
            submitted = st.form_submit_button("Submit decision")
        if submitted:
            try:
                decision = decision_from_form(outcome, payout, reason, adjuster)
            except FormError as exc:
                st.error(str(exc))
            else:
                out = resume(app.graph, claim_id, decision)
                final = out.get("final", {})
                st.success(f"{claim_id}: {final.get('outcome')} "
                           f"(decided by {final.get('decided_by')})")  # fmt: skip
                st.rerun()
    with st.expander("Audit trail"):
        st.dataframe(audit_rows(data_dir() / "audit", claim_id), hide_index=True,
                     use_container_width=True)  # fmt: skip


main()
