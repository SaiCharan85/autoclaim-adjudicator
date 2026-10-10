"""The claims app: an Auto section and a Flood section in the sidebar.

Run:  uv run streamlit run src/autoclaim/ui/app.py   (opens http://127.0.0.1:8501)
"""

import streamlit as st

from autoclaim.ui.assets import BRAND

st.set_page_config(page_title="autoclaim", page_icon=str(BRAND / "favicon.png"), layout="wide")
st.logo(str(BRAND / "logo.svg"), icon_image=str(BRAND / "favicon.svg"))
nav = st.navigation({
    "Auto": [st.Page("views/try_a_claim.py", title="Try a claim", url_path="try_a_claim",
                     default=True),
             st.Page("console.py", title="Adjuster console", url_path="console")],
    "Flood": [st.Page("views/flood_claim.py", title="Flood claim", url_path="flood_claim")],
})  # fmt: skip
nav.run()
