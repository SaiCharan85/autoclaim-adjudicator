"""One harness per Streamlit server, shared by every page (retriever and fraud models load once)."""

import streamlit as st
from dotenv import load_dotenv

from autoclaim.config import load_carrier_config
from autoclaim.lines.auto.build import AutoHarness, build
from autoclaim.llm.client import LLMClient
from autoclaim.paths import REPO_ROOT


@st.cache_resource
def harness() -> AutoHarness:
    load_dotenv(REPO_ROOT / ".env")  # API keys for live runs on the try-a-claim page
    return build()


@st.cache_resource
def llm_client() -> LLMClient:
    """A light client for reading a story into the form (no retriever or fraud models needed)."""
    load_dotenv(REPO_ROOT / ".env")
    return LLMClient.from_config(load_carrier_config().models)
