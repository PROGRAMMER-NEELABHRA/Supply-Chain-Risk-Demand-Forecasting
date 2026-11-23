"""Shared application branding using Streamlit's native logo slot.

The native st.logo API places the wordmark in Streamlit's toolbar area and
handles sidebar open/closed positioning without a fixed-position CSS overlay.
"""

from pathlib import Path

import streamlit as st


_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def render_app_header() -> None:
    """Render the theme-matched wordmark in Streamlit's native app header."""
    theme_name = getattr(st.context.theme, "type", None) or "light"
    if theme_name not in {"light", "dark"}:
        theme_name = "light"

    wordmark = _ASSETS_DIR / f"supply_chain_wordmark_{theme_name}.svg"
    st.logo(str(wordmark), size="large")
