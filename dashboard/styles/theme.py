"""Public theme API. Keeps existing imports working while implementation is split."""

import streamlit as st

from .palette import (
    THEMES, NAVY_BG, NAVY_SIDEBAR, CARD_BG, CARD_BORDER,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_ORANGE,
    DANGER_RED, TEXT_PRIMARY, TEXT_MUTED, TEXT_DIM,
    COLOUR_SEQUENCE, MODEL_COLOURS, CONFIDENCE_COLOURS,
)
from .plotly import get_plotly_layout, PLOTLY_LAYOUT, PLOTLY_TEMPLATE


def get_current_theme() -> str:
    """Return Streamlit's active native appearance setting."""
    return st.context.theme.type


def get_theme_tokens(theme: str | None = None) -> dict[str, str]:
    """Return palette tokens for the active native Streamlit theme."""
    selected = theme or get_current_theme() or "light"
    return THEMES.get(selected, THEMES["light"])


def inject_css() -> None:
    """Load shared CSS that follows Streamlit's native appearance setting."""
    from pathlib import Path

    css_path = Path(__file__).with_name("base.css")
    base_css = css_path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>\n{base_css}\n</style>",
        unsafe_allow_html=True,
    )
