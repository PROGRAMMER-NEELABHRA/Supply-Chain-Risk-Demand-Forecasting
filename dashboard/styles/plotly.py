"""Plotly layout configuration that follows the active dashboard theme."""

import streamlit as st

from .palette import THEMES


def get_plotly_layout(theme: str | None = None) -> dict:
    """Return a Plotly layout dictionary for the active or requested theme."""
    theme = theme or st.context.theme.type
    colors = THEMES.get(theme, THEMES["dark"])
    axis = {
        "gridcolor": colors["grid_color"],
        "linecolor": colors["border_color"],
        "tickfont": {"color": colors["text_muted"], "size": 11},
        "title": {"font": {"color": colors["text_muted"], "size": 12}},
        "zerolinecolor": colors["grid_color"],
    }
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": colors["plot_bg"],
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "color": colors["text_primary"],
            "size": 12,
        },
        "xaxis": dict(axis),
        "yaxis": dict(axis),
        "margin": {"l": 16, "r": 16, "t": 44, "b": 16},
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": colors["text_primary"], "size": 11},
        },
    }


PLOTLY_LAYOUT = get_plotly_layout("dark")
PLOTLY_TEMPLATE = "plotly_dark"
