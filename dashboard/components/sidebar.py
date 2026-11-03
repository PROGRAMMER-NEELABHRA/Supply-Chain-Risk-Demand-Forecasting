"""Centralized sidebar brand identity, top-right theme toggle, and shared filter components."""

from typing import Optional
import pandas as pd
import streamlit as st
from dashboard.styles.theme import get_current_theme, get_theme_tokens


def render_sidebar_header() -> None:
    """Render the application branding in the sidebar without theme buttons."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                <span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--accent-blue);"></span>
                Supply Chain Platform
            </div>
            <div class="sidebar-brand-subtitle">
                Enterprise Demand &amp; Risk Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_theme_toggle() -> None:
    """Render a sleek, compact theme toggle pinned in the top right corner."""
    current_theme = get_current_theme()
    is_dark = current_theme == "dark"
    is_light = current_theme == "light"

    with st.container(key="top_right_theme"):
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            if st.button("🌙 Dark", key="toggle_dark_btn", type="primary" if is_dark else "secondary"):
                if current_theme != "dark":
                    st.session_state["theme"] = "dark"
                    st.rerun()
        with col2:
            if st.button("☀️ Light", key="toggle_light_btn", type="primary" if is_light else "secondary"):
                if current_theme != "light":
                    st.session_state["theme"] = "light"
                    st.rerun()


def product_selector(
    demand_df: pd.DataFrame,
    label: str = "Select Product",
    key: str = "product_select",
    include_all: bool = False,
) -> Optional[int]:
    """Sidebar product selector returning product_id (int) or None if 'All'."""
    products = sorted(demand_df["product_id"].unique().tolist())
    options = (["All Products"] + [f"Product {p}" for p in products]) if include_all else [f"Product {p}" for p in products]
    selected = st.sidebar.selectbox(label, options, key=key)
    if include_all and selected == "All Products":
        return None
    return int(selected.replace("Product ", ""))


def period_filter(label: str = "Evaluation Horizon", key: str = "period_filter") -> str:
    """Sidebar selector for model evaluation period."""
    return st.sidebar.radio(
        label,
        ["First period", "Later period", "Both"],
        key=key,
    )


def model_filter(
    label: str = "Model Selection",
    key: str = "model_filter",
    default: str = "Random Forest",
) -> str:
    """Sidebar selectbox for model choice."""
    models = ["Random Forest", "Moving Average", "Naive"]
    return st.sidebar.selectbox(
        label,
        models,
        index=models.index(default) if default in models else 0,
        key=key,
    )


def metric_selector(key: str = "metric_select") -> str:
    """Sidebar selectbox for error metric choice."""
    mapping = {"MAE (Units)": "mae", "RMSE (Units)": "rmse", "WAPE (%)": "wape_pct"}
    chosen = st.sidebar.selectbox("Evaluation Metric", list(mapping.keys()), key=key)
    return mapping[chosen]


# Backwards compatibility
def render_sidebar_brand() -> None:
    render_sidebar_header()
