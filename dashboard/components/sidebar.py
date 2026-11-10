"""Centralized sidebar brand identity, theme toggle, and shared filter components."""

from __future__ import annotations

from typing import Optional
import pandas as pd
import streamlit as st


def render_sidebar_header() -> None:
    """No-op: Application title and theme toggle are removed from the navigation menu."""
    pass


# Backwards compatibility alias
render_sidebar_brand = render_sidebar_header


# ---------------------------------------------------------------------------
# Shared filter helpers used by individual dashboard pages
# ---------------------------------------------------------------------------

def product_selector(
    demand_df: pd.DataFrame,
    label: str = "Select Product",
    key: str = "product_select",
    include_all: bool = False,
) -> Optional[int]:
    """Sidebar product selector returning product_id (int) or None if 'All'."""
    products = sorted(demand_df["product_id"].unique().tolist())
    options = (
        (["All Products"] + [f"Product {p}" for p in products])
        if include_all
        else [f"Product {p}" for p in products]
    )
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
