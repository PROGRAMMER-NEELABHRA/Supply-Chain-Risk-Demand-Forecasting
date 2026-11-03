"""Inventory Management — Policy modeling and replenishment planning (Under Development)."""

import pandas as pd
import streamlit as st

from dashboard.components.kpi_cards import page_header, under_development_banner
from dashboard.data.loader import load_future_forecasts, load_monthly_demand

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    title="Inventory Management",
    subtitle="Replenishment policy modeling, dynamic safety stock, and service level targets",
    category="OPERATIONS",
    badge_text="In Development",
    badge_variant="orange",
)

# ---------------------------------------------------------------------------
# Development Governance Notice
# ---------------------------------------------------------------------------

under_development_banner(
    module_title="Inventory Optimisation Engine",
    target_module="src/inventory/inventory_analysis.py",
    description=(
        "Once fully implemented, this module will compute dynamic safety stock requirements, "
        "Economic Order Quantity (EOQ), lead-time variability buffers, and stockout risk alerts "
        "driven by empirical forecast error variances."
    ),
    disclaimer=(
        "The exploratory table below provides raw demand forecast projections to assist preliminary "
        "planners. It does NOT represent algorithmic inventory optimization, automated PO triggers, or "
        "validated stocking advice."
    ),
)

st.divider()

# ---------------------------------------------------------------------------
# Planning Input Preview (Forecast Projections)
# ---------------------------------------------------------------------------

st.markdown("### Forecast Demand Planning Inputs (Exploratory Preview)")
st.markdown(
    "<p style='color:var(--text-muted);font-size:0.85rem;margin-top:-0.3rem;margin-bottom:1rem;'>"
    "Aggregated 3-month forward demand predictions compared against last observed order volume."
    "</p>",
    unsafe_allow_html=True,
)

forecasts = load_future_forecasts()
demand = load_monthly_demand()

# Summarise per product
product_summary = forecasts.groupby("product_id").agg(
    total_forecast_demand=("forecast_demand", "sum"),
    avg_monthly_forecast=("forecast_demand", "mean"),
    confidence=("forecast_confidence", "first"),
    model=("model_name", "first"),
).reset_index()

# Attach last observed demand for historical context
last_obs = (
    demand.sort_values("month")
    .groupby("product_id")
    .last()["demand_units"]
    .reset_index()
    .rename(columns={"demand_units": "last_observed_demand"})
)
product_summary = product_summary.merge(last_obs, on="product_id", how="left")

# Simple directional trend indicator
product_summary["trend_signal"] = (
    product_summary["avg_monthly_forecast"] > product_summary["last_observed_demand"]
).map({True: "Growth Projected", False: "Stable / Decline"})

# Sidebar filter
st.sidebar.markdown("### Inventory Scope")
conf_options = st.sidebar.multiselect(
    "Confidence Tier",
    ["HIGH", "MEDIUM", "LOW"],
    default=["HIGH", "MEDIUM", "LOW"],
    key="inv_conf",
)

filtered = product_summary[product_summary["confidence"].isin(conf_options)]

display = filtered.rename(columns={
    "product_id": "Product ID",
    "total_forecast_demand": "3M Forward Demand",
    "avg_monthly_forecast": "Mean Monthly Forecast",
    "confidence": "Forecast Confidence",
    "model": "Assigned Model",
    "last_observed_demand": "Last Observed Demand",
    "trend_signal": "Directional Signal",
})

st.dataframe(
    display.style.format({
        "3M Forward Demand": "{:,.0f}",
        "Mean Monthly Forecast": "{:,.1f}",
        "Last Observed Demand": "{:,.0f}",
    }),
    width="stretch",
    height=420,
    hide_index=True,
)

st.caption(
    "Governance Disclaimer: Directional signals reflect raw arithmetic deltas between forecast means "
    "and trailing records. They are exploratory indicators and do not replace validated inventory optimization."
)
