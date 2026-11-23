"""Demand Analytics — In-depth historical demand patterns, seasonality, and concentration."""

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    demand_heatmap,
    demand_time_series,
    seasonality_line,
    top_products_bar,
)
from dashboard.components.kpi_cards import kpi_row, page_header
from dashboard.components.sidebar import product_selector
from dashboard.data.loader import load_demand_shift_analysis, load_monthly_demand

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

demand = load_monthly_demand()
shift_df = load_demand_shift_analysis()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Analytics Scope")

selected_product = product_selector(
    demand, label="Filter by SKU", key="da_product", include_all=True
)

top_n = st.sidebar.slider("Top SKU Depth", min_value=5, max_value=25, value=10, key="da_topn")

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    title="Demand Analytics",
    subtitle="Historical ordering dynamics, seasonal profiles, and product volume concentration",
    category="DEMAND INTELLIGENCE",
    badge_text="Observed History",
    badge_variant="blue",
)

# ---------------------------------------------------------------------------
# Key Metrics
# ---------------------------------------------------------------------------

if selected_product is not None:
    sub = demand[demand["product_id"] == selected_product]
    label_suffix = f"Product {selected_product}"
else:
    sub = demand
    label_suffix = "Full Portfolio"

if sub.empty:
    st.info(
        f"No demand records were found for Product {selected_product}. "
        "Select another SKU or choose All to view portfolio-wide data."
        if selected_product is not None
        else "No demand records are available for the portfolio."
    )
else:
    total_units = float(sub["demand_units"].sum())
    monthly_agg = sub.groupby("month")["demand_units"].sum()
    avg_monthly = float(monthly_agg.mean())
    peak_month = monthly_agg.idxmax()
    peak_val = float(monthly_agg.max())

    kpi_row([
        {
            "label": f"Historical Volume ({label_suffix})",
            "value": f"{total_units:,.0f} units",
            "help": "Sum of all recorded demand units for the selected scope.",
            "provenance": "monthly_demand_observed.csv",
        },
        {
            "label": "Mean Monthly Demand",
            "value": f"{avg_monthly:,.0f} units",
            "help": "Arithmetic mean of monthly totals.",
            "provenance": "Aggregated by month",
        },
        {
            "label": "Peak Volume Month",
            "value": peak_month.strftime("%b %Y"),
            "help": "Calendar month with the highest aggregate observed demand.",
            "provenance": "Max monthly volume",
        },
        {
            "label": "Peak Month Volume",
            "value": f"{peak_val:,.0f} units",
            "help": "Highest volume recorded in a single month for this scope.",
            "provenance": "Peak monthly demand",
        },
    ])

st.divider()

# ---------------------------------------------------------------------------
# Demand Trajectory Chart
# ---------------------------------------------------------------------------

if not sub.empty:
    chart_title = (
        f"Monthly Historical Demand — Product {selected_product}"
        if selected_product is not None
        else "Portfolio Aggregate Monthly Demand"
    )
    fig_ts = demand_time_series(demand, product_id=selected_product, title=chart_title)
    st.plotly_chart(
        fig_ts,
        width="stretch",
        alt=(
            f"Monthly observed demand from {sub['month'].min():%b %Y} through "
            f"{sub['month'].max():%b %Y} for {label_suffix}. "
            "The horizontal axis is month and the vertical axis is demand units; "
            "the line shows the demand trend."
        ),
    )

st.divider()

# ---------------------------------------------------------------------------
# Volume Ranking & Seasonality Profiles
# ---------------------------------------------------------------------------

if demand.empty:
    st.info(
        "Portfolio charts are unavailable because no demand data is present.",
        icon="ℹ️",
    )
else:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        fig_top = top_products_bar(
            demand, n=top_n, title=f"Top {top_n} SKUs by Cumulative Volume"
        )
        st.plotly_chart(
            fig_top,
            width="stretch",
            alt=(
                f"Top {top_n} products by cumulative historical demand across the "
                "portfolio. The vertical axis lists product IDs and the horizontal "
                "axis shows total units ordered."
            ),
        )

    with col2:
        fig_seasonal = seasonality_line(demand)
        st.plotly_chart(
            fig_seasonal,
            width="stretch",
            alt=(
                "Average portfolio demand by calendar month, calculated across "
                "observed years. The horizontal axis is calendar month and the "
                "vertical axis is mean monthly demand units; the line highlights "
                "seasonal variation."
            ),
        )

st.divider()

# ---------------------------------------------------------------------------
# Product-Month Demand Matrix (Heatmap)
# ---------------------------------------------------------------------------

if not demand.empty:
    with st.expander("Product × Month Demand Matrix (Heatmap)", expanded=False):
        st.markdown(
            "<p style='color:var(--text-muted);font-size:0.85rem;margin-bottom:0.75rem;'>"
            "Interactive matrix showing monthly order intensity across all product series. "
            "Darker cells represent lower or zero demand; vibrant cells highlight high order volumes."
            "</p>",
            unsafe_allow_html=True,
        )
        fig_heat = demand_heatmap(demand)
        st.plotly_chart(
            fig_heat,
            width="stretch",
            alt=(
                "Demand matrix for observed history by product and month. Rows are "
                "product IDs, columns are months, and cell color represents summed "
                "demand units; higher-intensity cells indicate greater demand."
            ),
        )

st.divider()

# ---------------------------------------------------------------------------
# Demand Shift Analysis (Train vs. Test)
# ---------------------------------------------------------------------------

st.markdown("### Demand Regime Transition (Train vs. Test)")
validation_end = (
    demand["month"].max().strftime("%b %Y")
    if not demand.empty
    else "Unavailable"
)
st.markdown(
    "<p style='color:var(--text-muted);font-size:0.85rem;margin-top:-0.3rem;margin-bottom:1rem;'>"
    "Audit of baseline demand shifts between the model training window "
    f"(Jan 2015–Apr 2017) and validation period "
    f"(May 2017–{validation_end})."
    "</p>",
    unsafe_allow_html=True,
)

shift_display = shift_df[[
    "product_id",
    "train_average_monthly_demand",
    "test_average_monthly_demand",
    "average_demand_change_pct",
    "demand_shift",
]].copy()

shift_display.columns = [
    "Product ID",
    "Train Avg Demand",
    "Test Avg Demand",
    "Change (%)",
    "Shift Category",
]
shift_display = shift_display.sort_values("Change (%)", ascending=False)

if selected_product is not None:
    shift_display = shift_display[shift_display["Product ID"] == selected_product]

if shift_display.empty:
    st.info(
        f"No Demand Regime Transition records match Product {selected_product}. "
        "Select another SKU or choose All to view available portfolio records."
        if selected_product is not None
        else "No Demand Regime Transition records are available for the portfolio."
    )
else:
    st.dataframe(
        shift_display.style.format({
            "Train Avg Demand": "{:.1f}",
            "Test Avg Demand": "{:.1f}",
            "Change (%)": "{:+.1f}%",
        }),
        width="stretch",
        height=280,
        hide_index=True,
    )
