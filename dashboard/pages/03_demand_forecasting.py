"""Demand Forecasting — Forward 3-month predictions and confidence distributions."""

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    confidence_donut,
    demand_forecast_chart,
    method_donut,
)
from dashboard.components.kpi_cards import kpi_row, page_header
from dashboard.data.loader import (
    load_forecast_data_warnings,
    load_forecast_method_summary,
    load_future_forecasts,
    load_monthly_demand,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

demand = load_monthly_demand()
forecasts = load_future_forecasts()
method_summary = load_forecast_method_summary()
warnings_df = load_forecast_data_warnings()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------

demand_available = not demand.empty
forecast_available = not forecasts.empty
selected_product = None

if forecast_available:
    st.sidebar.markdown("### SKU Inspection")

    all_products = sorted(
        forecasts["product_id"].dropna().unique().tolist()
    )
    if all_products:
        selected_product = st.sidebar.selectbox(
            "Target Product SKU",
            all_products,
            format_func=lambda pid: f"Product {pid}",
            key="fc_product",
        )
    else:
        st.sidebar.info("No product forecast rows are available.")

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    title="Demand Forecasting",
    subtitle=(
        (
            f"Forecast origin: {demand['month'].max():%b %Y}. "
            if demand_available
            else "Forecast origin: Unavailable. "
        )
        + (
            f"Forecast horizon: {forecasts['forecast_month'].min():%b %Y} – "
            f"{forecasts['forecast_month'].max():%b %Y}. "
            if forecast_available
            else "Forecast horizon: Unavailable."
        )
    ),
    category="DEMAND INTELLIGENCE",
    badge_text="Operational Forecast",
    badge_variant="green",
)

st.info(
    "Confidence is a data-quality heuristic, not a guarantee of forecast "
    "accuracy. The current rating is based on observed-history length, "
    "freshness, and gaps; it is not calibrated to historical backtest "
    "performance."
)

if not forecast_available:
    if demand_available:
        st.info(
            "No forecast data is currently available. "
            f"Historical demand is available from "
            f"{demand['month'].min():%b %Y} through "
            f"{demand['month'].max():%b %Y}. "
            "Forecast-dependent KPIs, charts, and the forecast register "
            "are unavailable until forecast rows are generated."
        )
    else:
        st.info(
            "No forecast data or historical demand data is currently available. "
            "Generate forecast results and load historical demand records "
            "before reviewing this page."
        )
    st.stop()

if not demand_available:
    st.info(
        "No historical demand data is available. Forecast-dependent model "
        "results remain available, but observed-history depth and forecast "
        "history overlays are unavailable."
    )

# ---------------------------------------------------------------------------
# Product-Specific KPIs
# ---------------------------------------------------------------------------

prod_fc = forecasts[forecasts["product_id"] == selected_product].copy()
prod_hist = demand[demand["product_id"] == selected_product].copy()

if not prod_fc.empty:
    model_used = prod_fc["model_name"].iloc[0]
    confidence = prod_fc["forecast_confidence"].iloc[0]
    method = prod_fc["forecast_method"].iloc[0]
    total_fc_demand = float(prod_fc["forecast_demand"].sum())

    kpi_metrics = [
        {
            "label": f"Assigned Model (SKU {selected_product})",
            "value": model_used,
            "help": "Modeling technique selected based on historical eligibility criteria.",
            "provenance": "future_demand_forecasts.csv",
        },
        {
            "label": "Confidence Rating",
            "value": confidence,
            "help": (
                "Heuristic based on data freshness and history completeness; "
                "it does not guarantee future forecast accuracy."
            ),
            "provenance": "data_quality_status",
        },
    ]
    if demand_available:
        obs_months = int(prod_fc["observed_history_months"].iloc[0])
        kpi_metrics.append({
            "label": "Observed History Depth",
            "value": f"{obs_months} Months",
            "help": "Total months of actual transaction records available for this SKU.",
            "provenance": "monthly_demand_observed.csv",
        })
    kpi_metrics.append({
        "label": "Cumulative 3M Demand",
        "value": f"{total_fc_demand:,.1f} units",
        "help": "Sum of predicted monthly units across the 3-month forward horizon.",
        "provenance": "Forecast projection sum",
    })
    kpi_row(kpi_metrics)

    st.divider()

    # ---------------------------------------------------------------------------
    # Forecast Trajectory Chart
    # ---------------------------------------------------------------------------

    if demand_available:
        fig = demand_forecast_chart(demand, forecasts, selected_product)
        st.plotly_chart(
            fig,
            width="stretch",
            alt=(
                f"Observed demand and the three-month forecast for Product "
                f"{selected_product}, with forecasts from "
                f"{prod_fc['forecast_month'].min():%b %Y} through "
                f"{prod_fc['forecast_month'].max():%b %Y}. The horizontal axis is "
                "month and the vertical axis is demand units; solid markers show "
                "actual history and the dashed line shows the forward forecast."
            ),
        )

    # ---------------------------------------------------------------------------
    # Forecast Schedule Table & Quality Notices
    # ---------------------------------------------------------------------------

    col_sched, col_notices = st.columns([1.5, 1], gap="medium")

    with col_sched:
        st.markdown("### Forward Schedule Breakdown")
        table_cols = [
            "forecast_month",
            "forecast_demand",
            "model_name",
            "forecast_confidence",
            "forecast_method",
            "data_quality_status",
            "months_since_last_record",
            "internal_missing_months",
        ]
        display = prod_fc[table_cols].copy()
        display["forecast_month"] = display["forecast_month"].dt.strftime("%b %Y")
        display.columns = [
            "Month",
            "Forecast Demand",
            "Model",
            "Confidence",
            "Methodology",
            "Data Quality Status",
            "Months Since Last",
            "Gaps",
        ]

        def _fmt_confidence(val: str) -> str:
            mapping = {
                "HIGH": (
                    "color: var(--text-primary); font-weight: 600;"
                    "border-left: 3px solid var(--accent-green);"
                ),
                "MEDIUM": (
                    "color: var(--text-primary); font-weight: 600;"
                    "border-left: 3px solid var(--accent-orange);"
                ),
                "LOW": (
                    "color: var(--text-primary); font-weight: 600;"
                    "border-left: 3px solid var(--danger-red);"
                ),
            }
            return mapping.get(val, "")

        st.dataframe(
            display.style.format({"Forecast Demand": "{:.1f}"}).map(
                _fmt_confidence, subset=["Confidence"]
            ),
            width="stretch",
            hide_index=True,
        )

    with col_notices:
        st.markdown("### Data Quality Flags")
        prod_warnings = warnings_df[warnings_df["product_id"] == selected_product]
        if not prod_warnings.empty:
            for _, row in prod_warnings.iterrows():
                st.warning(
                    f"**{row['warning_category']}** — {row['warning_message']}",
                    icon="⚠️",
                )
        else:
            st.success(
                "No data-quality warnings detected for this SKU. History extends continuously to the forecast cutoff.",
                icon="✅",
            )

else:
    st.error(f"No forecast output records available for Product {selected_product}.")

st.divider()

# ---------------------------------------------------------------------------
# Fleet-Level Distribution Summary
# ---------------------------------------------------------------------------

st.markdown("### Portfolio Forecast Distribution")

if method_summary.empty:
    st.info(
        "No portfolio forecast distribution results are available.",
        icon="ℹ️",
    )
else:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.plotly_chart(
            method_donut(method_summary),
            width="stretch",
            alt=(
                "Portfolio share of products assigned to Random Forest, Moving "
                "Average, or Naive forecasting methods. There are no axes; each "
                "segment represents a method and its share of products."
            ),
        )
    with col2:
        st.plotly_chart(
            confidence_donut(method_summary),
            width="stretch",
            alt=(
                "Portfolio share of products in the HIGH, MEDIUM, and LOW forecast "
                "confidence categories. There are no axes; each segment represents "
                "a confidence category and its share of products."
            ),
        )

st.divider()

# ---------------------------------------------------------------------------
# Full Portfolio Forecast Register (Filterable)
# ---------------------------------------------------------------------------

with st.expander("Complete Portfolio Forecast Register", expanded=False):
    conf_filter = st.multiselect(
        "Filter by Confidence Rating",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
        key="fc_conf_filter",
    )
    all_fc = forecasts[forecasts["forecast_confidence"].isin(conf_filter)].copy()
    if all_fc.empty:
        st.info(
            "No forecast register rows match the selected confidence ratings.",
            icon="ℹ️",
        )
    else:
        all_fc["forecast_month"] = all_fc["forecast_month"].dt.strftime("%b %Y")

        st.dataframe(
            all_fc[[
                "product_id",
                "forecast_month",
                "forecast_demand",
                "model_name",
                "forecast_confidence",
                "data_quality_status",
            ]].rename(columns={
                "product_id": "Product ID",
                "forecast_month": "Month",
                "forecast_demand": "Forecast Demand",
                "model_name": "Model",
                "forecast_confidence": "Confidence",
                "data_quality_status": "Data Quality Status",
            }),
            width="stretch",
            height=380,
            hide_index=True,
        )
