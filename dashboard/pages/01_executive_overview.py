"""Executive Overview — Strategic enterprise summary of demand and forecast health."""

import streamlit as st

from dashboard.components.charts import (
    demand_time_series,
    method_donut,
    top_products_bar,
    warning_bar,
)
from dashboard.components.kpi_cards import kpi_row, page_header, provenance_expander
from dashboard.data.loader import (
    load_data_quality_summary,
    load_forecast_method_summary,
    load_forecast_warning_summary,
    load_future_forecasts,
    load_monthly_demand,
)

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

demand = load_monthly_demand()
forecasts = load_future_forecasts()
method_summary = load_forecast_method_summary()
warning_summary = load_forecast_warning_summary()
quality_summary = load_data_quality_summary()

# ---------------------------------------------------------------------------
# Derive Traceable Metrics
# ---------------------------------------------------------------------------

total_products = int(demand["product_id"].nunique())
total_demand_units = float(demand["demand_units"].sum())
if demand.empty:
    first_month = "Unavailable"
    last_month = "Unavailable"
    total_months = 0
    coverage_span = "Unavailable"
    forecast_origin = "Unavailable"
else:
    first_month = demand["month"].min().strftime("%b %Y")
    last_month = demand["month"].max().strftime("%b %Y")
    total_months = int(demand["month"].nunique())
    coverage_span = f"{first_month} – {last_month} ({total_months}m)"
    forecast_origin = demand["month"].max().strftime("%b %Y")

total_forecast_demand = float(forecasts["forecast_demand"].sum())
if forecasts.empty:
    forecast_span = "Unavailable"
else:
    forecast_span = (
        f"{forecasts['forecast_month'].min().strftime('%b %Y')} – "
        f"{forecasts['forecast_month'].max().strftime('%b %Y')}"
    )

rf_count = int(
    method_summary.loc[
        method_summary["category"] == "random_forest_recursive", "product_count"
    ].sum()
)
rf_coverage_pct = (
    round((rf_count / total_products) * 100, 1)
    if total_products
    else None
)

stale_count = int(
    warning_summary.loc[
        warning_summary["warning_category"] == "STALE_HISTORY", "product_count"
    ].sum()
)
quality_score = (
    round((1 - stale_count / total_products) * 100, 1)
    if total_products
    else None
)

qs_val = quality_summary.set_index("metric")["value"].to_dict()
raw_tx = int(float(qs_val.get("raw_transaction_records", 0)))
observed_pm = int(float(qs_val.get("observed_product_months", 0)))

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    title="Executive Overview",
    subtitle="Enterprise demand trajectory, portfolio forward commitments, and pipeline data governance",
    category="OVERVIEW",
    badge_text="Audited Data",
    badge_variant="green",
)

# ---------------------------------------------------------------------------
# Executive KPI Deck
# ---------------------------------------------------------------------------

kpi_row([
    {
        "label": "Monitored Products",
        "value": f"{total_products:,}",
        "help": "Distinct active product SKUs recorded in observed demand logs.",
        "provenance": "monthly_demand_observed.csv",
    },
    {
        "label": "Total Observed Demand",
        "value": f"{total_demand_units:,.0f}",
        "help": "Cumulative aggregate units ordered across all calendar months and products.",
        "provenance": "Sum of demand_units",
    },
    {
        "label": "Portfolio Historical Span",
        "value": coverage_span,
        "help": (
            f"The {total_months}-month value is the portfolio calendar span, "
            "not complete history for every SKU."
        ),
        "provenance": "data_quality_summary.csv",
    },
])

st.markdown("<div style='margin-bottom:0.75rem;'></div>", unsafe_allow_html=True)

kpi_row([
    {
        "label": f"Forward 3M Forecast ({forecast_span})",
        "value": f"{total_forecast_demand:,.0f}",
        "caption": (
            f"Forecast origin: {forecast_origin}. "
            f"Forecast horizon: {forecast_span}."
        ),
        "help": (
            f"Portfolio forecast generated from observed data through "
            f"{forecast_origin}."
        ),
        "provenance": "future_demand_forecasts.csv",
    },
    {
        "label": "Machine Learning Coverage",
        "value": "Unavailable" if rf_coverage_pct is None else f"{rf_coverage_pct}%",
        "caption": (
            "No product demand data is available."
            if total_products == 0
            else (
                f"{rf_count:,} of {total_products:,} SKUs "
                f"({rf_coverage_pct}%) use Random Forest."
            )
        ),
        "help": "Products satisfying minimum historical threshold to qualify for recursive Random Forest model.",
        "provenance": "forecast_method_summary.csv",
    },
    {
        "label": "Not Flagged Stale",
        "value": "Unavailable" if quality_score is None else f"{quality_score}%",
        "caption": (
            "No product demand data is available."
            if total_products == 0
            else f"{stale_count:,} of {total_products:,} SKUs are flagged stale."
        ),
        "help": (
            "Share of SKUs not flagged stale at the forecast cutoff; "
            "this is not a comprehensive data-quality score."
        ),
        "provenance": "forecast_warning_summary.csv",
    },
])

if total_products == 0:
    st.divider()
    st.info(
        "No product demand data is available. The overview metrics and "
        "forecast visuals are unavailable until demand records are loaded.",
        icon="ℹ️",
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Visual Analytics Row 1: Demand Trend & Top Volume SKUs
# ---------------------------------------------------------------------------

coverage_limit = qs_val.get("source_coverage_limit", "")
if coverage_limit:
    st.info(
        f"**Dataset Coverage Limit:** {coverage_limit}",
        icon="ℹ️",
    )

col_trend, col_top = st.columns([1.6, 1], gap="medium")

with col_trend:
    fig_demand = demand_time_series(
        demand, title="Portfolio Aggregate Monthly Demand (Units)"
    )
    fig_demand.update_layout(height=360)
    st.plotly_chart(
        fig_demand,
        width="stretch",
        alt=(
            f"Monthly total observed demand for all products from {first_month} "
            f"through {last_month}. The horizontal axis is month and the vertical "
            "axis is demand units; the line shows the portfolio trend over time."
        ),
    )

with col_top:
    fig_top = top_products_bar(
        demand, n=8, title="Top 8 SKUs by Historical Volume"
    )
    fig_top.update_layout(height=360)
    st.plotly_chart(
        fig_top,
        width="stretch",
        alt=(
            "Top 8 products ranked by cumulative observed demand. The vertical "
            "axis lists product IDs and the horizontal axis shows total units "
            "ordered; longer bars indicate higher historical volume."
        ),
    )

st.divider()

# ---------------------------------------------------------------------------
# Visual Analytics Row 2: Method Distribution & Data Quality Warnings
# ---------------------------------------------------------------------------

col_method, col_warn = st.columns([1, 1], gap="medium")

with col_method:
    fig_method = method_donut(method_summary)
    fig_method.update_layout(height=360)
    st.plotly_chart(
        fig_method,
        width="stretch",
        alt=(
            "Share of products assigned to Random Forest, Moving Average, or "
            "Naive forecasting methods. There are no axes; each segment represents "
            "a method and its size shows its share of products."
        ),
    )

with col_warn:
    fig_warn = warning_bar(
        warning_summary,
        title="Primary Forecast Warning Categories",
    )
    fig_warn.update_layout(height=360)
    st.plotly_chart(
        fig_warn,
        width="stretch",
        alt=(
            "Products by primary data-quality warning category. The vertical axis "
            "lists warning categories and the horizontal axis shows affected "
            "product counts; longer bars indicate more affected products."
        ),
    )
    st.caption(
        "Each product receives one primary warning category; "
        "additional issues may appear in the data-quality summary."
    )

st.divider()

# ---------------------------------------------------------------------------
# Data Governance & Provenance Section
# ---------------------------------------------------------------------------

provenance_expander([
    {
        "metric": "Monitored Products & Historical Volume",
        "source": "data/processed/monthly_demand_observed.csv",
        "description": f"Contains {raw_tx:,} raw transactions aggregated into {observed_pm:,} monthly product observations.",
    },
    {
        "metric": "3-Month Demand Forecasts",
        "source": "outputs/forecasts/future_demand_forecasts.csv",
        "description": f"Multi-step forward forecast covering horizon {forecast_span}.",
    },
    {
        "metric": "Model Assignment Breakdown",
        "source": "outputs/forecasts/forecast_method_summary.csv",
        "description": "Classification into Random Forest Recursive vs. Moving Average / Naive baselines based on history eligibility.",
    },
    {
        "metric": "Data Quality Warnings",
        "source": "outputs/forecasts/forecast_warning_summary.csv & forecast_data_warnings.csv",
        "description": "Structured flags denoting stale series, baseline fallback, or internal lifecycle gaps.",
    },
])
