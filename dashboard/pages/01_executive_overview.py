"""Executive Overview — Strategic enterprise summary of demand and forecast health."""

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    confidence_donut,
    demand_time_series,
    method_donut,
    top_products_bar,
    warning_bar,
)
from dashboard.components.kpi_cards import kpi_row, page_header, provenance_expander
from dashboard.data.loader import (
    load_data_quality_summary,
    load_demand_shift_analysis,
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
shift_analysis = load_demand_shift_analysis()
quality_summary = load_data_quality_summary()

# ---------------------------------------------------------------------------
# Derive Traceable Metrics
# ---------------------------------------------------------------------------

total_products = int(demand["product_id"].nunique())
total_demand_units = float(demand["demand_units"].sum())
first_month = demand["month"].min().strftime("%b %Y")
last_month = demand["month"].max().strftime("%b %Y")
total_months = int(demand["month"].nunique())
coverage_span = f"{first_month} – {last_month} ({total_months}m)"

total_forecast_demand = float(forecasts["forecast_demand"].sum())
forecast_span = (
    f"{forecasts['forecast_month'].min().strftime('%b %Y')} – "
    f"{forecasts['forecast_month'].max().strftime('%b %Y')}"
)

rf_count = int(
    method_summary.loc[
        method_summary["category"] == "random_forest_recursive", "product_count"
    ].sum()
)
rf_coverage_pct = round((rf_count / total_products) * 100, 1)

stale_count = int(
    warning_summary.loc[
        warning_summary["warning_category"] == "STALE_HISTORY", "product_count"
    ].sum()
)
quality_score = round((1 - stale_count / total_products) * 100, 1)

qs_val = quality_summary.set_index("metric")["value"].to_dict()
raw_tx = int(float(qs_val.get("raw_transaction_records", 0)))
observed_pm = int(float(qs_val.get("observed_product_months", 0)))
expected_pm = int(float(qs_val.get("expected_full_grid_product_months", 0)))

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
        "label": "Historical Coverage",
        "value": coverage_span,
        "help": "Start to cutoff of continuous historical observation.",
        "provenance": "data_quality_summary.csv",
    },
])

st.markdown("<div style='margin-bottom:0.75rem;'></div>", unsafe_allow_html=True)

kpi_row([
    {
        "label": f"Forward 3M Forecast ({forecast_span})",
        "value": f"{total_forecast_demand:,.0f}",
        "help": "Portfolio projected demand over next 3 calendar months.",
        "provenance": "future_demand_forecasts.csv",
    },
    {
        "label": "Machine Learning Coverage",
        "value": f"{rf_coverage_pct}%",
        "delta": f"{rf_count}/{total_products} SKUs on RF",
        "help": "Products satisfying minimum historical threshold to qualify for recursive Random Forest model.",
        "provenance": "forecast_method_summary.csv",
    },
    {
        "label": "Data Freshness Index",
        "value": f"{quality_score}%",
        "delta": f"-{stale_count} with stale records",
        "delta_color": "inverse" if stale_count > 0 else "normal",
        "help": "Percentage of SKUs whose final record extends to cutoff without inactivity gaps.",
        "provenance": "forecast_warning_summary.csv",
    },
])

st.divider()

# ---------------------------------------------------------------------------
# Visual Analytics Row 1: Demand Trend & Top Volume SKUs
# ---------------------------------------------------------------------------

col_trend, col_top = st.columns([1.6, 1], gap="medium")

with col_trend:
    fig_demand = demand_time_series(
        demand, title="Portfolio Aggregate Monthly Demand (Units)"
    )
    st.plotly_chart(fig_demand, width="stretch")

with col_top:
    fig_top = top_products_bar(
        demand, n=8, title="Top 8 SKUs by Historical Volume"
    )
    st.plotly_chart(fig_top, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Visual Analytics Row 2: Method Distribution & Data Quality Warnings
# ---------------------------------------------------------------------------

col_method, col_warn = st.columns([1, 1], gap="medium")

with col_method:
    fig_method = method_donut(method_summary)
    st.plotly_chart(fig_method, width="stretch")

with col_warn:
    fig_warn = warning_bar(warning_summary)
    st.plotly_chart(fig_warn, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Data Governance & Provenance Section
# ---------------------------------------------------------------------------

late_regime = qs_val.get("late_2017_regime_change", "")
if late_regime:
    st.info(
        f"**Dataset Governance Notice — Regime Shift (Late 2017):** {late_regime}",
        icon="ℹ️",
    )

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
