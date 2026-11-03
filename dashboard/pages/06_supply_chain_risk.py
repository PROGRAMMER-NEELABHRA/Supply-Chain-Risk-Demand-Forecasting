"""Supply Chain Risk — Disruption indicators and data-quality risk audit (Under Development)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.kpi_cards import kpi_row, page_header, under_development_banner
from dashboard.data.loader import (
    load_forecast_data_warnings,
    load_forecast_warning_summary,
    load_future_forecasts,
)
from dashboard.styles.theme import get_plotly_layout, get_theme_tokens

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

warnings_df = load_forecast_data_warnings()
warning_summary = load_forecast_warning_summary()
forecasts = load_future_forecasts()
t = get_theme_tokens()

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    title="Supply Chain Risk",
    subtitle="Enterprise risk exposure, supplier vulnerability, and operational data reliability",
    category="OPERATIONS",
    badge_text="In Development",
    badge_variant="orange",
)

# ---------------------------------------------------------------------------
# Development Governance Notice
# ---------------------------------------------------------------------------

under_development_banner(
    module_title="Composite Supply Chain Risk Engine",
    target_module="src/risk/risk_score.py",
    description=(
        "The end-to-end risk module is scheduled to synthesize multi-tier supplier lead times, "
        "logistics bottleneck vulnerability, single-source dependency indexes, and geopolitical disruptions "
        "into a composite supply chain risk score."
    ),
    disclaimer=(
        "The indicators displayed below reflect data-quality uncertainty flags (stale time-series, "
        "short history fallback, and intermittent record gaps) extracted from the forecasting pipeline. "
        "They represent input data integrity only, NOT a validated supplier audit or logistics risk rating."
    ),
)

st.divider()

# ---------------------------------------------------------------------------
# Data-Quality Risk KPIs (Proxy Signals)
# ---------------------------------------------------------------------------

total_warned = int(warnings_df["product_id"].nunique())
stale_count = int(
    warning_summary.loc[
        warning_summary["warning_category"] == "STALE_HISTORY", "product_count"
    ].sum()
)
baseline_count = int(
    warning_summary.loc[
        warning_summary["warning_category"] == "BASELINE_METHOD", "product_count"
    ].sum()
)
gap_count = int(
    warning_summary.loc[
        warning_summary["warning_category"] == "INTERNAL_MISSING_MONTHS", "product_count"
    ].sum()
)

kpi_row([
    {
        "label": "SKUs with Data Warnings",
        "value": f"{total_warned}",
        "help": "Products flagged with at least one data reliability warning.",
        "provenance": "forecast_data_warnings.csv",
    },
    {
        "label": "Stale History (High Uncertainty)",
        "value": f"{stale_count}",
        "help": "Series whose final historical record occurred prior to the forecast cutoff.",
        "provenance": "STALE_HISTORY category",
    },
    {
        "label": "Baseline Fallback (Medium)",
        "value": f"{baseline_count}",
        "help": "Series lacking sufficient depth for Random Forest training.",
        "provenance": "BASELINE_METHOD category",
    },
    {
        "label": "Internal Gaps (Medium)",
        "value": f"{gap_count}",
        "help": "Series exhibiting non-consecutive monthly observations.",
        "provenance": "INTERNAL_MISSING_MONTHS category",
    },
])

st.divider()

# ---------------------------------------------------------------------------
# Visual Exposure & Breakdown
# ---------------------------------------------------------------------------

total_products = int(forecasts["product_id"].nunique())
risk_pct = round(total_warned / total_products * 100, 1) if total_products else 0.0

col_gauge, col_table = st.columns([1, 1.6], gap="medium")

with col_gauge:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={"text": "Portfolio Data Risk Exposure (%)", "font": {"color": t["text_primary"], "size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": t["text_muted"], "tickfont": {"size": 10}},
                "bar": {"color": t["danger_red"]},
                "steps": [
                    {"range": [0, 33], "color": "rgba(16,185,129,0.15)"},
                    {"range": [33, 66], "color": "rgba(245,158,11,0.15)"},
                    {"range": [66, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": t["accent_orange"], "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
                "bgcolor": t["bg_card"],
                "bordercolor": t["border_color"],
            },
            number={"suffix": "%", "font": {"color": t["text_primary"], "size": 32}},
        )
    )
    gauge_layout = dict(**get_plotly_layout())
    gauge_layout.update(height=280, margin=dict(l=20, r=20, t=50, b=20))
    fig_gauge.update_layout(**gauge_layout)
    st.plotly_chart(fig_gauge, width="stretch")
    st.caption("Proportion of portfolio SKUs exhibiting at least one empirical data-quality flag.")

with col_table:
    st.markdown("### Warning Category Distribution")
    label_map = {
        "STALE_HISTORY": "Stale History (Inactivity)",
        "BASELINE_METHOD": "Baseline Fallback (Insufficient Depth)",
        "INTERNAL_MISSING_MONTHS": "Internal Sequence Gaps",
    }
    ws_display = warning_summary.copy()
    ws_display["warning_category"] = ws_display["warning_category"].map(label_map).fillna(ws_display["warning_category"])
    ws_display.columns = ["Warning Category", "Affected SKUs", "Frequency Count"]
    st.dataframe(ws_display, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Per-Product Risk Register Table
# ---------------------------------------------------------------------------

st.markdown("### Data Reliability Audit Register")

st.sidebar.markdown("### Risk Filters")
risk_filter = st.sidebar.multiselect(
    "Warning Classification",
    ["STALE_HISTORY", "BASELINE_METHOD", "INTERNAL_MISSING_MONTHS"],
    default=["STALE_HISTORY", "BASELINE_METHOD", "INTERNAL_MISSING_MONTHS"],
    key="risk_warn_filter",
)

risk_display = warnings_df[warnings_df["warning_category"].isin(risk_filter)].copy()
risk_display["warning_category"] = risk_display["warning_category"].map(label_map).fillna(
    risk_display["warning_category"]
)
risk_display["last_recorded_month"] = pd.to_datetime(
    risk_display["last_recorded_month"]
).dt.strftime("%b %Y")

risk_display = risk_display.rename(columns={
    "product_id": "Product ID",
    "warning_category": "Risk Flag",
    "last_recorded_month": "Last Observation",
    "months_since_last_record": "Months Inactive",
    "forecast_method": "Assigned Model",
    "forecast_confidence": "Confidence",
})

st.dataframe(
    risk_display[[
        "Product ID",
        "Risk Flag",
        "Last Observation",
        "Months Inactive",
        "Confidence",
        "Assigned Model",
    ]],
    width="stretch",
    height=340,
    hide_index=True,
)

st.caption(
    "Audit Notice: This table catalogs data-quality anomalies to inform forecasters of model uncertainty. "
    "It is NOT an operational supplier or logistics risk assessment."
)
