"""Explainable demand and forecast data-risk indicators."""

import plotly.express as px
import streamlit as st

from dashboard.components.kpi_cards import kpi_row, page_header
from dashboard.operational_data import (
    DATACO_SOURCE,
    activate_operational_dataset,
    load_operational_dataset,
    select_operational_source,
    source_coverage_label,
)
from src.risk.risk_score import (
    build_demand_risk_assessment,
    build_uploaded_demand_risk_assessment,
    summarize_risk_indicators,
)


page_header(
    title="Supply Chain Risk",
    subtitle="Exploratory review of demand-forecast warnings and data-quality indicators across the supply chain dataset.",
    category="OPERATIONS",
    badge_text="Data Quality Review",
    badge_variant="blue",
)

selected_source = select_operational_source()
dataset, unavailable_reason = load_operational_dataset(
    selected_source,
    st.session_state,
    include_evaluation=True,
)
if dataset is None:
    st.info(
        f"{unavailable_reason} Visit **User Demand Forecasting**, prepare the "
        "CSV, run model comparison, and generate forecasts; then select "
        "**User-uploaded dataset** here."
    )
    st.stop()

activate_operational_dataset(st.session_state, dataset)
st.caption(source_coverage_label(dataset))
if dataset.source != DATACO_SOURCE:
    st.caption("Demand-data risk monitoring for the selected user-uploaded dataset.")
st.info(
    "This page assesses demand history, forecast methods and data quality "
    "only. Supplier reliability, logistics bottlenecks, geopolitical exposure, "
    "single-source dependency and actual supply disruption are not measured."
)

if dataset.source == DATACO_SOURCE:
    assessment = build_demand_risk_assessment(
        dataset.history,
        dataset.forecasts,
        dataset.warnings,
        dataset.evaluation,
    )
    evidence_count = int(assessment["evaluation_period"].notna().sum())
    entity_label = "Products"
    indicator_filters = [
        "Stale product history",
        "Short history (<12 observed months)",
        "Baseline forecast method",
        "Internal missing months",
    ]
    filter_columns = {
        "Stale product history": "stale_history",
        "Short history (<12 observed months)": "short_history",
        "Baseline forecast method": "baseline_fallback",
        "Internal missing months": "internal_missing_months",
    }
else:
    user_data = dataset.user_data
    if user_data is None:
        raise RuntimeError("Uploaded dataset context was not provided.")
    assessment = build_uploaded_demand_risk_assessment(
        user_data.prepared.demand,
        user_data.prepared.diagnostics,
        dataset.forecasts,
        dataset.warnings,
        user_data.backtest,
    )
    evidence_count = int(assessment["backtest_origins"].gt(0).sum())
    entity_label = "Series"
    indicator_filters = [
        "Stale history",
        "Short history",
        "Simple baseline selected",
        "Internal unknown periods",
        "Unknown periods",
        "Intermittent or irregular pattern",
    ]
    filter_columns = {
        "Stale history": "stale_history",
        "Short history": "short_history",
        "Simple baseline selected": "baseline_fallback",
        "Internal unknown periods": "internal_missing_months",
        "Unknown periods": "unknown_periods",
        "Intermittent or irregular pattern": "intermittent_or_irregular",
    }
    indicator_filters.append("Trend screening signal")
    filter_columns["Trend screening signal"] = "trend_evidence"

summary = summarize_risk_indicators(assessment)
indicator_columns = [
    "stale_history",
    "short_history",
    "baseline_fallback",
    "internal_missing_months",
]
if "trend_evidence" in assessment.columns:
    indicator_columns.append("trend_evidence")
indicator_count = int(
    assessment[
        indicator_columns
    ].astype(bool).any(axis=1).sum()
)
portfolio_count = len(assessment)
kpi_row(
    [
        {
            "label": f"{entity_label} with Listed Indicators",
            "value": f"{indicator_count} / {portfolio_count}",
            "help": "Count with at least one independently listed indicator; this is not a composite risk score.",
            "provenance": "Selected dataset",
        },
        {
            "label": "Stale Histories",
            "value": int(assessment["stale_history"].sum()),
            "help": "A later period is unknown or recorded history ends before the available forecast origin.",
            "provenance": "Demand history",
        },
        {
            "label": "Short Histories",
            "value": int(assessment["short_history"].sum()),
            "help": "History is below the selected workflow's minimum diagnostic length.",
            "provenance": "Demand diagnostics",
        },
        {
            "label": "With Comparable Backtest Evidence",
            "value": f"{evidence_count} / {portfolio_count}",
            "help": "Only sufficiently comparable historical origins are counted; past errors do not guarantee future performance.",
            "provenance": (
                "Forecast evaluation outputs"
                if dataset.source == DATACO_SOURCE
                else "Rolling-origin comparison"
            ),
        },
    ]
)

st.markdown("### Indicator definitions")
definitions = {
    "Stale history": (
        "The latest prepared period is unknown or known demand ends before the "
        "forecast origin; confirm whether recent records are missing or the series stopped."
    ),
    "Short history": (
        "Too few eligible periods for the configured descriptive diagnostics; "
        "collect more observed history before relying on model comparisons."
    ),
    "Baseline forecast method": (
        "A simple naive or moving-average method was selected. This is a method "
        "description, not evidence that a complex model is more reliable."
    ),
    "Internal unknown periods": (
        "Periods between supplied observations have unknown demand and are not "
        "treated as zero."
    ),
    "Confirmed-zero periods": (
        "Periods explicitly marked as covered with zero demand; these remain "
        "distinct from unknown periods."
    ),
    "Intermittent or irregular pattern": (
        "A descriptive demand-pattern screening label, not a statistical test."
    ),
}
if dataset.source != DATACO_SOURCE:
    definitions.update(
        {
            "Trend screening signal": (
                "The existing demand diagnostics found a fitted trend meeting "
                "their configured fit and relative-change thresholds. This is "
                "a screening signal, not proof of an abrupt change or its cause."
            ),
            "Recent period change": (
                "The absolute and percentage change between the latest two "
                "consecutive known periods. Unknown periods are not bridged; "
                "percentage change is unavailable when prior demand is zero."
            ),
        }
    )
for indicator, explanation in definitions.items():
    st.markdown(f"- **{indicator}:** {explanation}")

col_chart, col_note = st.columns([1.2, 1])
with col_chart:
    st.plotly_chart(
        px.bar(
            summary,
            x="affected_products",
            y="indicator",
            orientation="h",
            title=f"Portfolio distribution of independent indicators ({entity_label.lower()})",
            labels={
                "affected_products": entity_label,
                "indicator": "Indicator",
            },
        ),
        width="stretch",
    )
with col_note:
    st.markdown("### Evaluation evidence")
    st.write(
        "Backtest metrics are shown only when the selected method has enough "
        "comparable origins under the configured evaluation rules. Missing or "
        "non-comparable metrics remain unavailable."
    )
    st.caption(
        "No composite risk score is calculated. Demand variability is "
        "descriptive and is not assigned an arbitrary risk band."
    )

st.markdown("### Series-level data reliability register")
filters = st.multiselect(
    "Show series with any selected indicator",
    indicator_filters,
    default=indicator_filters,
    key="risk_indicators",
)
selected_columns = [filter_columns[item] for item in filters]
if selected_columns:
    visible = assessment.loc[
        assessment[selected_columns].astype(bool).any(axis=1)
    ].copy()
else:
    visible = assessment.iloc[0:0].copy()

if dataset.source == DATACO_SOURCE:
    table_columns = [
        "product_id",
        "last_observed_month",
        "observed_months",
        "stale_history",
        "short_history",
        "baseline_fallback",
        "internal_missing_months",
        "mean_observed_monthly_demand",
        "observed_monthly_demand_std",
        "demand_coefficient_of_variation",
        "evaluation_period",
        "evaluation_model",
        "evaluation_mae",
        "evaluation_wape_pct",
        "baseline_comparison",
        "recommended_action",
    ]
    column_names = {
        "product_id": "Product ID",
        "last_observed_month": "Last Observed Month",
        "observed_months": "Observed Months",
        "stale_history": "Stale",
        "short_history": "Short History",
        "baseline_fallback": "Baseline Method",
        "internal_missing_months": "Internal Missing Months",
        "mean_observed_monthly_demand": "Mean Monthly Demand",
        "observed_monthly_demand_std": "Monthly Demand Std",
        "demand_coefficient_of_variation": "Demand CV (descriptive)",
        "evaluation_period": "Backtest Period",
        "evaluation_model": "Evaluated Model",
        "evaluation_mae": "Backtest MAE",
        "evaluation_wape_pct": "Backtest WAPE (%)",
        "baseline_comparison": "MAE vs Baselines",
        "recommended_action": "Recommended Investigation",
    }
else:
    table_columns = [
        "series_label",
        "frequency",
        "first_period",
        "last_known_period",
        "history_periods",
        "eligible_periods",
        "unknown_periods",
        "confirmed_zero_periods",
        "observed_zero_periods",
        "stale_history",
        "short_history",
        "internal_missing_months",
        "intermittent_or_irregular",
        "trend_evidence",
        "trend_direction",
        "trend_slope_per_period",
        "trend_r_squared",
        "recent_change_units",
        "recent_change_pct",
        "demand_pattern",
        "mean_demand_per_period",
        "demand_cv",
        "model_eligible",
        "selected_model",
        "baseline_fallback",
        "forecast_warning",
        "backtest_origins",
        "backtest_metric",
        "backtest_mae",
        "backtest_wape",
        "baseline_comparison",
        "evaluation_evidence",
        "recommended_action",
    ]
    column_names = {
        "series_label": "Product / series",
        "frequency": "Frequency",
        "first_period": "First period",
        "last_known_period": "Last known period",
        "history_periods": "History periods",
        "eligible_periods": "Eligible periods",
        "unknown_periods": "Unknown periods",
        "confirmed_zero_periods": "Confirmed-zero periods",
        "observed_zero_periods": "Observed-zero periods",
        "stale_history": "Stale",
        "short_history": "Short history",
        "internal_missing_months": "Internal unknown periods",
        "intermittent_or_irregular": "Intermittent / irregular",
        "trend_evidence": "Trend screening signal",
        "trend_direction": "Trend direction",
        "trend_slope_per_period": "Trend slope per period",
        "trend_r_squared": "Trend R²",
        "recent_change_units": "Recent change (units)",
        "recent_change_pct": "Recent change (%)",
        "demand_pattern": "Demand pattern",
        "mean_demand_per_period": "Mean demand per period",
        "demand_cv": "Demand CV",
        "model_eligible": "Forecast produced",
        "selected_model": "Selected model",
        "baseline_fallback": "Simple baseline",
        "forecast_warning": "Forecast warning",
        "backtest_origins": "Comparable origins",
        "backtest_metric": "Backtest metric",
        "backtest_mae": "Backtest MAE",
        "backtest_wape": "Backtest WAPE",
        "baseline_comparison": "Model vs baseline",
        "evaluation_evidence": "Evaluation evidence",
        "recommended_action": "Recommended investigation",
    }

st.dataframe(
    visible[table_columns].rename(columns=column_names),
    width="stretch",
    height=480,
    hide_index=True,
)
