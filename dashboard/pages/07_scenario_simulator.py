"""Deterministic demand scenarios based on the selected forecast."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.kpi_cards import kpi_row, page_header
from dashboard.operational_data import (
    DATACO_SOURCE,
    activate_operational_dataset,
    load_operational_dataset,
    select_operational_source,
    source_coverage_label,
)
from dashboard.styles.theme import get_plotly_layout, get_theme_tokens
from src.scenarios.simulation import (
    aggregate_portfolio_scenarios,
    build_demand_scenarios,
)


page_header(
    title="Scenario Simulator",
    subtitle=(
        "Deterministic demand comparisons based on the selected dataset's "
        "actual generated forecast horizon."
    ),
    category="OPERATIONS",
    badge_text="Deterministic Scenarios",
    badge_variant="blue",
)

selected_source = select_operational_source()
dataset, unavailable_reason = load_operational_dataset(
    selected_source,
    st.session_state,
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
st.info(
    "Scenarios scale the saved forecast values; they do not retrain the model "
    "or change source data. This is not Monte Carlo simulation, a calibrated "
    "uncertainty interval, a supplier shock model, or a demand-elasticity estimate."
)
st.caption(
    "Defaults are optimistic +10% and pessimistic -10%; they are transparent "
    "planning assumptions, not estimates of forecast uncertainty."
)
if dataset.forecasts.empty:
    st.warning(
        "No eligible future forecast rows are available for scenario analysis. "
        "Review forecast warnings on User Demand Forecasting."
    )
    if not dataset.warnings.empty:
        st.dataframe(dataset.warnings, hide_index=True)
    st.stop()

product_ids = (
    sorted(dataset.forecasts["product_id"].dropna().unique().tolist(), key=str)
    if "product_id" in dataset.forecasts.columns
    else []
)
has_product_ids = bool(product_ids)
scope_options = list(range(len(product_ids) + 1)) if has_product_ids else [0]
if st.session_state.get("scenario_product") not in scope_options:
    st.session_state.pop("scenario_product", None)
choice = st.selectbox(
    "Analysis scope",
    scope_options,
    format_func=(
        lambda index: "Portfolio"
        if index == 0
        else f"Product {product_ids[index - 1]}"
    )
    if has_product_ids
    else lambda _: "Single series",
    key="scenario_product",
)
selected_product = (
    product_ids[choice - 1] if has_product_ids and choice > 0 else None
)

control_col1, control_col2, control_col3 = st.columns(3)
with control_col1:
    optimistic_multiplier = st.slider(
        "Optimistic multiplier",
        min_value=1.0,
        max_value=2.5,
        value=1.10,
        step=0.05,
        key="scenario_optimistic",
    )
with control_col2:
    pessimistic_multiplier = st.slider(
        "Pessimistic multiplier",
        min_value=0.5,
        max_value=1.0,
        value=0.90,
        step=0.05,
        key="scenario_pessimistic",
    )
with control_col3:
    user_multiplier = st.slider(
        "User multiplier",
        min_value=0.5,
        max_value=2.5,
        value=1.0,
        step=0.05,
        key="scenario_user",
    )

product_scenarios = build_demand_scenarios(
    dataset.forecasts,
    product_id=selected_product,
    optimistic_multiplier=optimistic_multiplier,
    pessimistic_multiplier=pessimistic_multiplier,
    user_multiplier=user_multiplier,
)
if selected_product is None:
    display_scenarios = aggregate_portfolio_scenarios(product_scenarios)
else:
    display_scenarios = product_scenarios

baseline_total = float(display_scenarios["baseline_demand"].sum())
user_total = float(display_scenarios["user_scenario_demand"].sum())
delta_units = user_total - baseline_total
delta_pct = delta_units / baseline_total * 100 if baseline_total else None
first_period = pd.to_datetime(display_scenarios["forecast_month"]).min()
last_period = pd.to_datetime(display_scenarios["forecast_month"]).max()
kpi_row(
    [
        {
            "label": "Baseline Forecast Total",
            "value": f"{baseline_total:,.1f} units",
            "help": "Sum of generated forecast rows over the available horizon.",
            "provenance": dataset.name,
        },
        {
            "label": "User Scenario Total",
            "value": f"{user_total:,.1f} units",
            "delta": (
                f"{delta_units:+,.1f} units ({delta_pct:+.1f}%)"
                if delta_pct is not None
                else f"{delta_units:+,.1f} units; percentage unavailable at zero baseline"
            ),
            "delta_color": "normal" if delta_units >= 0 else "inverse",
            "help": "Deterministic scaling of the selected baseline forecast.",
            "provenance": "User multiplier",
        },
        {
            "label": "Forecast Horizon",
            "value": f"{first_period:%Y-%m-%d} – {last_period:%Y-%m-%d}",
            "help": "The original forecast dates are unchanged by scenario multipliers.",
            "provenance": f"{dataset.frequency.capitalize()} forecast periods",
        },
    ]
)

theme_tokens = get_theme_tokens()
series = [
    ("baseline_demand", "Baseline", theme_tokens["historical_demand"], "solid"),
    ("optimistic_demand", "Optimistic", theme_tokens["forecast"], "dash"),
    ("pessimistic_demand", "Pessimistic", theme_tokens["warning"], "dot"),
    (
        "user_scenario_demand",
        "User scenario",
        theme_tokens["secondary_comparison"],
        "dashdot",
    ),
]
figure = go.Figure()
for column, name, color, dash in series:
    figure.add_trace(
        go.Scatter(
            x=display_scenarios["forecast_month"],
            y=display_scenarios[column],
            name=name,
            mode="lines+markers",
            line={"color": color, "dash": dash},
        )
    )
layout = dict(get_plotly_layout())
scope_label = (
    "Portfolio"
    if selected_product is None and has_product_ids
    else f"Product {selected_product}"
    if selected_product is not None
    else "Single series"
)
layout.update(
    title=f"{scope_label} {dataset.frequency} demand scenarios",
    xaxis_title=f"Forecast {dataset.frequency} period",
    yaxis_title="Demand units",
    legend={"orientation": "h"},
)
figure.update_layout(**layout)
st.plotly_chart(
    figure,
    width="stretch",
    alt=(
        f"{scope_label} demand scenarios from {first_period:%Y-%m-%d} through "
        f"{last_period:%Y-%m-%d} at {dataset.frequency} frequency. The horizontal "
        "axis is forecast period and the vertical axis is demand units; four "
        "traces compare baseline, optimistic, pessimistic, and user scenarios."
    ),
)

st.markdown(f"### {dataset.frequency.capitalize()} scenario comparison")
table = display_scenarios[
    [
        "forecast_month",
        "baseline_demand",
        "optimistic_demand",
        "optimistic_delta_units",
        "optimistic_delta_pct",
        "pessimistic_demand",
        "pessimistic_delta_units",
        "pessimistic_delta_pct",
        "user_scenario_demand",
        "user_scenario_delta_units",
        "user_scenario_delta_pct",
    ]
].rename(
    columns={
        "forecast_month": "Forecast period",
        "baseline_demand": "Baseline demand",
        "optimistic_demand": "Optimistic demand",
        "optimistic_delta_units": "Optimistic Δ units",
        "optimistic_delta_pct": "Optimistic Δ (%)",
        "pessimistic_demand": "Pessimistic demand",
        "pessimistic_delta_units": "Pessimistic Δ units",
        "pessimistic_delta_pct": "Pessimistic Δ (%)",
        "user_scenario_demand": "User scenario demand",
        "user_scenario_delta_units": "User Δ units",
        "user_scenario_delta_pct": "User Δ (%)",
    }
)
st.dataframe(
    table.style.format(
        {
            column: "{:,.1f}"
            for column in table.columns
            if column != "Forecast period"
        },
        na_rep="Unavailable",
    ),
    width="stretch",
    hide_index=True,
)
st.download_button(
    "Download scenario CSV",
    data=display_scenarios.to_csv(index=False),
    file_name=(
        "dataco_demand_scenarios.csv"
        if dataset.source == DATACO_SOURCE and selected_product is None
        else (
            f"dataco_product_{selected_product}_scenarios.csv"
            if dataset.source == DATACO_SOURCE
            else "user_upload_demand_scenarios.csv"
        )
    ),
    mime="text/csv",
)
