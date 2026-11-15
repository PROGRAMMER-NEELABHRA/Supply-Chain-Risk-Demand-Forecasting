"""Product-level and single-series inventory planning."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.kpi_cards import page_header
from dashboard.operational_data import (
    DATACO_SOURCE,
    activate_operational_dataset,
    load_operational_dataset,
    select_operational_source,
    source_coverage_label,
)
from src.inventory.inventory_analysis import build_product_inventory_plan


page_header(
    title="Inventory Management",
    subtitle=(
        "Demand planning from the selected dataset and its generated forecast, "
        "with inventory assumptions supplied explicitly by you."
    ),
    category="OPERATIONS",
    badge_text="Exploratory Planning",
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
    "Current stock, supplier lead time, lead-time variability, ordering cost "
    "and holding cost are not inferred from demand data. Enter the inputs you "
    "know below; unsupported calculations remain unavailable."
    if dataset.source != DATACO_SOURCE
    else (
        "DataCo has no current-stock or supplier lead-time fields. Enter them "
        "below to enable depletion and replenishment estimates."
    )
)

if dataset.forecasts.empty:
    st.warning(
        "No eligible future forecast rows are available for inventory planning. "
        "Review the forecast warnings on User Demand Forecasting."
    )
    if not dataset.warnings.empty:
        st.dataframe(dataset.warnings, hide_index=True)
    st.stop()

for widget_key, value_key, default in (
    (
        "inventory_target_service_level",
        "inventory_target_service_level_value",
        95,
    ),
    (
        "inventory_provide_order_cost",
        "inventory_provide_order_cost_value",
        False,
    ),
    (
        "inventory_ordering_cost",
        "inventory_ordering_cost_value",
        25.0,
    ),
    (
        "inventory_provide_holding_cost",
        "inventory_provide_holding_cost_value",
        False,
    ),
    (
        "inventory_holding_cost",
        "inventory_holding_cost_value",
        2.0,
    ),
):
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(value_key, default)

product_ids = (
    sorted(
        dataset.forecasts["product_id"].dropna().unique().tolist(),
        key=str,
    )
    if "product_id" in dataset.forecasts.columns
    else []
)
has_product_ids = bool(product_ids)
input_template = pd.DataFrame(
    {
        **({"product_id": product_ids} if has_product_ids else {}),
        "current_stock": pd.Series(
            [pd.NA] * (len(product_ids) if has_product_ids else 1),
            dtype="Float64",
        ),
        "lead_time_days": pd.Series(
            [pd.NA] * (len(product_ids) if has_product_ids else 1),
            dtype="Float64",
        ),
        "lead_time_std_days": pd.Series(
            [pd.NA] * (len(product_ids) if has_product_ids else 1),
            dtype="Float64",
        ),
    }
)
saved_inputs = st.session_state.get("inventory_product_inputs_value")
if isinstance(saved_inputs, pd.DataFrame):
    input_columns = (
        "current_stock",
        "lead_time_days",
        "lead_time_std_days",
    )
    if has_product_ids and "product_id" in saved_inputs.columns:
        saved_by_product = saved_inputs.drop_duplicates(
            "product_id",
            keep="last",
        ).set_index("product_id")
        for column in input_columns:
            if column in saved_by_product.columns:
                input_template[column] = input_template["product_id"].map(
                    saved_by_product[column]
                ).astype("Float64")
    elif not has_product_ids and len(saved_inputs) == 1:
        for column in input_columns:
            if column in saved_inputs.columns:
                input_template.loc[0, column] = saved_inputs.iloc[0][column]

st.markdown("### Inventory inputs")
user_inputs = st.data_editor(
    input_template,
    hide_index=True,
    width="stretch",
    num_rows="fixed",
    disabled=["product_id"] if has_product_ids else False,
    column_config={
        "current_stock": st.column_config.NumberColumn(
            "Current stock (units)", min_value=0, step=1,
        ),
        "lead_time_days": st.column_config.NumberColumn(
            "Supplier lead time (days)", min_value=1, step=1,
        ),
        "lead_time_std_days": st.column_config.NumberColumn(
            "Lead-time standard deviation (days)", min_value=0, step=1,
        ),
    },
    key="inventory_product_inputs",
)
st.session_state["inventory_product_inputs_value"] = user_inputs.copy(
    deep=True
)

service_level = st.slider(
    "Target service level",
    min_value=50,
    max_value=99,
    step=1,
    format="%d%%",
    key="inventory_target_service_level",
    help="Used only for the safety-stock normal quantile; it is not an observed service level.",
)
st.session_state["inventory_target_service_level_value"] = service_level
cost_col1, cost_col2 = st.columns(2)
with cost_col1:
    provide_order_cost = st.checkbox(
        "Provide ordering cost",
        key="inventory_provide_order_cost",
    )
    st.session_state["inventory_provide_order_cost_value"] = provide_order_cost
    ordering_cost = (
        st.number_input(
            "Ordering cost per order",
            min_value=0.01,
            key="inventory_ordering_cost",
        )
        if provide_order_cost
        else None
    )
    if ordering_cost is not None:
        st.session_state["inventory_ordering_cost_value"] = ordering_cost
with cost_col2:
    provide_holding_cost = st.checkbox(
        "Provide annual holding cost",
        key="inventory_provide_holding_cost",
    )
    st.session_state["inventory_provide_holding_cost_value"] = provide_holding_cost
    holding_cost = (
        st.number_input(
            "Holding cost per unit per year",
            min_value=0.01,
            key="inventory_holding_cost",
        )
        if provide_holding_cost
        else None
    )
    if holding_cost is not None:
        st.session_state["inventory_holding_cost_value"] = holding_cost

plans = build_product_inventory_plan(
    dataset.history,
    dataset.forecasts,
    user_inputs,
    target_service_level=service_level / 100,
    ordering_cost=ordering_cost,
    holding_cost_per_unit_per_year=holding_cost,
    frequency=dataset.frequency,
)

st.markdown("### Product planning table")
st.caption(
    f"Demand periods use {dataset.frequency} frequency. Unknown periods are "
    "excluded from demand averages and prevent variability calculations when "
    "they create gaps in the observed series."
)
display_columns = [
    "product_id",
    "observed_periods",
    "average_observed_demand_per_period",
    "average_forecast_demand_per_period",
    "current_stock",
    "lead_time_days",
    "lead_time_demand_units",
    "safety_stock_units",
    "reorder_point_units",
    "eoq_units",
    "stockout_date",
    "stockout_before_replenishment",
    "safety_stock_status",
    "eoq_status",
    "lead_time_demand_basis",
    "depletion_status",
]
joined = plans.copy()
if has_product_ids:
    joined = joined.merge(
        user_inputs[["product_id", "current_stock", "lead_time_days"]],
        on="product_id",
        how="left",
    )
else:
    for column in ("current_stock", "lead_time_days"):
        joined[column] = user_inputs.iloc[0][column]
display = joined[display_columns].rename(
    columns={
        "product_id": "Product ID",
        "observed_periods": "Observed periods",
        "average_observed_demand_per_period": "Avg observed demand per period",
        "average_forecast_demand_per_period": "Avg forecast demand per period",
        "current_stock": "User stock input",
        "lead_time_days": "User lead time (days)",
        "lead_time_demand_units": "Lead-time demand",
        "safety_stock_units": "Safety stock",
        "reorder_point_units": "Reorder point",
        "eoq_units": "EOQ",
        "stockout_date": "Projected stockout",
        "stockout_before_replenishment": "Stockout before lead time",
        "safety_stock_status": "Safety stock availability",
        "eoq_status": "EOQ availability",
        "lead_time_demand_basis": "Lead-time demand basis",
        "depletion_status": "Depletion estimate",
    }
)
if not has_product_ids:
    display = display.drop(columns=["Product ID"])
st.dataframe(
    display.style.format(
        {
            column: "{:,.1f}"
            for column in (
                "Avg observed demand per period",
                "Avg forecast demand per period",
                "User stock input",
                "Lead-time demand",
                "Safety stock",
                "Reorder point",
                "EOQ",
            )
            if column in display.columns
        },
        na_rep="Unavailable",
    ),
    width="stretch",
    height=420,
    hide_index=True,
)

if has_product_ids:
    selected_product = st.selectbox(
        "Product to inspect",
        product_ids,
        key="inventory_chart_product",
    )
    product_forecasts = dataset.forecasts.loc[
        dataset.forecasts["product_id"] == selected_product
    ].sort_values("forecast_period")
    history = dataset.history.loc[
        dataset.history["product_id"] == selected_product
    ].sort_values("period")
    selected_plan = plans.loc[plans["product_id"] == selected_product].iloc[0]
else:
    selected_product = None
    product_forecasts = dataset.forecasts.sort_values("forecast_period")
    history = dataset.history.sort_values("period")
    selected_plan = plans.iloc[0]

chart_history = history[["period", "demand_quantity"]].rename(
    columns={"period": "Period", "demand_quantity": "Demand"}
)
chart_forecast = product_forecasts[
    ["forecast_period", "predicted_demand"]
].rename(
    columns={"forecast_period": "Period", "predicted_demand": "Demand"}
)
chart_history["Series"] = "Known history"
chart_forecast["Series"] = "Forecast"
chart_data = pd.concat([chart_history, chart_forecast], ignore_index=True)
st.plotly_chart(
    px.line(
        chart_data,
        x="Period",
        y="Demand",
        color="Series",
        markers=True,
        title=(
            f"Product {selected_product}: history and forecast"
            if has_product_ids
            else "Single series: history and forecast"
        ),
    ),
    width="stretch",
)

if pd.notna(selected_plan["stockout_before_replenishment"]) and bool(
    selected_plan["stockout_before_replenishment"]
):
    st.warning(
        "The series is projected to deplete before the user-entered supplier "
        "lead time ends. This is a forecast-based estimate, not a confirmed stockout."
    )
for field, label in [
    ("safety_stock_status", "Safety stock"),
    ("eoq_status", "EOQ"),
    ("lead_time_demand_basis", "Lead-time demand"),
    ("depletion_status", "Stock depletion"),
]:
    st.caption(f"{label}: {selected_plan[field]}")
