"""Shared data-source selection and adapters for operational dashboard pages."""

import hashlib
from dataclasses import dataclass
from typing import MutableMapping

import pandas as pd
import streamlit as st

from dashboard.data.loader import (
    load_forecast_data_warnings,
    load_forecast_error_by_product,
    load_future_forecasts,
    load_monthly_demand,
)
from dashboard.user_demand_forecasting import (
    UserOperationalData,
    build_forecast_export,
    user_operational_data,
)


DATACO_SOURCE = "DataCo demonstration dataset"
USER_UPLOAD_SOURCE = "User-uploaded dataset"
_SOURCE_KEY = "operational_dashboard_source"
_SOURCE_WIDGET_KEY = "operational_dashboard_source_selector"
_DATASET_CONTEXT_KEY = "operational_dashboard_dataset_context"
_SOURCES = (DATACO_SOURCE, USER_UPLOAD_SOURCE)
_OPERATIONAL_WIDGET_KEYS = (
    "inventory_product_inputs",
    "inventory_product_inputs_value",
    "inventory_chart_product",
    "inventory_target_service_level",
    "inventory_target_service_level_value",
    "inventory_provide_order_cost",
    "inventory_provide_order_cost_value",
    "inventory_ordering_cost",
    "inventory_ordering_cost_value",
    "inventory_provide_holding_cost",
    "inventory_provide_holding_cost_value",
    "inventory_holding_cost",
    "inventory_holding_cost_value",
    "risk_indicators",
    "scenario_product",
    "scenario_optimistic",
    "scenario_pessimistic",
    "scenario_user",
)


def set_operational_source(
    session: MutableMapping[str, object],
    source: str,
) -> None:
    """Set the shared operational source and its selector value."""
    if source not in _SOURCES:
        raise ValueError(f"Unsupported operational data source: {source!r}")
    session[_SOURCE_KEY] = source
    session[_SOURCE_WIDGET_KEY] = source


@dataclass(frozen=True)
class OperationalDataset:
    """Normalized, in-memory view of one operational dashboard data source."""

    source: str
    context_id: str
    name: str
    frequency: str
    history: pd.DataFrame
    forecasts: pd.DataFrame
    warnings: pd.DataFrame
    evaluation: pd.DataFrame
    user_data: UserOperationalData | None
    coverage_start: pd.Timestamp | None
    coverage_end: pd.Timestamp | None


def select_operational_source() -> str:
    """Render the source selector shared by the three operations pages."""
    if st.session_state.get(_SOURCE_KEY) not in _SOURCES:
        st.session_state[_SOURCE_KEY] = DATACO_SOURCE
    if st.session_state.get(_SOURCE_WIDGET_KEY) not in _SOURCES:
        st.session_state[_SOURCE_WIDGET_KEY] = st.session_state[_SOURCE_KEY]

    def remember_source() -> None:
        st.session_state[_SOURCE_KEY] = st.session_state[_SOURCE_WIDGET_KEY]

    source = st.selectbox(
        "Data source",
        _SOURCES,
        key=_SOURCE_WIDGET_KEY,
        on_change=remember_source,
    )
    st.session_state[_SOURCE_KEY] = source
    return source


def load_operational_dataset(
    source: str,
    session: MutableMapping[str, object],
    *,
    include_evaluation: bool = False,
) -> tuple[OperationalDataset | None, str | None]:
    """Load DataCo outputs or a signature-verified user-upload result bundle."""
    if source == DATACO_SOURCE:
        history = load_monthly_demand().copy()
        history["period"] = pd.to_datetime(history["month"], errors="raise")
        history["demand_quantity"] = pd.to_numeric(
            history["demand_units"], errors="raise"
        )
        history["observation_status"] = "observed_demand"
        history["training_eligible"] = True
        forecasts = load_future_forecasts().copy()
        forecasts["forecast_period"] = pd.to_datetime(
            forecasts["forecast_month"], errors="raise"
        )
        forecasts["predicted_demand"] = pd.to_numeric(
            forecasts["forecast_demand"], errors="raise"
        )
        warnings = (
            load_forecast_data_warnings()
            if include_evaluation
            else pd.DataFrame()
        )
        evaluation = (
            load_forecast_error_by_product()
            if include_evaluation
            else pd.DataFrame()
        )
        return (
            OperationalDataset(
                source=DATACO_SOURCE,
                context_id=DATACO_SOURCE,
                name=DATACO_SOURCE,
                frequency="monthly",
                history=history,
                forecasts=forecasts,
                warnings=warnings,
                evaluation=evaluation,
                user_data=None,
                coverage_start=pd.Timestamp(history["period"].min()),
                coverage_end=pd.Timestamp(history["period"].max()),
            ),
            None,
        )
    if source != USER_UPLOAD_SOURCE:
        raise ValueError(f"Unsupported operational data source: {source!r}")

    user_data, reason = user_operational_data(session)
    if user_data is None:
        return None, reason

    demand = user_data.prepared.demand
    history = demand.data.copy(deep=True)
    history["month"] = history["period"]
    history["demand_units"] = history["demand_quantity"]
    forecast_export = build_forecast_export(
        user_data.forecasts,
        user_data.backtest,
    )
    context_identity = (
        user_data.upload_hash,
        user_data.preparation_signature,
        user_data.backtest_signature,
        session.get("user_demand_forecasting_forecast_signature"),
    )
    context_id = hashlib.sha256(
        repr(context_identity).encode("utf-8")
    ).hexdigest()
    forecasts = forecast_export.rename(
        columns={
            "period": "forecast_period",
            "predicted_demand": "forecast_demand",
        }
    )
    forecasts["period"] = forecasts["forecast_period"]
    forecasts["predicted_demand"] = forecasts["forecast_demand"]
    forecasts["forecast_month"] = forecasts["forecast_period"]
    forecasts["forecast_method"] = forecasts.get(
        "model_name",
        pd.Series(index=forecasts.index, dtype="object"),
    )
    warning_rows = user_data.forecast_warnings.rename(
        columns={"warning": "warning_message"}
    ).copy()
    warning_rows["warning_category"] = "FORECAST_UNAVAILABLE"
    quality_date_range = demand.quality_report.date_range
    return (
        OperationalDataset(
            source=USER_UPLOAD_SOURCE,
            context_id=context_id,
            name=user_data.filename,
            frequency=demand.frequency,
            history=history,
            forecasts=forecasts,
            warnings=warning_rows,
            evaluation=pd.DataFrame(),
            user_data=user_data,
            coverage_start=(
                pd.Timestamp(quality_date_range[0])
                if quality_date_range
                else None
            ),
            coverage_end=(
                pd.Timestamp(quality_date_range[1])
                if quality_date_range
                else None
            ),
        ),
        None,
    )


def source_coverage_label(dataset: OperationalDataset) -> str:
    """Describe the actual input coverage and generated forecast horizon."""
    history = (
        f"{dataset.coverage_start:%Y-%m-%d}–{dataset.coverage_end:%Y-%m-%d}"
        if dataset.coverage_start is not None and dataset.coverage_end is not None
        else "unavailable"
    )
    forecast_periods = dataset.forecasts.get("forecast_period")
    if forecast_periods is not None and not forecast_periods.empty:
        first = pd.to_datetime(forecast_periods).min()
        last = pd.to_datetime(forecast_periods).max()
        horizon = f"{first:%Y-%m-%d}–{last:%Y-%m-%d}"
    else:
        horizon = "unavailable"
    has_product_ids = (
        "product_id" in dataset.history.columns
        and dataset.history["product_id"].notna().any()
    )
    series_structure = (
        "Product-level"
        if has_product_ids
        else "Aggregate-only single series"
    )
    source_name = (
        f"{USER_UPLOAD_SOURCE}: {dataset.name}"
        if dataset.source == USER_UPLOAD_SOURCE
        else dataset.name
    )
    return (
        f"Source: {source_name} · Frequency: {dataset.frequency} · "
        f"Series structure: {series_structure} · "
        f"Historical coverage: {history} · Forecast horizon: {horizon}"
    )


def activate_operational_dataset(
    session: MutableMapping[str, object],
    dataset: OperationalDataset,
) -> None:
    """Clear page inputs when their selected dataset or preparation changes."""
    if session.get(_DATASET_CONTEXT_KEY) == dataset.context_id:
        return
    for key in _OPERATIONAL_WIDGET_KEYS:
        session.pop(key, None)
    session[_DATASET_CONTEXT_KEY] = dataset.context_id
