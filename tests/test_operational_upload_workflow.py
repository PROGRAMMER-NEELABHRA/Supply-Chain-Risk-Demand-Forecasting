from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dashboard.operational_data as operational_data
from dashboard.operational_data import (
    DATACO_SOURCE,
    USER_UPLOAD_SOURCE,
    activate_operational_dataset,
    load_operational_dataset,
    set_operational_source,
    source_coverage_label,
)
from dashboard.user_demand_forecasting import (
    compare_user_demand_models,
    generate_user_forecasts,
    prepare_user_upload,
    store_uploaded_csv,
    user_operational_data,
)
from src.inventory.inventory_analysis import build_product_inventory_plan
from src.risk.risk_score import (
    build_uploaded_demand_risk_assessment,
    summarize_risk_indicators,
)
from src.scenarios.simulation import (
    aggregate_portfolio_scenarios,
    build_demand_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "dashboard" / "pages"


def _ready_session(
    frequency: str = "monthly",
    *,
    products: bool = True,
    missing_period: bool = False,
    confirmed_zeros: bool = False,
    trend: bool = False,
) -> dict[str, object]:
    frequency_spec = {
        "daily": ("2023-01-01", "D"),
        "weekly": ("2023-01-02", "W-MON"),
        "monthly": ("2022-01-01", "MS"),
    }
    start, pandas_frequency = frequency_spec[frequency]
    periods = pd.date_range(start, periods=24, freq=pandas_frequency)
    if missing_period:
        periods = periods.delete(12)
    rows = []
    for product in (("A", "B") if products else (None,)):
        for index, period in enumerate(periods):
            row = {
                "date": period,
                "demand": float(5 + index * 2 if trend else 5 + index % 4),
            }
            if products:
                row["sku"] = product
            rows.append(row)
    source = pd.DataFrame(rows)
    content = source.to_csv(index=False).encode("utf-8")
    session: dict[str, object] = {}
    uploaded = store_uploaded_csv(session, "sample_demand.csv", content)
    product_column = "sku" if products else None
    prepared = prepare_user_upload(
        uploaded,
        date_column="date",
        demand_column="demand",
        product_column=product_column,
        frequency=frequency,
        complete_source_coverage=confirmed_zeros,
        confirm_missing_periods_are_zero=confirmed_zeros,
    )
    signature = (
        session["user_demand_forecasting_upload_hash"],
        "date",
        "demand",
        product_column,
        frequency,
        False,
        confirmed_zeros,
        confirmed_zeros,
        None,
        None,
        False,
    )
    session["user_demand_forecasting_prepared"] = prepared
    session["user_demand_forecasting_preparation_signature"] = signature
    session["user_demand_forecasting_current_preparation_signature"] = signature

    horizon = 2
    min_train_periods = 4
    min_successful_origins = 1
    min_comparable_origins = 1
    selection_metric = "mae"
    min_success_fraction = 0.1
    backtest = compare_user_demand_models(
        prepared.demand,
        prepared.diagnostics,
        horizon=horizon,
        min_train_periods=min_train_periods,
        min_successful_origins=min_successful_origins,
        min_comparable_origins=min_comparable_origins,
        min_success_fraction=min_success_fraction,
        selection_metric=selection_metric,
    )
    backtest_signature = (
        signature,
        horizon,
        min_train_periods,
        min_successful_origins,
        min_comparable_origins,
        selection_metric,
        min_success_fraction,
    )
    session["user_demand_forecasting_backtest"] = backtest
    session["user_demand_forecasting_backtest_signature"] = backtest_signature
    session["user_demand_forecasting_current_backtest_signature"] = (
        backtest_signature
    )
    selected_models = {
        selection.product_id: selection.selected_model
        for selection in backtest.selections
    }
    forecasts, warnings = generate_user_forecasts(
        prepared.demand,
        prepared.diagnostics,
        backtest,
        horizon=horizon,
        selected_models=selected_models,
    )
    forecast_signature = (backtest_signature, tuple(selected_models.items()))
    session["user_demand_forecasting_forecasts"] = forecasts
    session["user_demand_forecasting_forecast_warnings"] = warnings
    session["user_demand_forecasting_forecast_signature"] = forecast_signature
    session["user_demand_forecasting_current_forecast_signature"] = (
        forecast_signature
    )
    return session


@pytest.mark.parametrize("frequency", ["daily", "weekly", "monthly"])
def test_uploaded_frequency_flows_to_inventory_and_scenario_outputs(frequency):
    session = _ready_session(frequency)
    dataset, reason = load_operational_dataset(USER_UPLOAD_SOURCE, session)

    assert reason is None
    assert dataset is not None
    assert dataset.frequency == frequency
    assert dataset.name == "sample_demand.csv"
    assert len(dataset.forecasts) == 4

    inputs = pd.DataFrame(
        {
            "product_id": ["A", "B"],
            "current_stock": [pd.NA, pd.NA],
            "lead_time_days": [pd.NA, pd.NA],
            "lead_time_std_days": [pd.NA, pd.NA],
        }
    )
    plans = build_product_inventory_plan(
        dataset.history,
        dataset.forecasts,
        inputs,
        target_service_level=0.95,
        ordering_cost=None,
        holding_cost_per_unit_per_year=None,
        frequency=frequency,
    )
    assert plans["product_id"].tolist() == ["A", "B"]
    assert plans["average_observed_demand_per_period"].notna().all()
    assert plans["lead_time_demand_units"].isna().all()

    scenarios = build_demand_scenarios(dataset.forecasts)
    portfolio = aggregate_portfolio_scenarios(scenarios)
    csv_result = pd.read_csv(
        StringIO(portfolio.to_csv(index=False))
    )
    assert csv_result["forecast_month"].tolist() == portfolio[
        "forecast_month"
    ].astype(str).tolist()
    assert csv_result["baseline_demand"].sum() == pytest.approx(
        portfolio["baseline_demand"].sum()
    )


def test_single_series_without_product_ids_remains_a_single_series():
    session = _ready_session("weekly", products=False)
    dataset, reason = load_operational_dataset(USER_UPLOAD_SOURCE, session)

    assert reason is None
    assert dataset is not None
    assert "product_id" not in dataset.history.columns
    assert dataset.forecasts["product_id"].isna().all()

    plans = build_product_inventory_plan(
        dataset.history,
        dataset.forecasts,
        pd.DataFrame(
            {
                "current_stock": [pd.NA],
                "lead_time_days": [pd.NA],
                "lead_time_std_days": [pd.NA],
            }
        ),
        target_service_level=0.95,
        ordering_cost=None,
        holding_cost_per_unit_per_year=None,
        frequency="weekly",
    )
    assert len(plans) == 1
    assert plans.loc[0, "product_id"] is None

    scenarios = build_demand_scenarios(dataset.forecasts)
    assert len(scenarios) == len(dataset.forecasts)
    assert scenarios["baseline_demand"].tolist() == dataset.forecasts[
        "forecast_demand"
    ].tolist()


@pytest.mark.parametrize("confirmed_zeros", [False, True])
def test_risk_keeps_unknown_and_confirmed_zero_periods_distinct(confirmed_zeros):
    session = _ready_session(
        "monthly",
        products=False,
        missing_period=True,
        confirmed_zeros=confirmed_zeros,
    )
    dataset, reason = load_operational_dataset(USER_UPLOAD_SOURCE, session)
    assert reason is None
    assert dataset is not None and dataset.user_data is not None

    risk = build_uploaded_demand_risk_assessment(
        dataset.user_data.prepared.demand,
        dataset.user_data.prepared.diagnostics,
        dataset.forecasts,
        dataset.warnings,
        dataset.user_data.backtest,
    )
    result = risk.iloc[0]
    if confirmed_zeros:
        assert result["unknown_periods"] == 0
        assert result["confirmed_zero_periods"] == 1
    else:
        assert result["unknown_periods"] == 1
        assert result["confirmed_zero_periods"] == 0
        assert "unknown" in result["recommended_action"].lower()


def test_upload_risk_exposes_existing_trend_evidence_without_a_score():
    session = _ready_session(products=False, trend=True)
    dataset, reason = load_operational_dataset(USER_UPLOAD_SOURCE, session)
    assert reason is None
    assert dataset is not None and dataset.user_data is not None

    assessment = build_uploaded_demand_risk_assessment(
        dataset.user_data.prepared.demand,
        dataset.user_data.prepared.diagnostics,
        dataset.forecasts,
        dataset.warnings,
        dataset.user_data.backtest,
    )
    result = assessment.iloc[0]

    assert bool(result["trend_evidence"])
    assert result["trend_direction"] == "increasing"
    assert result["recent_change_units"] == pytest.approx(2)
    assert result["recent_change_pct"] == pytest.approx(2 / 49 * 100)
    assert "Investigate the fitted increasing demand trend" in result[
        "recommended_action"
    ]
    assert "risk_score" not in assessment.columns
    summary = summarize_risk_indicators(assessment)
    assert "Trend screening signal" in summary["indicator"].tolist()


def test_changed_preparation_invalidates_old_operational_forecasts():
    session = _ready_session()
    session["user_demand_forecasting_current_preparation_signature"] = (
        "changed-mapping",
    )

    result, reason = user_operational_data(session)

    assert result is None
    assert "Re-prepare" in reason
    assert "user_demand_forecasting_forecasts" not in session
    assert "user_demand_forecasting_backtest" not in session


def test_no_upload_returns_actionable_instruction_and_upload_replaces_results():
    result, reason = user_operational_data({})
    assert result is None
    assert "No user CSV" in reason

    session = _ready_session()
    store_uploaded_csv(
        session,
        "new_upload.csv",
        b"date,demand\n2024-01-01,3\n",
    )
    assert "user_demand_forecasting_prepared" not in session
    assert "user_demand_forecasting_forecasts" not in session


def test_switching_sources_does_not_use_uploaded_frames_as_dataco(monkeypatch):
    session = _ready_session(products=False)
    data_co_history = pd.DataFrame(
        {
            "month": [pd.Timestamp("2017-08-01")],
            "product_id": [7],
            "demand_units": [12.0],
        }
    )
    data_co_forecasts = pd.DataFrame(
        {
            "product_id": [7],
            "forecast_month": [pd.Timestamp("2017-09-01")],
            "forecast_demand": [13.0],
        }
    )
    monkeypatch.setattr(
        operational_data, "load_monthly_demand", lambda: data_co_history
    )
    monkeypatch.setattr(
        operational_data, "load_future_forecasts", lambda: data_co_forecasts
    )

    dataco, _ = load_operational_dataset(DATACO_SOURCE, session)
    uploaded, _ = load_operational_dataset(USER_UPLOAD_SOURCE, session)

    assert dataco is not None and uploaded is not None
    assert dataco.history["product_id"].tolist() == [7]
    assert uploaded.name == "sample_demand.csv"
    assert "product_id" not in uploaded.history.columns


def test_operational_widget_inputs_are_isolated_by_prepared_dataset_identity():
    first_session = _ready_session("monthly")
    second_session = _ready_session("weekly")
    first, first_reason = load_operational_dataset(
        USER_UPLOAD_SOURCE, first_session
    )
    second, second_reason = load_operational_dataset(
        USER_UPLOAD_SOURCE, second_session
    )
    assert first_reason is None and second_reason is None
    assert first is not None and second is not None
    assert first.context_id != second.context_id

    state = {
        "inventory_product_inputs": pd.DataFrame({"current_stock": [15]}),
        "inventory_product_inputs_value": pd.DataFrame(
            {"current_stock": [15]}
        ),
        "scenario_optimistic": 1.5,
    }
    activate_operational_dataset(state, first)
    assert "inventory_product_inputs" not in state
    assert "scenario_optimistic" not in state

    state["inventory_product_inputs"] = "user input"
    activate_operational_dataset(state, first)
    assert state["inventory_product_inputs"] == "user input"

    activate_operational_dataset(state, second)
    assert "inventory_product_inputs" not in state
    assert "inventory_product_inputs_value" not in state


def test_inventory_inputs_restore_after_widget_state_is_cleared():
    session = _ready_session(products=False)
    dataset, reason = load_operational_dataset(USER_UPLOAD_SOURCE, session)
    assert reason is None and dataset is not None
    set_operational_source(session, USER_UPLOAD_SOURCE)
    session.update(
        {
            "operational_dashboard_dataset_context": dataset.context_id,
            "inventory_product_inputs": {
                "edited_rows": {
                    0: {
                        "current_stock": 42.0,
                        "lead_time_days": 3.0,
                        "lead_time_std_days": 1.0,
                    }
                },
                "added_rows": [],
                "deleted_rows": [],
            },
            "inventory_target_service_level": 90,
            "inventory_provide_order_cost": True,
            "inventory_ordering_cost": 17.5,
            "inventory_provide_holding_cost": True,
            "inventory_holding_cost": 2.5,
        }
    )
    app = AppTest.from_file(
        str(PAGE_DIR / "05_inventory_management.py"),
        default_timeout=45,
    )
    for key, value in session.items():
        app.session_state[key] = value
    app.run()

    assert not app.exception
    persisted_values = app.session_state.to_dict()
    assert persisted_values["inventory_product_inputs_value"].loc[
        0, "current_stock"
    ] == 42
    widget_keys = {
        "inventory_product_inputs",
        "inventory_target_service_level",
        "inventory_provide_order_cost",
        "inventory_ordering_cost",
        "inventory_provide_holding_cost",
        "inventory_holding_cost",
    }
    restored_session = {
        key: value
        for key, value in session.items()
        if key not in widget_keys
    }
    for key in (
        "inventory_product_inputs_value",
        "inventory_target_service_level_value",
        "inventory_provide_order_cost_value",
        "inventory_ordering_cost_value",
        "inventory_provide_holding_cost_value",
        "inventory_holding_cost_value",
    ):
        restored_session[key] = persisted_values[key]

    restored_app = AppTest.from_file(
        str(PAGE_DIR / "05_inventory_management.py"),
        default_timeout=45,
    )
    for key, value in restored_session.items():
        restored_app.session_state[key] = value
    restored_app.run()

    assert not restored_app.exception
    inputs = restored_app.get("dataframe")[0].value
    assert inputs.loc[0, "current_stock"] == 42
    assert inputs.loc[0, "lead_time_days"] == 3
    assert restored_app.slider[0].value == 90
    assert [checkbox.value for checkbox in restored_app.checkbox] == [True, True]
    assert [widget.value for widget in restored_app.number_input] == [17.5, 2.5]


def test_source_caption_identifies_product_level_or_aggregate_series():
    product_level, reason = load_operational_dataset(
        USER_UPLOAD_SOURCE, _ready_session(products=True)
    )
    aggregate, aggregate_reason = load_operational_dataset(
        USER_UPLOAD_SOURCE, _ready_session(products=False)
    )

    assert reason is None and aggregate_reason is None
    assert product_level is not None and aggregate is not None
    assert "User-uploaded dataset: sample_demand.csv" in source_coverage_label(
        product_level
    )
    assert "Product-level" in source_coverage_label(product_level)
    assert "Aggregate-only single series" in source_coverage_label(aggregate)


@pytest.mark.parametrize(
    "filename",
    [
        "05_inventory_management.py",
        "06_supply_chain_risk.py",
        "07_scenario_simulator.py",
    ],
)
def test_operational_pages_explain_when_uploaded_data_is_unavailable(filename):
    app = AppTest.from_file(str(PAGE_DIR / filename), default_timeout=45)
    app.session_state["operational_dashboard_source"] = USER_UPLOAD_SOURCE
    app.run()

    assert not app.exception, (filename, app.exception)
    assert any(
        "Visit **User Demand Forecasting**" in item.value
        for item in app.info
    )


@pytest.mark.parametrize(
    "filename",
    [
        "05_inventory_management.py",
        "06_supply_chain_risk.py",
        "07_scenario_simulator.py",
    ],
)
def test_operational_pages_render_current_uploaded_dataset(filename):
    session = _ready_session()
    page = PAGE_DIR / filename
    app = AppTest.from_file(str(page), default_timeout=45)
    for key, value in session.items():
        app.session_state[key] = value
    app.session_state["operational_dashboard_source"] = USER_UPLOAD_SOURCE
    app.run()

    assert not app.exception, (filename, app.exception)
    assert any("sample_demand.csv" in item.value for item in app.caption)
