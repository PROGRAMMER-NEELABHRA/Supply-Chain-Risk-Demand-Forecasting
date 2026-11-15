import math
from statistics import NormalDist

import pandas as pd
import pytest

from src.inventory.inventory_analysis import (
    build_product_inventory_plan,
    calculate_inventory_plan,
)


def _series(history_values, forecast_values=(30.0, 31.0, 30.0)):
    history = pd.DataFrame(
        {
            "month": pd.date_range("2016-09-01", periods=len(history_values), freq="MS"),
            "demand_units": history_values,
        }
    )
    forecasts = pd.DataFrame(
        {
            "forecast_month": pd.date_range("2017-09-01", periods=len(forecast_values), freq="MS"),
            "forecast_demand": forecast_values,
        }
    )
    return history, forecasts


def test_inventory_plan_matches_hand_calculated_reorder_point_and_eoq():
    history, forecasts = _series([10.0] * 12, (30.0, 31.0, 30.0))

    plan = calculate_inventory_plan(
        history,
        forecasts,
        current_stock=15,
        lead_time_days=30,
        lead_time_std_days=0,
        ordering_cost=25,
        holding_cost_per_unit_per_year=2,
    )

    assert plan["average_observed_monthly_demand"] == 10
    assert plan["average_forecast_monthly_demand"] == pytest.approx(91 / 3)
    assert plan["lead_time_demand_units"] == pytest.approx(30)
    assert plan["safety_stock_units"] == pytest.approx(0)
    assert plan["reorder_point_units"] == pytest.approx(30)
    assert plan["reorder_point_includes_safety_stock"] is True
    assert plan["eoq_units"] == pytest.approx(math.sqrt(3000))
    assert pd.Timestamp(plan["stockout_date"]) == pd.Timestamp("2017-09-15")
    assert plan["stockout_before_replenishment"] is True


def test_short_or_gapped_history_withholds_safety_stock():
    history, forecasts = _series([10.0] * 5)
    history["month"] = pd.to_datetime(
        ["2016-09-01", "2016-10-01", "2016-12-01", "2017-01-01", "2017-02-01"]
    )

    plan = calculate_inventory_plan(
        history,
        forecasts,
        lead_time_days=15,
        lead_time_std_days=2,
    )

    assert plan["safety_stock_units"] is None
    assert "12 contiguous" in plan["safety_stock_status"]
    assert plan["lead_time_demand_units"] is not None
    assert plan["eoq_units"] is None


def test_safety_stock_uses_user_lead_time_variability_and_service_level():
    history, forecasts = _series([8.0, 10.0, 12.0] * 4)
    plan = calculate_inventory_plan(
        history,
        forecasts,
        lead_time_days=20,
        lead_time_std_days=3,
        target_service_level=0.90,
    )
    monthly_mean = history["demand_units"].mean()
    monthly_std = history["demand_units"].std(ddof=1)
    daily_mean = monthly_mean / (365.2425 / 12)
    daily_std = monthly_std / math.sqrt(365.2425 / 12)
    expected = NormalDist().inv_cdf(0.90) * math.sqrt(
        20 * daily_std**2 + daily_mean**2 * 3**2
    )

    assert plan["safety_stock_units"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("frequency", "start", "pandas_frequency", "period_days"),
    [
        ("daily", "2024-01-01", "D", 1),
        ("weekly", "2024-01-01", "W-MON", 7),
        ("monthly", "2023-01-01", "MS", 365.2425 / 12),
    ],
)
def test_frequency_conversion_uses_calendar_periods_for_inventory(
    frequency, start, pandas_frequency, period_days
):
    history = pd.DataFrame(
        {
            "period": pd.date_range(start, periods=12, freq=pandas_frequency),
            "demand_quantity": [10 + 2 * index for index in range(12)],
            "training_eligible": True,
        }
    )
    forecast_start = history["period"].iloc[-1] + (
        pd.Timedelta(days=1)
        if frequency == "daily"
        else pd.Timedelta(days=7)
        if frequency == "weekly"
        else pd.offsets.MonthBegin(1)
    )
    forecasts = pd.DataFrame(
        {
            "forecast_period": pd.date_range(
                forecast_start, periods=3, freq=pandas_frequency
            ),
            "predicted_demand": [20.0, 20.0, 20.0],
        }
    )
    plan = calculate_inventory_plan(
        history,
        forecasts,
        current_stock=1,
        lead_time_days=2,
        lead_time_std_days=0,
        ordering_cost=10,
        holding_cost_per_unit_per_year=2,
        frequency=frequency,
    )
    mean = history["demand_quantity"].mean()
    standard_deviation = history["demand_quantity"].std(ddof=1)
    daily_standard_deviation = standard_deviation / math.sqrt(period_days)
    forecast_period_days = (
        1
        if frequency == "daily"
        else 7
        if frequency == "weekly"
        else int(
            (
                forecast_start
                + pd.offsets.MonthBegin(1)
                - forecast_start
            ).days
        )
    )
    expected_safety_stock = NormalDist().inv_cdf(0.95) * math.sqrt(
        2 * daily_standard_deviation**2
    )

    assert plan["average_observed_demand_per_period"] == pytest.approx(mean)
    assert plan["annualized_observed_demand"] == pytest.approx(
        mean * 365.2425 / period_days
    )
    assert plan["lead_time_demand_units"] == pytest.approx(
        20.0 * 2 / forecast_period_days
    )
    assert plan["safety_stock_units"] == pytest.approx(expected_safety_stock)
    expected_stockout = forecast_start + (
        pd.Timedelta(days=1) if frequency == "monthly" else pd.Timedelta(days=0)
    )
    assert pd.Timestamp(plan["stockout_date"]) == pd.Timestamp(expected_stockout)


def test_missing_lead_time_and_cost_inputs_are_explicit():
    history, forecasts = _series([10.0] * 12)

    plan = calculate_inventory_plan(history, forecasts)

    assert plan["lead_time_demand_units"] is None
    assert plan["reorder_point_units"] is None
    assert plan["eoq_units"] is None
    assert plan["stockout_date"] is None
    assert "not supplied" in plan["lead_time_demand_basis"]


def test_zero_demand_has_zero_eoq_and_no_projected_stockout():
    history, forecasts = _series([0.0] * 12, (0.0, 0.0, 0.0))

    plan = calculate_inventory_plan(
        history,
        forecasts,
        current_stock=0,
        lead_time_days=20,
        lead_time_std_days=0,
        ordering_cost=10,
        holding_cost_per_unit_per_year=1,
    )

    assert plan["eoq_units"] == 0
    assert plan["lead_time_demand_units"] == 0
    assert plan["stockout_date"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lead_time_days": 0}, "greater than zero"),
        ({"current_stock": -1}, "non-negative"),
        ({"lead_time_std_days": -1}, "non-negative"),
        ({"ordering_cost": float("inf")}, "finite"),
        ({"target_service_level": 1.0}, "between 0 and 1"),
        (
            {"ordering_cost": 10, "holding_cost_per_unit_per_year": 0},
            "holding cost must be greater than zero",
        ),
    ],
)
def test_invalid_inventory_inputs_are_rejected(kwargs, message):
    history, forecasts = _series([10.0] * 12)
    arguments = {
        "current_stock": None,
        "lead_time_days": None,
        "lead_time_std_days": None,
        "ordering_cost": None,
        "holding_cost_per_unit_per_year": None,
        "target_service_level": 0.95,
    }
    arguments.update(kwargs)

    if "holding_cost_per_unit_per_year" in kwargs:
        plan = calculate_inventory_plan(history, forecasts, **arguments)
        assert message in plan["eoq_status"]
    else:
        with pytest.raises(ValueError, match=message):
            calculate_inventory_plan(history, forecasts, **arguments)


def test_inventory_data_validation_rejects_negative_observed_demand():
    history, forecasts = _series([10.0, -1.0])

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_inventory_plan(history, forecasts)


def test_product_inventory_builder_keeps_missing_user_inputs_unavailable():
    history, forecasts = _series([10.0] * 12)
    history["product_id"] = 1
    forecasts["product_id"] = 1
    inputs = pd.DataFrame(
        {
            "product_id": [1],
            "current_stock": [pd.NA],
            "lead_time_days": [pd.NA],
            "lead_time_std_days": [pd.NA],
        }
    )

    plans = build_product_inventory_plan(
        history,
        forecasts,
        inputs,
        target_service_level=0.95,
        ordering_cost=None,
        holding_cost_per_unit_per_year=None,
    )

    assert plans.loc[0, "lead_time_demand_units"] is None
    assert plans.loc[0, "eoq_units"] is None
