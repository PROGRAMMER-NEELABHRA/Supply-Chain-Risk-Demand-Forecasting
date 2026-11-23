"""Inventory planning calculations using observed demand and supplied inputs."""

from math import ceil, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd


MIN_VARIABILITY_HISTORY_PERIODS = 12
AVERAGE_DAYS_PER_MONTH = 365.2425 / 12


def _period_days(period: pd.Timestamp, frequency: str) -> int:
    if frequency == "daily":
        return 1
    if frequency == "weekly":
        return 7
    return int((period + pd.offsets.MonthBegin(1) - period).days)


def _next_period(period: pd.Timestamp, frequency: str) -> pd.Timestamp:
    if frequency == "daily":
        return period + pd.Timedelta(days=1)
    if frequency == "weekly":
        return period + pd.Timedelta(days=7)
    return period + pd.offsets.MonthBegin(1)


def _period_frequency(frequency: str) -> str:
    if frequency == "daily":
        return "D"
    if frequency == "weekly":
        return "W-MON"
    if frequency == "monthly":
        return "MS"
    raise ValueError("Frequency must be daily, weekly, or monthly.")


def _canonical_period_frame(
    frame: pd.DataFrame,
    *,
    date_candidates: tuple[str, ...],
    value_candidates: tuple[str, ...],
    label: str,
    eligible_only: bool = False,
) -> pd.DataFrame:
    date_column = next(
        (column for column in date_candidates if column in frame.columns), None
    )
    value_column = next(
        (column for column in value_candidates if column in frame.columns), None
    )
    if date_column is None or value_column is None:
        raise ValueError(
            f"{label} missing a supported date or demand column."
        )
    source = frame
    if eligible_only and "training_eligible" in source.columns:
        source = source.loc[source["training_eligible"].astype(bool)]
    return _validated_series(
        source,
        date_column=date_column,
        value_column=value_column,
        label=label,
    ).rename(columns={date_column: "period", value_column: "demand"})


def _validated_series(
    frame: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    label: str,
) -> pd.DataFrame:
    required = {date_column, value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")
    result = frame[[date_column, value_column]].copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    if result.empty or result.isna().any().any():
        raise ValueError(f"{label} must contain valid, non-empty values.")
    if not np.isfinite(result[value_column]).all():
        raise ValueError(f"{label} values must be finite.")
    if (result[value_column] < 0).any():
        raise ValueError(f"{label} values cannot be negative.")
    if result[date_column].duplicated().any():
        raise ValueError(f"{label} dates must be unique.")
    return result.sort_values(date_column).reset_index(drop=True)


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _forecast_demand_over_days(
    forecasts: pd.DataFrame,
    lead_time_days: float,
    historical_daily_mean: float | None,
    frequency: str,
) -> tuple[float | None, str]:
    remaining = float(lead_time_days)
    total = 0.0
    current = forecasts["period"].min()

    while remaining > 0:
        period_days = _period_days(current, frequency)
        used_days = min(remaining, period_days)
        match = forecasts.loc[forecasts["period"] == current]
        if not match.empty:
            daily_rate = float(match["demand"].iloc[0]) / period_days
        elif current > forecasts["period"].max():
            if historical_daily_mean is None:
                return None, "Lead time extends beyond forecast horizon; no historical demand estimate is available."
            daily_rate = historical_daily_mean
        else:
            return None, f"Forecast period {current:%Y-%m-%d} is missing."
        total += daily_rate * used_days
        remaining -= used_days
        current = _next_period(current, frequency)

    forecast_days = sum(
        _period_days(period, frequency) for period in forecasts["period"]
    )
    if lead_time_days > forecast_days:
        return total, "Beyond-horizon lead-time demand uses observed average daily demand."
    return total, f"Lead-time demand uses the available {frequency} forecast profile."


def calculate_inventory_plan(
    demand_history: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    current_stock: float | None = None,
    lead_time_days: float | None = None,
    target_service_level: float = 0.95,
    lead_time_std_days: float | None = None,
    ordering_cost: float | None = None,
    holding_cost_per_unit_per_year: float | None = None,
    frequency: str = "monthly",
) -> dict[str, object]:
    """Calculate supported product inventory indicators; never infer stock or lead time."""
    if frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("Frequency must be daily, weekly, or monthly.")
    history = _canonical_period_frame(
        demand_history,
        date_candidates=("month", "period"),
        value_candidates=("demand_units", "demand_quantity"),
        label="Demand history",
        eligible_only=True,
    )
    forecast = _canonical_period_frame(
        forecasts,
        date_candidates=("forecast_month", "forecast_period", "period"),
        value_candidates=("forecast_demand", "predicted_demand"),
        label="Forecasts",
    )
    frequency_code = _period_frequency(frequency)
    if frequency == "daily":
        history["period"] = history["period"].dt.normalize()
        forecast["period"] = forecast["period"].dt.normalize()
    elif frequency == "weekly":
        history["period"] = (
            history["period"].dt.to_period("W-SUN").dt.start_time
        )
        forecast["period"] = (
            forecast["period"].dt.to_period("W-SUN").dt.start_time
        )
    else:
        history["period"] = history["period"].dt.to_period("M").dt.to_timestamp()
        forecast["period"] = (
            forecast["period"].dt.to_period("M").dt.to_timestamp()
        )
    if history["period"].duplicated().any() or forecast["period"].duplicated().any():
        raise ValueError(f"{frequency.capitalize()} demand and forecast dates must be unique.")

    stock = _optional_nonnegative(current_stock, "Current stock")
    lead = _optional_nonnegative(lead_time_days, "Lead time")
    lead_sd = _optional_nonnegative(lead_time_std_days, "Lead-time standard deviation")
    order_cost = _optional_nonnegative(ordering_cost, "Ordering cost")
    holding_cost = _optional_nonnegative(
        holding_cost_per_unit_per_year,
        "Annual holding cost per unit",
    )
    if lead == 0:
        raise ValueError("Lead time must be greater than zero.")
    if not 0 < target_service_level < 1:
        raise ValueError("Target service level must be between 0 and 1.")

    observed_periods = len(history)
    average_period_days = (
        1.0 if frequency == "daily"
        else 7.0 if frequency == "weekly"
        else AVERAGE_DAYS_PER_MONTH
    )
    average_period_demand = float(history["demand"].mean())
    forecast_average_period = float(forecast["demand"].mean())
    annualized_demand = (
        average_period_demand * 365.2425 / average_period_days
    )
    historical_daily_mean = average_period_demand / average_period_days
    result: dict[str, object] = {
        "average_observed_demand_per_period": average_period_demand,
        "average_forecast_demand_per_period": forecast_average_period,
        "average_observed_monthly_demand": average_period_demand,
        "average_forecast_monthly_demand": forecast_average_period,
        "annualized_observed_demand": annualized_demand,
        "observed_periods": observed_periods,
        "observed_months": observed_periods,
        "safety_stock_units": None,
        "lead_time_demand_units": None,
        "reorder_point_units": None,
        "reorder_point_includes_safety_stock": False,
        "eoq_units": None,
        "stockout_date": None,
        "stockout_before_replenishment": None,
        "lead_time_demand_basis": "Not calculated: supplier lead time was not supplied.",
        "safety_stock_status": "Not calculated: variability inputs are incomplete.",
        "eoq_status": "Not calculated: ordering and holding costs are required.",
        "depletion_status": "Not calculated: current stock and lead time are required.",
    }

    if order_cost is not None and holding_cost is not None:
        if holding_cost == 0:
            result["eoq_status"] = "Not calculated: annual holding cost must be greater than zero."
        elif average_period_demand == 0:
            result["eoq_units"] = 0.0
            result["eoq_status"] = "Zero observed demand; EOQ is zero under the supplied cost inputs."
        else:
            result["eoq_units"] = sqrt(
                2 * annualized_demand * order_cost / holding_cost
            )
            result["eoq_status"] = "Calculated from annualized observed demand and supplied costs."

    if lead is not None:
        lead_demand, lead_basis = _forecast_demand_over_days(
            forecast,
            lead,
            historical_daily_mean if observed_periods else None,
            frequency,
        )
        result["lead_time_demand_units"] = lead_demand
        result["lead_time_demand_basis"] = lead_basis
        if lead_demand is not None:
            result["reorder_point_units"] = lead_demand

            expected_periods = pd.date_range(
                history["period"].min(),
                history["period"].max(),
                freq=frequency_code,
            )
            contiguous = pd.DatetimeIndex(history["period"]).equals(
                expected_periods
            )
            enough_history = observed_periods >= MIN_VARIABILITY_HISTORY_PERIODS
            if not (contiguous and enough_history):
                result["safety_stock_status"] = (
                    f"Not calculated: at least 12 contiguous observed {frequency} periods are required."
                )
            elif lead_sd is None:
                result["safety_stock_status"] = (
                    "Not calculated: supplier lead-time variability was not supplied."
                )
            else:
                period_sd = float(history["demand"].std(ddof=1))
                daily_sd = period_sd / sqrt(average_period_days)
                z_score = NormalDist().inv_cdf(target_service_level)
                safety_stock = z_score * sqrt(
                    lead * daily_sd**2
                    + historical_daily_mean**2 * lead_sd**2
                )
                result["safety_stock_units"] = max(0.0, safety_stock)
                result["reorder_point_units"] = lead_demand + max(
                    0.0, safety_stock
                )
                result["reorder_point_includes_safety_stock"] = True
                result["safety_stock_status"] = (
                    "Calculated from contiguous observed demand variability, "
                    "supplied lead-time variability, and the selected normal quantile. "
                    "Period-level variance is converted to a daily rate assuming "
                    "independent within-period demand."
                )

        if stock is not None:
            remaining_stock = stock
            for _, row in forecast.iterrows():
                period = row["period"]
                period_days = _period_days(period, frequency)
                daily_rate = float(row["demand"]) / period_days
                if daily_rate > 0 and remaining_stock <= daily_rate * period_days:
                    days_until_stockout = max(
                        0,
                        ceil(remaining_stock / daily_rate) - 1,
                    )
                    stockout = period + pd.Timedelta(days=days_until_stockout)
                    result["stockout_date"] = stockout
                    result["stockout_before_replenishment"] = (
                        days_until_stockout < lead
                    )
                    break
                remaining_stock -= daily_rate * period_days

            if result["stockout_date"] is None:
                result["depletion_status"] = (
                    "No stockout projected within the available forecast horizon."
                )
            else:
                result["depletion_status"] = (
                    "Stockout date is an estimate based on uniform daily demand "
                    f"within each forecast {frequency} period."
                )
        else:
            result["depletion_status"] = (
                "Not calculated: current stock was not supplied."
            )
    elif stock is not None:
        result["depletion_status"] = (
            "Not calculated: supplier lead time was not supplied."
        )

    return result


def build_product_inventory_plan(
    monthly_demand: pd.DataFrame,
    forecasts: pd.DataFrame,
    user_inputs: pd.DataFrame,
    *,
    target_service_level: float,
    ordering_cost: float | None,
    holding_cost_per_unit_per_year: float | None,
    frequency: str = "monthly",
) -> pd.DataFrame:
    """Build per-series plans from demand, forecasts, and explicit user inputs."""
    plans = []
    has_product = "product_id" in forecasts.columns and forecasts[
        "product_id"
    ].notna().any()
    groups = (
        forecasts.groupby("product_id", sort=False, dropna=False)
        if has_product
        else [(None, forecasts)]
    )
    for product_id, forecast in groups:
        if "product_id" not in monthly_demand.columns:
            history = monthly_demand
        elif product_id is None or pd.isna(product_id):
            history = monthly_demand.loc[monthly_demand["product_id"].isna()]
        else:
            history = monthly_demand.loc[
                monthly_demand["product_id"] == product_id
            ]
        if "product_id" not in user_inputs.columns:
            supplied = user_inputs
        elif product_id is None or pd.isna(product_id):
            supplied = user_inputs.loc[user_inputs["product_id"].isna()]
        else:
            supplied = user_inputs.loc[user_inputs["product_id"] == product_id]
        values = supplied.iloc[0] if not supplied.empty else pd.Series(dtype=object)
        plan = calculate_inventory_plan(
            history,
            forecast,
            current_stock=values.get("current_stock"),
            lead_time_days=values.get("lead_time_days"),
            target_service_level=target_service_level,
            lead_time_std_days=values.get("lead_time_std_days"),
            ordering_cost=ordering_cost,
            holding_cost_per_unit_per_year=holding_cost_per_unit_per_year,
            frequency=frequency,
        )
        plans.append({"product_id": product_id, **plan})
    return pd.DataFrame(plans)