"""Deterministic demand scenarios based on the saved forecast horizon."""

import numpy as np
import pandas as pd


REQUIRED_FORECAST_COLUMNS = {
    "product_id",
    "forecast_month",
    "forecast_demand",
}


def build_demand_scenarios(
    forecasts: pd.DataFrame,
    *,
    product_id: int | str | None = None,
    optimistic_multiplier: float = 1.1,
    pessimistic_multiplier: float = 0.9,
    user_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Return one row per product-month with deterministic scenario deltas."""
    missing = REQUIRED_FORECAST_COLUMNS - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecasts missing columns: {sorted(missing)}")
    if forecasts.empty:
        raise ValueError("Forecasts cannot be empty.")
    data = forecasts.copy()
    data["forecast_month"] = pd.to_datetime(
        data["forecast_month"], errors="coerce"
    )
    data["forecast_demand"] = pd.to_numeric(
        data["forecast_demand"], errors="coerce"
    )
    if data[["forecast_month", "forecast_demand"]].isna().any().any():
        raise ValueError("Forecast rows must contain valid dates and demand values.")
    has_product_ids = data["product_id"].notna().any()
    if has_product_ids and data["product_id"].isna().any():
        raise ValueError("Forecast product IDs must be consistently present or absent.")
    if not np.isfinite(data["forecast_demand"]).all() or (
        data["forecast_demand"] < 0
    ).any():
        raise ValueError("Forecast demand must be finite and non-negative.")
    duplicate_columns = (
        ["product_id", "forecast_month"]
        if has_product_ids
        else ["forecast_month"]
    )
    if data.duplicated(duplicate_columns).any():
        raise ValueError("Forecasts contain duplicate product-period rows.")

    for value, name in [
        (optimistic_multiplier, "Optimistic multiplier"),
        (pessimistic_multiplier, "Pessimistic multiplier"),
        (user_multiplier, "User multiplier"),
    ]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero.")
    if optimistic_multiplier < 1:
        raise ValueError("Optimistic multiplier must be at least 1.")
    if pessimistic_multiplier > 1:
        raise ValueError("Pessimistic multiplier must be at most 1.")

    if product_id is not None:
        data = data.loc[data["product_id"] == product_id].copy()
        if data.empty:
            raise ValueError(f"No forecast rows found for product {product_id}.")
    data = data.sort_values(["product_id", "forecast_month"]).reset_index(drop=True)
    data["baseline_demand"] = data["forecast_demand"]
    data["optimistic_demand"] = (
        data["baseline_demand"] * optimistic_multiplier
    )
    data["pessimistic_demand"] = (
        data["baseline_demand"] * pessimistic_multiplier
    )
    data["user_scenario_demand"] = data["baseline_demand"] * user_multiplier
    for scenario in ("optimistic", "pessimistic", "user_scenario"):
        data[f"{scenario}_delta_units"] = (
            data[f"{scenario}_demand"] - data["baseline_demand"]
        )
        data[f"{scenario}_delta_pct"] = np.where(
            data["baseline_demand"] != 0,
            data[f"{scenario}_delta_units"] / data["baseline_demand"] * 100,
            np.nan,
        )
    return data


def aggregate_portfolio_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Sum product scenarios by forecast month and recompute portfolio deltas."""
    required = {
        "forecast_month",
        "baseline_demand",
        "optimistic_demand",
        "pessimistic_demand",
        "user_scenario_demand",
    }
    missing = required - set(scenarios.columns)
    if missing:
        raise ValueError(f"Scenario table missing columns: {sorted(missing)}")
    result = (
        scenarios.groupby("forecast_month", as_index=False)[
            [
                "baseline_demand",
                "optimistic_demand",
                "pessimistic_demand",
                "user_scenario_demand",
            ]
        ]
        .sum()
        .sort_values("forecast_month")
        .reset_index(drop=True)
    )
    for scenario in ("optimistic", "pessimistic", "user_scenario"):
        result[f"{scenario}_delta_units"] = (
            result[f"{scenario}_demand"] - result["baseline_demand"]
        )
        result[f"{scenario}_delta_pct"] = np.where(
            result["baseline_demand"] != 0,
            result[f"{scenario}_delta_units"]
            / result["baseline_demand"]
            * 100,
            np.nan,
        )
    return result