
"""Validation utilities for supply-chain datasets."""

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = {
    "order_date",
    "product_id",
    "quantity",
}

DEMAND_REQUIRED_COLUMNS = {
    "month",
    "product_id",
    "demand_units",
}

REQUIRED_FORECASTING_COLUMNS = {
    "date",
    "product_id",
    "quantity",
}


def parse_raw_order_dates(values: pd.Series) -> pd.Series:
    """Parse ISO dates and source-style day-first order dates safely."""
    text = values.astype("string")
    iso_dates = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    if iso_dates.any():
        parsed.loc[iso_dates] = pd.to_datetime(
            text.loc[iso_dates],
            format="ISO8601",
            errors="coerce",
        )
    if (~iso_dates).any():
        parsed.loc[~iso_dates] = pd.to_datetime(
            text.loc[~iso_dates],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

    return parsed


def validate_forecasting_columns(df: pd.DataFrame) -> None:
    """Validate the legacy forecasting dataset schema."""
    missing = REQUIRED_FORECASTING_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required forecasting columns: {sorted(missing)}"
        )


def validate_forecasting_data(df: pd.DataFrame) -> None:
    """Validate the legacy date, product ID and quantity schema."""
    validate_forecasting_columns(df)

    if df.empty:
        raise ValueError("Forecasting data cannot be empty.")

    if df[list(REQUIRED_FORECASTING_COLUMNS)].isna().any().any():
        raise ValueError("Forecasting data contains missing values.")

    if pd.to_datetime(df["date"], errors="coerce").isna().any():
        raise ValueError("Forecasting data contains invalid dates.")

    quantity = pd.to_numeric(df["quantity"], errors="coerce")

    if quantity.isna().any():
        raise ValueError("Quantity must be numeric.")

    if not np.isfinite(quantity).all():
        raise ValueError("Quantity contains non-finite values.")


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate raw order-level supply-chain data."""
    missing = RAW_REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing raw data columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Raw dataset cannot be empty.")

    if df.duplicated().any():
        raise ValueError("Raw data contains duplicate records.")

    if df[list(RAW_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Raw data contains missing required values.")

    if "order_item_id" in df.columns:
        if df["order_item_id"].isna().any():
            raise ValueError("Raw data contains missing order item IDs.")

        if df["order_item_id"].duplicated().any():
            raise ValueError("Raw data contains duplicate order item IDs.")

    dates = parse_raw_order_dates(df["order_date"])
    quantities = pd.to_numeric(df["quantity"], errors="coerce")
    product_ids = pd.to_numeric(df["product_id"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Raw data contains invalid order dates.")

    if not np.isfinite(quantities).all():
        raise ValueError("Raw quantities contain invalid values.")

    if not np.isfinite(product_ids).all():
        raise ValueError("Raw product IDs must be numeric.")

    if (quantities <= 0).any():
        raise ValueError("Raw transaction quantities must be positive.")


def validate_monthly_demand(df: pd.DataFrame) -> None:
    """Validate aggregated monthly product demand."""
    missing = DEMAND_REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing monthly demand columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Monthly demand dataset cannot be empty.")

    if df[list(DEMAND_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Monthly demand contains missing values.")

    months = pd.to_datetime(df["month"], errors="coerce")
    demand = pd.to_numeric(df["demand_units"], errors="coerce")
    product_ids = pd.to_numeric(df["product_id"], errors="coerce")

    if months.isna().any():
        raise ValueError("Monthly demand contains invalid dates.")

    if not np.isfinite(demand).all():
        raise ValueError("Demand contains non-finite values.")

    if not np.isfinite(product_ids).all():
        raise ValueError("Product IDs contain invalid values.")

    if (demand < 0).any():
        raise ValueError("Demand cannot be negative.")

    if (months.dt.day != 1).any():
        raise ValueError("Monthly dates must represent month starts.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")


def validate_forecast_output(df: pd.DataFrame) -> None:
    """Validate generated product-month forecast rows."""
    required = {
        "product_id",
        "forecast_month",
        "forecast_demand",
        "model_name",
        "forecast_method",
        "forecast_confidence",
        "data_quality_status",
        "first_recorded_month",
        "last_recorded_month",
        "observed_history_months",
        "months_since_last_record",
        "internal_missing_months",
        "forecast_origin_month",
        "forecast_horizon_month",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required forecast columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Forecast output cannot be empty.")

    if df[list(required)].isna().any().any():
        raise ValueError("Forecast output contains missing required values.")

    date_columns = [
        "forecast_month",
        "first_recorded_month",
        "last_recorded_month",
        "forecast_origin_month",
    ]
    parsed_dates = {
        column: pd.to_datetime(df[column], errors="coerce")
        for column in date_columns
    }
    forecast_months = parsed_dates["forecast_month"]
    demand = pd.to_numeric(df["forecast_demand"], errors="coerce")
    horizons = pd.to_numeric(
        df["forecast_horizon_month"], errors="coerce"
    )
    product_ids = pd.to_numeric(df["product_id"], errors="coerce")
    numeric_metadata = {
        column: pd.to_numeric(df[column], errors="coerce")
        for column in (
            "observed_history_months",
            "months_since_last_record",
            "internal_missing_months",
        )
    }

    if any(values.isna().any() for values in parsed_dates.values()):
        raise ValueError("Forecast output contains invalid dates.")

    if any((values.dt.day != 1).any() for values in parsed_dates.values()):
        raise ValueError("Forecast dates must represent month starts.")

    if not np.isfinite(demand).all():
        raise ValueError("Forecast demand contains non-finite values.")

    if (demand < 0).any():
        raise ValueError("Forecast demand cannot be negative.")

    if not np.isfinite(horizons).all() or (horizons < 1).any():
        raise ValueError("Forecast horizons must be positive finite values.")

    if not np.isfinite(product_ids).all():
        raise ValueError("Forecast product IDs must be numeric.")

    for column, values in numeric_metadata.items():
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"Forecast metadata {column} must be finite and nonnegative."
            )
    if (numeric_metadata["observed_history_months"] < 1).any():
        raise ValueError("Forecast products must have observed history.")

    valid_confidence = {"HIGH", "MEDIUM", "LOW"}
    if not set(df["forecast_confidence"]).issubset(valid_confidence):
        raise ValueError("Forecast confidence contains an unknown value.")

    if df.duplicated(["product_id", "forecast_month"]).any():
        raise ValueError("Duplicate product-month forecasts found.")

    for product_id, group in df.groupby("product_id"):
        product_horizons = sorted(
            pd.to_numeric(group["forecast_horizon_month"]).astype(int)
        )
        if product_horizons != [1, 2, 3]:
            raise ValueError(
                f"Product {product_id} must have forecast horizons 1, 2, and 3."
            )
