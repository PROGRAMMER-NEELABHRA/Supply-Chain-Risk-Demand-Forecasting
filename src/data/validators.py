"""
Validation utilities for forecasting input data.
"""

import pandas as pd


REQUIRED_FORECASTING_COLUMNS = {
    "date",
    "product_id",
    "quantity",
}


def validate_forecasting_columns(df: pd.DataFrame) -> None:
    """
    Validate that the forecasting dataset contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasting dataset.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = REQUIRED_FORECASTING_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required forecasting columns: {missing}"
        )


def validate_forecasting_data(df: pd.DataFrame) -> None:
    """
    Validate basic forecasting data requirements.

    Parameters
    ----------
    df : pd.DataFrame
        Forecasting dataset.

    Raises
    ------
    ValueError
        If required columns contain invalid data.
    """
    validate_forecasting_columns(df)

    if df["date"].isna().any():
        raise ValueError("Forecasting data contains missing dates.")

    if df["product_id"].isna().any():
        raise ValueError("Forecasting data contains missing product IDs.")

    if df["quantity"].isna().any():
        raise ValueError("Forecasting data contains missing quantities.")

    if not pd.api.types.is_numeric_dtype(df["quantity"]):
        raise ValueError("Quantity must be numeric.")