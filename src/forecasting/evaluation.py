
"""Evaluation metrics for demand forecasting models."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def select_products_with_complete_history(
    dataframe: pd.DataFrame,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
) -> pd.DataFrame:
    """Select products observed in every month of an inclusive date range."""
    expected_months = pd.date_range(start_month, end_month, freq="MS")
    coverage = dataframe.groupby("product_id")["month"].agg(
        month_count="nunique",
        first_month="min",
        last_month="max",
    )
    eligible_ids = coverage.index[
        (coverage["month_count"] == len(expected_months))
        & (coverage["first_month"] == expected_months[0])
        & (coverage["last_month"] == expected_months[-1])
    ]

    return dataframe.loc[
        dataframe["product_id"].isin(eligible_ids)
    ].copy()


def calculate_product_metrics(
    results: pd.DataFrame,
    model_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Calculate forecast metrics for each product and named model."""
    metric_rows = []

    for product_id, group in results.groupby("product_id"):
        actual = group["actual"]

        for model_name, column in model_columns.items():
            predicted = group[column]
            metric_rows.append(
                {
                    "product_id": product_id,
                    "model": model_name,
                    "mae": mean_absolute_error(actual, predicted),
                    "rmse": root_mean_squared_error(actual, predicted),
                    "wape_pct": weighted_absolute_percentage_error(
                        actual, predicted
                    ),
                }
            )

    return pd.DataFrame(metric_rows)


def mean_absolute_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate the average absolute prediction error."""
    actual, predicted = _validate_inputs(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate RMSE, penalizing larger errors more heavily."""
    actual, predicted = _validate_inputs(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def weighted_absolute_percentage_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate WAPE as a percentage, when total actual demand is positive."""
    actual, predicted = _validate_inputs(actual, predicted)
    denominator = np.abs(actual).sum()

    if denominator == 0:
        raise ValueError("WAPE is undefined when total actual demand is zero.")

    return float(np.abs(actual - predicted).sum() / denominator * 100)


def _validate_inputs(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert actual and predicted values."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")

    if len(actual_values) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted values must have equal lengths.")

    if not np.isfinite(actual_values).all():
        raise ValueError("Actual values contain missing or non-finite values.")

    if not np.isfinite(predicted_values).all():
        raise ValueError("Predicted values contain missing or non-finite values.")

    return actual_values, predicted_values
