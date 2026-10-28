
"""Evaluation metrics for demand forecasting models."""

import numpy as np
import pandas as pd


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
