
"""Tests for forecasting evaluation metrics."""

import pandas as pd
import pytest

from src.forecasting.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)


def test_mae():
    actual = pd.Series([100, 200, 300])
    predicted = pd.Series([110, 180, 330])

    assert mean_absolute_error(actual, predicted) == pytest.approx(20.0)


def test_rmse():
    actual = pd.Series([100, 200, 300])
    predicted = pd.Series([110, 180, 330])

    expected = ((10**2 + 20**2 + 30**2) / 3) ** 0.5
    assert root_mean_squared_error(actual, predicted) == pytest.approx(expected)


def test_wape():
    actual = pd.Series([100, 200, 300])
    predicted = pd.Series([110, 180, 330])

    expected = (10 + 20 + 30) / 600 * 100
    assert weighted_absolute_percentage_error(actual, predicted) == pytest.approx(
        expected
    )


def test_wape_rejects_zero_total_actual():
    actual = pd.Series([0, 0])
    predicted = pd.Series([1, 2])

    with pytest.raises(ValueError):
        weighted_absolute_percentage_error(actual, predicted)


def test_mae_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mean_absolute_error(pd.Series([1, 2]), pd.Series([1]))
