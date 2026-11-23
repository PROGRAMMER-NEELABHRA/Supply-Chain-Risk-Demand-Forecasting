
"""Tests for forecasting evaluation metrics."""

import pandas as pd
import pytest

from src.forecasting import run_random_forest_experiment as experiment
from src.forecasting.evaluation import (
    calculate_product_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    select_products_with_complete_history,
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


def test_select_products_with_complete_history():
    months = pd.date_range("2020-01-01", periods=3, freq="MS")
    demand = pd.DataFrame(
        {
            "product_id": [1, 1, 1, 2, 2],
            "month": [*months, months[0], months[2]],
            "demand_units": [10, 20, 30, 5, 15],
        }
    )

    selected = select_products_with_complete_history(
        demand,
        months[0],
        months[-1],
    )

    assert selected["product_id"].unique().tolist() == [1]


def test_calculate_product_metrics():
    results = pd.DataFrame(
        {
            "product_id": [1, 1, 2, 2],
            "actual": [10, 20, 30, 40],
            "naive_forecast": [10, 10, 20, 40],
            "moving_forecast": [5, 15, 25, 35],
        }
    )

    metrics = calculate_product_metrics(
        results,
        {
            "Naive": "naive_forecast",
            "Moving Average": "moving_forecast",
        },
    )

    assert metrics[["product_id", "model"]].values.tolist() == [
        [1, "Naive"],
        [1, "Moving Average"],
        [2, "Naive"],
        [2, "Moving Average"],
    ]
    assert metrics.loc[metrics["model"] == "Naive", "mae"].tolist() == [
        5.0,
        5.0,
    ]


def test_recursive_backtest_does_not_feed_test_actuals_into_predictions(
    monkeypatch,
):
    class IncrementingModel:
        def predict(self, features):
            return [float(features["lag_1"].iloc[0]) + 1]

    monkeypatch.setattr(
        experiment,
        "train_random_forest",
        lambda features, target: IncrementingModel(),
    )
    months = pd.date_range(
        experiment.START_MONTH,
        experiment.TEST_END,
        freq="MS",
    )
    product = pd.DataFrame(
        {
            "product_id": 1,
            "month": months,
            "demand_units": list(range(1, 23)) + [1000, 2000, 3000],
        }
    )

    results = experiment.forecast_one_product(product)

    assert [row["random_forest_forecast"] for row in results] == [
        23.0,
        24.0,
        25.0,
    ]
    assert [row["actual"] for row in results] == [1000, 2000, 3000]
