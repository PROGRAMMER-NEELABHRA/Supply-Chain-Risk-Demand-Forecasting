import pandas as pd
import pytest

from src.forecasting.baseline import (
    moving_average_forecast,
    naive_forecast,
)


def test_naive_forecast():
    series = pd.Series([10, 20, 30])

    result = naive_forecast(series)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 10
    assert result.iloc[2] == 20


def test_moving_average_forecast():
    series = pd.Series([10, 20, 30, 40])

    result = moving_average_forecast(series, window=2)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 15
    assert result.iloc[3] == 25


def test_invalid_window():
    series = pd.Series([10, 20, 30])

    with pytest.raises(ValueError):
        moving_average_forecast(series, window=0)