"""
Baseline forecasting methods.

These methods provide simple benchmarks that advanced forecasting
models must outperform.
"""

import pandas as pd


def naive_forecast(series: pd.Series) -> pd.Series:
    """
    Forecast each observation using the previous observation.

    Parameters
    ----------
    series : pd.Series
        Historical demand series ordered chronologically.

    Returns
    -------
    pd.Series
        Naive forecast aligned with the input series.
    """
    if series.empty:
        raise ValueError("Demand series cannot be empty.")

    return series.shift(1)


def moving_average_forecast(
    series: pd.Series,
    window: int = 3,
) -> pd.Series:
    """
    Forecast demand using a rolling moving average.

    Parameters
    ----------
    series : pd.Series
        Historical demand series ordered chronologically.
    window : int, default=3
        Number of previous observations used in the moving average.

    Returns
    -------
    pd.Series
        Moving-average forecast.
    """
    if series.empty:
        raise ValueError("Demand series cannot be empty.")

    if window < 1:
        raise ValueError("Window must be at least 1.")

    return series.shift(1).rolling(window=window).mean()