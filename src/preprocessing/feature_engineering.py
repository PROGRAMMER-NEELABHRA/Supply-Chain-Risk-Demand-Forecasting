
"""Create leakage-safe features for monthly demand forecasting."""

import pandas as pd


REQUIRED_COLUMNS = {"month", "product_id", "demand_units"}


def create_demand_features(
    data: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """Create lag, rolling-average, and calendar features per product.

    Rows are sorted by product and month. Lag and rolling features use
    only earlier observations, never the current month's demand.

    Parameters
    ----------
    data:
        DataFrame with month, product_id, and demand_units columns.
    lags:
        Previous monthly observations to include as features.

    Returns
    -------
    pd.DataFrame
        Feature rows with the original target and generated features.
        Rows without sufficient lag history are excluded.
    """
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not lags or any(lag < 1 for lag in lags):
        raise ValueError("Lags must be positive integers.")

    df = data[list(REQUIRED_COLUMNS)].copy()
    df["month"] = pd.to_datetime(df["month"], errors="raise")
    df["demand_units"] = pd.to_numeric(
        df["demand_units"], errors="raise"
    )

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Input contains missing required values.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    df = df.sort_values(["product_id", "month"]).reset_index(drop=True)

    grouped = df.groupby("product_id", sort=False)["demand_units"]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Shift first so the rolling average excludes the current month's target.
    df["rolling_mean_3"] = df.groupby("product_id")["demand_units"].transform(
        lambda values: values.shift(1).rolling(window=3).mean()
    )
    df["rolling_mean_6"] = df.groupby("product_id")["demand_units"].transform(
        lambda values: values.shift(1).rolling(window=6).mean()
    )

    df["month_number"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter

    feature_columns = [
        *(f"lag_{lag}" for lag in lags),
        "rolling_mean_3",
        "rolling_mean_6",
        "month_number",
        "quarter",
    ]

    result = df.dropna(subset=feature_columns).copy()

    if result.empty:
        raise ValueError(
            "No feature rows remain. Check the history length and lag settings."
        )

    return result
