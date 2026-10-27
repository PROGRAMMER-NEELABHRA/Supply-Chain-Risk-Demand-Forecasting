
"""Validate the complete product-month demand time series."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

MONTHLY_PATH = DATA_DIR / "monthly_demand.csv"
TIMESERIES_PATH = DATA_DIR / "demand_timeseries.csv"


def main() -> None:
    """Check completeness, uniqueness, and demand preservation."""
    monthly = pd.read_csv(MONTHLY_PATH)
    timeseries = pd.read_csv(TIMESERIES_PATH)

    required = {"month", "product_id", "demand_units"}
    if not required.issubset(timeseries.columns):
        raise ValueError("Time series is missing required columns.")

    if timeseries[["month", "product_id"]].isna().any().any():
        raise ValueError("Missing month or product ID found.")

    if timeseries[["month", "product_id"]].duplicated().any():
        raise ValueError("Duplicate month-product combinations found.")

    timeseries["demand_units"] = pd.to_numeric(
        timeseries["demand_units"], errors="raise"
    )

    if (timeseries["demand_units"] < 0).any():
        raise ValueError("Negative demand found.")

    product_count = monthly["product_id"].nunique()
    month_count = monthly["month"].nunique()
    expected_rows = product_count * month_count

    if len(timeseries) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows; found {len(timeseries)}."
        )

    if timeseries["demand_units"].sum() != monthly["demand_units"].sum():
        raise ValueError("Demand totals changed during zero-filling.")

    zero_rows = int((timeseries["demand_units"] == 0).sum())

    print("=== TIME SERIES VALIDATION ===")
    print(f"Expected rows: {expected_rows:,}")
    print(f"Actual rows: {len(timeseries):,}")
    print(f"Unique products: {timeseries['product_id'].nunique():,}")
    print(f"Unique months: {timeseries['month'].nunique():,}")
    print(f"Zero-demand rows: {zero_rows:,}")
    print(f"Zero-demand percentage: {zero_rows / len(timeseries) * 100:.2f}%")
    print(f"Monthly demand total: {timeseries['demand_units'].sum():,.0f}")
    print("\nPASS: Time series validation succeeded.")


if __name__ == "__main__":
    main()
