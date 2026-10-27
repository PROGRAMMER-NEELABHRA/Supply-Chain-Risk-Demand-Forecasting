
"""Prepare a stable product-month dataset for baseline forecasting."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

TIMESERIES_PATH = DATA_DIR / "demand_timeseries.csv"
OUTPUT_PATH = DATA_DIR / "forecasting_dataset.csv"

CUTOFF_MONTH = "2017-09"
MIN_ACTIVE_MONTHS = 24


def prepare_forecasting_data(data: pd.DataFrame) -> pd.DataFrame:
    """Keep pre-shift months and products with sufficient active history."""
    required_columns = {"month", "product_id", "demand_units"}

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = data.copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data["demand_units"] = pd.to_numeric(
        data["demand_units"], errors="raise"
    )

    cutoff = pd.Period(CUTOFF_MONTH, freq="M")
    data = data.loc[data["month"] <= cutoff].copy()

    active_months = (
        data.loc[data["demand_units"] > 0]
        .groupby("product_id")["month"]
        .nunique()
    )
    eligible_products = active_months[
        active_months >= MIN_ACTIVE_MONTHS
    ].index

    result = data.loc[data["product_id"].isin(eligible_products)].copy()
    result = result.sort_values(["product_id", "month"])

    if result.empty:
        raise ValueError(
            "No products meet the minimum active-month requirement."
        )

    return result


def main() -> None:
    """Load, prepare, validate, and save the forecasting dataset."""
    if not TIMESERIES_PATH.is_file():
        raise FileNotFoundError(
            f"Demand time series not found: {TIMESERIES_PATH}"
        )

    data = pd.read_csv(TIMESERIES_PATH)
    result = prepare_forecasting_data(data)

    result.to_csv(OUTPUT_PATH, index=False)

    print("=== FORECASTING DATASET ===")
    print(f"Rows: {len(result):,}")
    print(f"Products: {result['product_id'].nunique():,}")
    print(f"Months: {result['month'].nunique():,}")
    print(f"Total recorded demand units: {result['demand_units'].sum():,}")
    print(f"Date range: {result['month'].min()} to {result['month'].max()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
