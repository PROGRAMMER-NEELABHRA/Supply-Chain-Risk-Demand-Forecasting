
"""Create a continuous monthly demand time series for each product."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

MONTHLY_DEMAND_PATH = DATA_DIR / "monthly_demand.csv"
OUTPUT_PATH = DATA_DIR / "demand_timeseries.csv"


def build_demand_timeseries(monthly_data: pd.DataFrame) -> pd.DataFrame:
    """Fill absent product-month combinations with zero recorded demand."""
    required = {"month", "product_id", "demand_units"}
    missing = required - set(monthly_data.columns)

    if missing:
        raise ValueError(
            f"Monthly demand data is missing columns: {sorted(missing)}"
        )

    data = monthly_data.copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M")

    if data[["month", "product_id"]].duplicated().any():
        raise ValueError("Duplicate month-product combinations found.")

    first_month = data["month"].min()
    last_month = data["month"].max()

    all_months = pd.period_range(
        start=first_month,
        end=last_month,
        freq="M",
    )
    all_products = sorted(data["product_id"].unique())

    complete_index = pd.MultiIndex.from_product(
        [all_months, all_products],
        names=["month", "product_id"],
    )

    result = (
        data.set_index(["month", "product_id"])[["demand_units"]]
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    result["month"] = result["month"].astype(str)
    result = result.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)

    return result


def main() -> None:
    """Build and save the complete product-month time series."""
    if not MONTHLY_DEMAND_PATH.is_file():
        raise FileNotFoundError(
            f"Monthly demand file not found: {MONTHLY_DEMAND_PATH}"
        )

    monthly_data = pd.read_csv(MONTHLY_DEMAND_PATH)
    timeseries = build_demand_timeseries(monthly_data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(OUTPUT_PATH, index=False)

    print("=== DEMAND TIME SERIES CREATED ===")
    print(f"Products: {timeseries['product_id'].nunique():,}")
    print(f"Months: {timeseries['month'].nunique():,}")
    print(f"Expected product-month rows: {len(timeseries):,}")
    print(f"Total demand units: {timeseries['demand_units'].sum():,.0f}")
    print(
        "Zero-demand product-months: "
        f"{(timeseries['demand_units'] == 0).sum():,}"
    )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
