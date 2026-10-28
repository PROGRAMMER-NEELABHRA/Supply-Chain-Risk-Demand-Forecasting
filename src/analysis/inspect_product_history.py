
"""Inspect the monthly demand history of one product."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)

PRODUCT_ID = 365


def main() -> None:
    """Print monthly demand and identify missing months."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["month"])

    product = (
        df.loc[df["product_id"] == PRODUCT_ID]
        .sort_values("month")
        .copy()
    )

    if product.empty:
        raise ValueError(f"No records found for product {PRODUCT_ID}.")

    expected_months = pd.date_range(
        product["month"].min(),
        product["month"].max(),
        freq="MS",
    )

    observed_months = pd.DatetimeIndex(product["month"])
    missing_months = expected_months.difference(observed_months)

    print(f"Product ID: {PRODUCT_ID}")
    print(f"Observed months: {len(product)}")
    print(f"First month: {product['month'].min():%Y-%m}")
    print(f"Last month: {product['month'].max():%Y-%m}")
    print(f"Missing months within history: {len(missing_months)}")

    if len(missing_months):
        print("Missing months:", ", ".join(m.strftime("%Y-%m") for m in missing_months))

    print("\nMonthly demand:")
    print(product[["month", "demand_units"]].to_string(index=False))

    print("\nDemand summary:")
    print(product["demand_units"].describe().to_string())


if __name__ == "__main__":
    main()
