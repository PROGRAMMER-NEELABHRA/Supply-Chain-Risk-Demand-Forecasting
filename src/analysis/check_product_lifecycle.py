
"""Check a product's presence across the full dataset timeline."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "integrated_supply_chain.csv"

PRODUCT_ID = 191


def main() -> None:
    """Show monthly presence and demand for the selected product."""
    df = pd.read_csv(
        DATA_FILE,
        usecols=["order_date", "product_id", "quantity"],
    )

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce", dayfirst=True
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["order_date", "product_id", "quantity"])

    df["month"] = df["order_date"].dt.to_period("M")

    product = df.loc[df["product_id"] == PRODUCT_ID]

    monthly = (
        product.groupby("month")
        .agg(
            demand_units=("quantity", "sum"),
            order_count=("quantity", "size"),
        )
    )

    full_months = pd.period_range("2017-06", "2018-01", freq="M")
    monthly = monthly.reindex(full_months)
    monthly.index.name = "month"

    print(f"Product: {PRODUCT_ID}")
    print("\nMonthly records (blank means no recorded orders):")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
