
"""Investigate an unusual monthly demand value in the raw data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "integrated_supply_chain.csv"

PRODUCT_ID = 365
ANOMALY_MONTH = "2017-10"


def main() -> None:
    """Compare monthly transactions and inspect the anomalous month."""
    df = pd.read_csv(DATA_FILE)

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce", dayfirst=True
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    product = df.loc[df["product_id"] == PRODUCT_ID].copy()
    product = product.dropna(subset=["order_date", "quantity"])
    product["month"] = product["order_date"].dt.to_period("M")

    monthly = (
        product.groupby("month")
        .agg(
            demand_units=("quantity", "sum"),
            order_item_count=("order_item_id", "nunique"),
            active_days=("order_date", lambda dates: dates.dt.date.nunique()),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
        )
        .reset_index()
    )

    print("\n=== MONTHLY TRANSACTION SUMMARY ===")
    print(monthly.tail(8).to_string(index=False))

    anomaly = product.loc[
        product["month"] == pd.Period(ANOMALY_MONTH, freq="M")
    ]

    print(f"\n=== RAW RECORDS FOR {ANOMALY_MONTH} ===")
    print(f"Order-item rows: {len(anomaly)}")
    print(f"Total quantity: {anomaly['quantity'].sum():,.0f}")

    columns = [
        column
        for column in [
            "order_date",
            "order_item_id",
            "order_id",
            "product_id",
            "quantity",
            "product_status",
        ]
        if column in anomaly.columns
    ]

    print(anomaly[columns].sort_values("order_date").to_string(index=False))


if __name__ == "__main__":
    main()
