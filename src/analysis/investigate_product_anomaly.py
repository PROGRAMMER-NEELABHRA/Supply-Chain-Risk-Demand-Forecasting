
"""Investigate an unusual monthly demand value in the raw data."""

import pandas as pd

from config import DATACO_RAW_DATA_FILE


DATA_FILE = DATACO_RAW_DATA_FILE

PRODUCT_ID = 365


def latest_observed_product_month(months: pd.Series) -> pd.Period:
    """Return the latest month with an actual observation for a product."""
    observed = months.dropna()
    if observed.empty:
        raise ValueError("No valid product observations are available.")
    return observed.max()


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
    if product.empty:
        raise ValueError(
            f"No valid observations are available for product {PRODUCT_ID}."
        )
    latest_observed_month = latest_observed_product_month(product["month"])

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
        product["month"] == latest_observed_month
    ]

    print(f"\n=== RAW RECORDS FOR {latest_observed_month} ===")
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
