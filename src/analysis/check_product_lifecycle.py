
"""Check a product's presence across the full dataset timeline."""

import pandas as pd

from config import DATACO_RAW_DATA_FILE


DATA_FILE = DATACO_RAW_DATA_FILE

PRODUCT_ID = 191


def observed_month_range(order_dates: pd.Series) -> pd.PeriodIndex:
    """Return calendar months bounded by valid dates in the source."""
    dates = pd.to_datetime(order_dates, errors="coerce").dropna()
    if dates.empty:
        raise ValueError("No valid order dates are available.")

    return pd.period_range(
        dates.min().to_period("M"),
        dates.max().to_period("M"),
        freq="M",
    )


def main() -> None:
    """Show monthly presence and demand for the selected product."""
    df = pd.read_csv(
        DATA_FILE,
        usecols=["order_date", "product_id", "quantity"],
    )

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce", format="mixed", dayfirst=True
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

    full_months = observed_month_range(df["order_date"])
    monthly = monthly.reindex(full_months)
    monthly.index.name = "month"

    print(f"Product: {PRODUCT_ID}")
    print("\nMonthly records (blank means no recorded orders):")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
