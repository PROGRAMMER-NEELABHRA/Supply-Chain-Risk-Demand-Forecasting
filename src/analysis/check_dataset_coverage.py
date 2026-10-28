
"""Check whether the source dataset ends partway through a month."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "integrated_supply_chain.csv"


def main() -> None:
    """Report overall and monthly date coverage."""
    df = pd.read_csv(DATA_FILE, usecols=["order_date", "product_id", "quantity"])

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce", dayfirst=True
    )
    df = df.dropna(subset=["order_date"])

    print(f"Earliest order: {df['order_date'].min()}")
    print(f"Latest order: {df['order_date'].max()}")

    df["month"] = df["order_date"].dt.to_period("M")

    monthly = (
        df.groupby("month")
        .agg(
            order_count=("quantity", "size"),
            product_count=("product_id", "nunique"),
            active_days=("order_date", lambda dates: dates.dt.day.nunique()),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        )
        .reset_index()
    )

    print("\nLast six months in the source dataset:")
    print(monthly.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
