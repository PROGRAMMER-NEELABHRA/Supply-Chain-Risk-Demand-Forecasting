
"""Investigate monthly sales and demand patterns for data-quality issues."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SALES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "dataco" / "sales.csv"
)


def main():
    """Compare monthly demand metrics and inspect daily coverage."""

    if not SALES_PATH.is_file():
        raise FileNotFoundError(f"Sales dataset not found: {SALES_PATH}")

    sales = pd.read_csv(SALES_PATH)

    sales["order_date"] = pd.to_datetime(
        sales["order_date"],
        errors="coerce",
    )

    if sales["order_date"].isna().any():
        raise ValueError("Invalid order dates found in sales.csv.")

    sales["month"] = sales["order_date"].dt.to_period("M").astype(str)

    print("=== MONTHLY DATA QUALITY CHECK ===")

    monthly = sales.groupby("month").agg(
        rows=("order_item_id", "size"),
        distinct_orders=("order_id", "nunique"),
        units_sold=("quantity", "sum"),
        average_quantity_per_item=("quantity", "mean"),
        sales_total=("sales", "sum"),
        order_item_total=("order_item_total", "sum"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
    )

    monthly["average_quantity_per_item"] = (
        monthly["average_quantity_per_item"].round(2)
    )
    monthly["sales_total"] = monthly["sales_total"].round(2)
    monthly["order_item_total"] = monthly["order_item_total"].round(2)

    print(monthly.tail(12).to_string())

    print("\n=== DAILY COVERAGE: SEPTEMBER 2017 TO JANUARY 2018 ===")

    start_date = pd.Timestamp("2017-09-01")
    end_date = pd.Timestamp("2018-02-01")

    recent = sales.loc[
        sales["order_date"].ge(start_date)
        & sales["order_date"].lt(end_date)
    ].copy()

    daily = recent.groupby(
        recent["order_date"].dt.strftime("%Y-%m-%d")
    ).agg(
        rows=("order_item_id", "size"),
        units_sold=("quantity", "sum"),
        sales_total=("sales", "sum"),
    )

    daily["sales_total"] = daily["sales_total"].round(2)

    print(daily.to_string())

    print("\n=== OVERALL DATE COVERAGE ===")
    print(f"Earliest order date: {sales['order_date'].min()}")
    print(f"Latest order date:   {sales['order_date'].max()}")
    print(f"Total order-item rows: {len(sales):,}")


if __name__ == "__main__":
    main()
