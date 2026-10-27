
"""Investigate monthly patterns in the original DataCo source data."""

import pandas as pd

from src.data.loaders import load_dataco_data


def main():
    """Compare order and quantity patterns across months."""

    raw = load_dataco_data()

    raw["order date (DateOrders)"] = pd.to_datetime(
        raw["order date (DateOrders)"],
        format="mixed",
        errors="coerce",
    )

    if raw["order date (DateOrders)"].isna().any():
        raise ValueError("Invalid order dates found in the raw dataset.")

    raw["month"] = (
        raw["order date (DateOrders)"]
        .dt.to_period("M")
        .astype(str)
    )

    raw["Order Item Quantity"] = pd.to_numeric(
        raw["Order Item Quantity"],
        errors="coerce",
    )

    # Inspect monthly patterns directly in the original source.
    monthly = raw.groupby("month").agg(
        rows=("Order Item Id", "size"),
        unique_orders=("Order Id", "nunique"),
        total_quantity=("Order Item Quantity", "sum"),
        quantity_one_rows=(
            "Order Item Quantity",
            lambda values: (values == 1).sum(),
        ),
        distinct_products=("Product Card Id", "nunique"),
        distinct_unit_prices=("Order Item Product Price", "nunique"),
    )

    monthly["rows_per_order"] = (
        monthly["rows"] / monthly["unique_orders"]
    ).round(2)

    monthly["quantity_one_pct"] = (
        monthly["quantity_one_rows"] / monthly["rows"] * 100
    ).round(2)

    print("=== RAW DATA MONTHLY PATTERNS: JUNE 2017 ONWARD ===")
    print(monthly.loc["2017-06":].to_string())

    # Check how many order items each order contains.
    raw["order_month"] = raw["month"]

    items_per_order = (
        raw.groupby(["order_month", "Order Id"])
        .size()
        .rename("items_in_order")
        .reset_index()
    )

    order_summary = items_per_order.groupby("order_month").agg(
        orders=("Order Id", "size"),
        single_item_orders=(
            "items_in_order",
            lambda values: (values == 1).sum(),
        ),
        maximum_items_in_order=("items_in_order", "max"),
    )

    order_summary["single_item_order_pct"] = (
        order_summary["single_item_orders"]
        / order_summary["orders"]
        * 100
    ).round(2)

    print("\n=== ORDER STRUCTURE: JUNE 2017 ONWARD ===")
    print(order_summary.loc["2017-06":].to_string())


if __name__ == "__main__":
    main()
