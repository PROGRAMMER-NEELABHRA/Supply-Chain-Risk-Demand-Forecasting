
"""Inspect product timelines and prices in the raw DataCo dataset."""

import pandas as pd

from src.data.loaders import load_dataco_data


def main():
    """Summarize first/last order dates and prices for affected products."""

    data = load_dataco_data()

    data["order date (DateOrders)"] = pd.to_datetime(
        data["order date (DateOrders)"],
        format="mixed",
        errors="coerce",
    )

    data["Order Item Product Price"] = pd.to_numeric(
        data["Order Item Product Price"],
        errors="coerce",
    )

    product_summary = (
        data.groupby(["Product Card Id", "Product Name"], dropna=False)
        .agg(
            first_order=("order date (DateOrders)", "min"),
            last_order=("order date (DateOrders)", "max"),
            rows=("Order Item Id", "count"),
            total_quantity=("Order Item Quantity", "sum"),
            distinct_prices=("Order Item Product Price", "nunique"),
            min_price=("Order Item Product Price", "min"),
            max_price=("Order Item Product Price", "max"),
        )
        .reset_index()
        .sort_values("first_order")
    )

    cutoff = pd.Timestamp("2017-10-01")
    affected = product_summary[
        product_summary["first_order"] >= cutoff
    ].copy()

    print("=== PRODUCTS FIRST APPEARING FROM OCTOBER 2017 ===")
    print(
        affected.to_string(
            index=False,
            formatters={
                "min_price": "{:.2f}".format,
                "max_price": "{:.2f}".format,
            },
        )
    )

    print("\n=== PRODUCTS FIRST APPEARING BEFORE OCTOBER 2017 ===")
    print(
        product_summary[
            product_summary["first_order"] < cutoff
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
