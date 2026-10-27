
"""Investigate product and order-status changes in raw DataCo data."""

from src.data.loaders import load_dataco_data


def main():
    """Compare product and order-status patterns before and after October 2017."""

    data = load_dataco_data()

    data["order date (DateOrders)"] = __import__("pandas").to_datetime(
        data["order date (DateOrders)"],
        format="mixed",
        errors="coerce",
    )

    data["period"] = data["order date (DateOrders)"].dt.to_period("M")

    periods = {
        "Before change (Jun-Sep 2017)": data["period"].between(
            "2017-06", "2017-09"
        ),
        "Transition (Oct 2017)": data["period"] == "2017-10",
        "After change (Nov 2017-Jan 2018)": data["period"].between(
            "2017-11", "2018-01"
        ),
    }

    for label, mask in periods.items():
        subset = data.loc[mask]

        print(f"\n=== {label} ===")
        print(f"Rows: {len(subset):,}")
        print(f"Distinct products: {subset['Product Card Id'].nunique()}")

        print("\nTop products by row count:")
        print(
            subset.groupby(["Product Card Id", "Product Name"])
            .size()
            .sort_values(ascending=False)
            .head(10)
            .to_string()
        )

        print("\nOrder status:")
        print(subset["Order Status"].value_counts(dropna=False).to_string())

        print("\nShipping mode:")
        print(subset["Shipping Mode"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
