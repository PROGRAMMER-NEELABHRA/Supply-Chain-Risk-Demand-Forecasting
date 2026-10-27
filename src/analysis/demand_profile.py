
"""Generate a monthly demand profile for quality analysis."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

SALES_PATH = DATA_DIR / "sales.csv"
TIMESERIES_PATH = DATA_DIR / "demand_timeseries.csv"
OUTPUT_PATH = DATA_DIR / "monthly_demand_profile.csv"

ANOMALY_START_MONTH = "2017-10"


def main() -> None:
    """Summarize monthly demand and flag the known source-data shift."""
    sales = pd.read_csv(SALES_PATH)
    timeseries = pd.read_csv(TIMESERIES_PATH)

    sales["order_date"] = pd.to_datetime(
        sales["order_date"], errors="raise"
    )
    sales["month"] = sales["order_date"].dt.to_period("M").astype(str)

    monthly_sales = sales.groupby("month").agg(
        total_units=("quantity", "sum"),
        order_item_rows=("order_item_id", "nunique"),
        distinct_orders=("order_id", "nunique"),
    )

    product_months = timeseries.groupby("month").agg(
        products_in_catalog=("product_id", "nunique"),
        products_with_recorded_demand=(
            "demand_units",
            lambda values: (values > 0).sum(),
        ),
        zero_demand_product_months=(
            "demand_units",
            lambda values: (values == 0).sum(),
        ),
    )

    profile = monthly_sales.join(product_months).reset_index()
    profile["period_flag"] = profile["month"].apply(
        lambda month: (
            "investigate_source_shift"
            if month >= ANOMALY_START_MONTH
            else "before_source_shift"
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUTPUT_PATH, index=False)

    print("=== MONTHLY DEMAND PROFILE ===")
    print(profile.to_string(index=False))
    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
