
"""Validate monthly product demand against processed sales data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

SALES_PATH = DATA_DIR / "sales.csv"
MONTHLY_DEMAND_PATH = DATA_DIR / "monthly_demand.csv"


def validate_monthly_demand() -> None:
    """Check row counts, demand totals, and monthly aggregation."""
    if not SALES_PATH.is_file():
        raise FileNotFoundError(f"Sales file not found: {SALES_PATH}")

    if not MONTHLY_DEMAND_PATH.is_file():
        raise FileNotFoundError(
            f"Monthly demand file not found: {MONTHLY_DEMAND_PATH}"
        )

    sales = pd.read_csv(SALES_PATH)
    monthly = pd.read_csv(MONTHLY_DEMAND_PATH)

    required = {"order_date", "product_id", "quantity"}
    if not required.issubset(sales.columns):
        raise ValueError("Sales data is missing required columns.")

    monthly_required = {
        "month", "product_id", "demand_units",
        "order_item_rows", "active_days",
    }
    if not monthly_required.issubset(monthly.columns):
        raise ValueError("Monthly demand data is missing required columns.")

    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="raise")
    monthly["demand_units"] = pd.to_numeric(
        monthly["demand_units"], errors="raise"
    )

    source_units = sales["quantity"].sum()
    aggregated_units = monthly["demand_units"].sum()

    print("=== MONTHLY DEMAND VALIDATION ===")
    print(f"Source sales rows: {len(sales):,}")
    print(f"Monthly product rows: {len(monthly):,}")
    print(f"Source demand units: {source_units:,.0f}")
    print(f"Aggregated demand units: {aggregated_units:,.0f}")
    print(f"Products in source: {sales['product_id'].nunique():,}")
    print(f"Products in monthly data: {monthly['product_id'].nunique():,}")

    if not monthly[["month", "product_id"]].duplicated().sum() == 0:
        raise ValueError("Duplicate month-product combinations found.")

    if source_units != aggregated_units:
        raise ValueError("Demand totals do not match the source sales data.")

    if sales["product_id"].nunique() != monthly["product_id"].nunique():
        raise ValueError("Product counts do not match.")

    source_dates = pd.to_datetime(sales["order_date"], errors="raise")
    if monthly["month"].isna().any():
        raise ValueError("Missing month values found.")

    expected_months = source_dates.dt.to_period("M").nunique()
    actual_months = monthly["month"].nunique()

    print(f"Months in source: {expected_months}")
    print(f"Months in monthly data: {actual_months}")

    if expected_months != actual_months:
        raise ValueError("Month counts do not match.")

    print("\nPASS: Monthly demand aggregation checks succeeded.")


if __name__ == "__main__":
    validate_monthly_demand()
