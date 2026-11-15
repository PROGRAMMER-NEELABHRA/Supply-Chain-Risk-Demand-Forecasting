
"""Investigate missing product-month demand records against raw orders."""

from pathlib import Path

import pandas as pd

from config import DATACO_RAW_DATA_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = DATACO_RAW_DATA_FILE
MONTHLY_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"


def main() -> None:
    """Compare raw orders with processed monthly demand."""
    raw = pd.read_csv(RAW_FILE, low_memory=False)
    monthly = pd.read_csv(MONTHLY_FILE)

    required_raw = {"order_date", "product_id", "quantity"}
    missing_raw = required_raw - set(raw.columns)
    if missing_raw:
        raise ValueError(
            f"Raw dataset is missing columns: {sorted(missing_raw)}. "
            "Inspect the raw quantity column name before continuing."
        )

    required_monthly = {"month", "product_id", "demand_units"}
    missing_monthly = required_monthly - set(monthly.columns)
    if missing_monthly:
        raise ValueError(
            f"Monthly dataset is missing columns: {sorted(missing_monthly)}"
        )

    raw["order_date"] = pd.to_datetime(
        raw["order_date"],
        errors="coerce",
        dayfirst=True,
    )
    raw["product_id"] = pd.to_numeric(raw["product_id"], errors="raise")
    monthly["month"] = pd.to_datetime(monthly["month"], errors="raise")
    monthly["product_id"] = pd.to_numeric(
        monthly["product_id"], errors="raise"
    )

    print("Raw columns:", raw.columns.tolist())

    # Check the actual quantity field before calculating raw demand.
    quantity_column = "quantity"
    if quantity_column not in raw.columns:
        raise ValueError(
            "The raw dataset has no 'quantity' column. "
            "Check the printed columns and identify the order quantity field."
        )

    raw[quantity_column] = pd.to_numeric(
        raw[quantity_column], errors="raise"
    )
    raw["month"] = raw["order_date"].dt.to_period("M").dt.to_timestamp()

    raw_monthly = (
        raw.dropna(subset=["month", "product_id"])
        .groupby(["product_id", "month"], as_index=False)
        .agg(raw_order_rows=("order_date", "size"),
             raw_demand_units=(quantity_column, "sum"))
    )

    processed = monthly.rename(
        columns={"month": "month"}
    ).copy()

    comparison = raw_monthly.merge(
        processed,
        on=["product_id", "month"],
        how="outer",
        indicator=True,
    )

    print("\n=== RAW VS PROCESSED MONTHLY COVERAGE ===")
    print(comparison["_merge"].value_counts().to_string())

    missing_processed = comparison.loc[
        comparison["_merge"] == "left_only"
    ].copy()

    missing_processed["month"] = pd.to_datetime(missing_processed["month"])

    print("\n=== RAW MONTHS MISSING FROM PROCESSED DATA ===")
    print(f"Missing product-month rows: {len(missing_processed)}")
    if not missing_processed.empty:
        print(missing_processed.head(20).to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "raw_processed_monthly_comparison.csv"
    comparison.to_csv(output_file, index=False)

    print(f"\nSaved comparison: {output_file}")


if __name__ == "__main__":
    main()
