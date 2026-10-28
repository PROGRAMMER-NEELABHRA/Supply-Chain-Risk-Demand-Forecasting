
"""Aggregate order records into monthly product demand."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "integrated_supply_chain.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Create a monthly demand table for each product."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"order_date", "product_id", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce", dayfirst=True
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    if df[["order_date", "product_id", "quantity"]].isna().any().any():
        raise ValueError("Missing or invalid dates, product IDs, or quantities.")

    if (df["quantity"] < 0).any():
        raise ValueError("Negative quantities found; investigate before aggregating.")

    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["month", "product_id"], as_index=False)
        .agg(demand_units=("quantity", "sum"))
        .sort_values(["product_id", "month"])
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "monthly_demand_observed.csv"
    monthly.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Monthly product records: {len(monthly):,}")
    print(f"Products: {monthly['product_id'].nunique():,}")
    print(f"Months: {monthly['month'].nunique():,}")
    print(f"Total demand: {monthly['demand_units'].sum():,.0f}")


if __name__ == "__main__":
    main()
