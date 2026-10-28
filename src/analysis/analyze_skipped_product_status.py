
"""Compare skipped-product history with raw product status."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "integrated_supply_chain.csv"
SKIPPED_FILE = (
    PROJECT_ROOT / "outputs" / "forecasts" / "forecast_products_skipped.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"


def main() -> None:
    """Summarize status and demand coverage for skipped products."""
    for path in (RAW_FILE, SKIPPED_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    raw = pd.read_csv(
        RAW_FILE,
        usecols=["product_id", "product_status", "order_date", "quantity"],
    )
    skipped = pd.read_csv(SKIPPED_FILE)

    raw["product_id"] = raw["product_id"].astype(str)
    skipped["product_id"] = skipped["product_id"].astype(str)
    raw["order_date"] = pd.to_datetime(
        raw["order_date"], errors="coerce", dayfirst=True
    )
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")

    status_counts = raw.groupby("product_id")["product_status"].nunique()
    conflicting_ids = status_counts[status_counts > 1].index

    status_summary = (
        raw.groupby("product_id", as_index=False)
        .agg(
            product_status=("product_status", "first"),
            distinct_status_count=("product_status", "nunique"),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            order_rows=("order_date", "count"),
            total_quantity=("quantity", "sum"),
        )
    )

    status_summary["product_id"] = status_summary["product_id"].astype(str)
    report = skipped.merge(
        status_summary,
        on="product_id",
        how="left",
        validate="one_to_one",
    )

    report["status_conflict"] = report["product_id"].isin(conflicting_ids)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "skipped_product_status_analysis.csv"
    report.to_csv(output_file, index=False)

    print("\n=== SKIPPED PRODUCT STATUS SUMMARY ===")
    print(report["product_status"].value_counts(dropna=False).to_string())
    print(f"\nProducts with conflicting statuses: {report['status_conflict'].sum()}")
    print("\n=== SAMPLE PRODUCTS ===")
    print(report.head(20).to_string(index=False))
    print(f"\nSaved report: {output_file}")


if __name__ == "__main__":
    main()
