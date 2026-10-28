
"""Inspect the recorded history of products excluded from forecasting."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
SKIPPED_FILE = PROJECT_ROOT / "outputs" / "forecasts" / "forecast_products_skipped.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"


def main() -> None:
    """Summarize recorded demand history for skipped products."""
    for path in (DATA_FILE, SKIPPED_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    demand = pd.read_csv(DATA_FILE, parse_dates=["month"])
    skipped = pd.read_csv(SKIPPED_FILE)

    demand["month"] = pd.to_datetime(demand["month"], errors="raise")
    demand["product_id"] = demand["product_id"].astype(str)
    skipped["product_id"] = skipped["product_id"].astype(str)

    rows = []

    for product_id in skipped["product_id"]:
        product = demand.loc[demand["product_id"] == product_id].sort_values("month")

        if product.empty:
            rows.append({
                "product_id": product_id,
                "recorded_months": 0,
                "first_month": None,
                "last_month": None,
                "missing_months_between_records": None,
                "total_demand": 0,
                "reason": "No matching product in processed dataset",
            })
            continue

        first_month = product["month"].min()
        last_month = product["month"].max()
        expected_months = pd.date_range(first_month, last_month, freq="MS")
        missing_months = len(expected_months) - product["month"].nunique()

        rows.append({
            "product_id": product_id,
            "recorded_months": product["month"].nunique(),
            "first_month": first_month.strftime("%Y-%m"),
            "last_month": last_month.strftime("%Y-%m"),
            "missing_months_between_records": missing_months,
            "total_demand": product["demand_units"].sum(),
            "reason": "Review history and missing months",
        })

    report = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "skipped_product_history_analysis.csv"
    report.to_csv(output_file, index=False)

    print("\n=== SKIPPED PRODUCT HISTORY ===")
    print(f"Products inspected: {len(report)}")
    print(f"Products with gaps between first and last record: "
          f"{(report['missing_months_between_records'].fillna(0) > 0).sum()}")
    print(report.head(20).to_string(index=False))
    print(f"\nSaved report: {output_file}")


if __name__ == "__main__":
    main()
