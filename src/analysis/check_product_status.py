
"""Check product status against recent monthly demand coverage."""

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
    """Summarize product status and last recorded demand month."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    if not MONTHLY_FILE.exists():
        raise FileNotFoundError(f"Monthly dataset not found: {MONTHLY_FILE}")

    raw = pd.read_csv(RAW_FILE, low_memory=False)
    monthly = pd.read_csv(MONTHLY_FILE, parse_dates=["month"])

    required = {"product_id", "product_status"}
    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Raw dataset is missing columns: {sorted(missing)}"
        )

    if "product_id" not in monthly.columns or "month" not in monthly.columns:
        raise ValueError("Monthly dataset must contain product_id and month.")

    # Keep one status per product and flag conflicting statuses.
    status_rows = raw[["product_id", "product_status"]].drop_duplicates()

    status_summary = (
        status_rows.groupby("product_id")["product_status"]
        .agg(
            lambda values: " | ".join(
                sorted(values.dropna().astype(str).unique())
            )
        )
        .rename("product_status")
    )

    status_counts = (
        status_rows.groupby("product_id")["product_status"]
        .nunique(dropna=True)
        .rename("distinct_status_count")
    )

    last_recorded = (
        monthly.groupby("product_id")["month"]
        .max()
        .rename("last_recorded_month")
    )

    report = pd.concat(
        [status_summary, status_counts, last_recorded],
        axis=1,
    ).reset_index()

    cutoff_month = monthly["month"].max().to_period("M")

    report["months_since_last_record"] = (
        cutoff_month - report["last_recorded_month"].dt.to_period("M")
    ).map(lambda difference: difference.n)

    report["status_conflict"] = report["distinct_status_count"] > 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "product_status_coverage.csv"
    report.to_csv(output_file, index=False)

    print("\n=== RAW DATA STATUS VALUES ===")
    print(
        raw["product_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== STATUS AND DEMAND COVERAGE ===")
    print(f"Raw products: {raw['product_id'].nunique()}")
    print(f"Monthly products: {monthly['product_id'].nunique()}")
    print(f"Monthly data cutoff: {cutoff_month}")
    print(f"Products with conflicting statuses: {report['status_conflict'].sum()}")

    print("\n=== PRODUCTS WITH OLDEST LAST-RECORDED DEMAND ===")
    print(
        report.sort_values("last_recorded_month")
        .head(20)
        .to_string(index=False)
    )

    print(f"\nSaved report: {output_file}")

    print("\n=== PRODUCT STATUS AND PRODUCT COVERAGE ===")

raw = pd.read_csv(
    DATACO_RAW_DATA_FILE
)

print("Unique product status values:")
print(raw["product_status"].value_counts(dropna=False))

print("\nDistinct products per status:")
print(
    raw.groupby("product_status")["product_id"]
    .nunique()
    .to_string()
)

print("\nSample products by status:")
print(
    raw[["product_id", "product_name", "product_status"]]
    .drop_duplicates()
    .groupby("product_status")
    .head(10)
    .to_string(index=False)
)


if __name__ == "__main__":
    main()
