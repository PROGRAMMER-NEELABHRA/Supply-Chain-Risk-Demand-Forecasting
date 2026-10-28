
"""Inspect product status values and raw order-date coverage."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "integrated_supply_chain.csv"


def main() -> None:
    """Print the raw dataset columns and status/date summaries."""
    df = pd.read_csv(RAW_FILE, low_memory=False)

    print("=== RELEVANT COLUMNS ===")
    for column in df.columns:
        if any(
            word in column.lower()
            for word in ("product", "status", "date", "order")
        ):
            print(f"{column}: {df[column].dtype}")

    print("\n=== PRODUCT STATUS VALUES ===")
    print(df["product_status"].value_counts(dropna=False).to_string())

    date_columns = [
        column for column in df.columns
        if "date" in column.lower() or "time" in column.lower()
    ]

    print("\n=== RAW DATE COVERAGE ===")
    for column in date_columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().any():
            print(
                f"{column}: {parsed.min()} to {parsed.max()} "
                f"({parsed.notna().sum()} valid values)"
            )

    print("\n=== SAMPLE PRODUCT RECORDS ===")
    product_columns = [
        column for column in df.columns
        if "product" in column.lower() or "status" in column.lower()
    ]
    print(df[product_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
