
"""Identify products whose recorded demand ends before the dataset cutoff."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
    / "product_demand_ending_analysis.csv"
)


def main() -> None:
    """Summarize when each product's recorded demand ends."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["month"])
    df["month"] = pd.to_datetime(df["month"], errors="raise")

    cutoff = df["month"].max()

    summary = (
        df.groupby("product_id")
        .agg(
            first_recorded_month=("month", "min"),
            last_recorded_month=("month", "max"),
            recorded_months=("month", "nunique"),
            total_demand=("demand_units", "sum"),
        )
        .reset_index()
    )

    summary["months_since_last_record"] = (
        (cutoff.year - summary["last_recorded_month"].dt.year) * 12
        + cutoff.month
        - summary["last_recorded_month"].dt.month
    )

    summary["history_category"] = summary["recorded_months"].map(
        lambda n: (
            "12+ recorded months"
            if n >= 12
            else "6–11 recorded months"
            if n >= 6
            else "1–5 recorded months"
        )
    )

    summary = summary.sort_values(
        ["months_since_last_record", "recorded_months"],
        ascending=[False, True],
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\n=== PRODUCT DEMAND ENDING ANALYSIS ===")
    print(f"Dataset cutoff: {cutoff:%Y-%m}")
    print(f"Products analyzed: {len(summary)}")
    print("\nHistory categories:")
    print(summary["history_category"].value_counts().to_string())
    print("\nProducts with oldest last-recorded demand:")
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
