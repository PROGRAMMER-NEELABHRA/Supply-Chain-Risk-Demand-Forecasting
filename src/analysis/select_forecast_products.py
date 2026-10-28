
"""Identify products with sufficient observed history for initial testing."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Summarize product history and save an initial forecasting shortlist."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["month"])

    history = (
        df.groupby("product_id")
        .agg(
            observed_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            total_demand=("demand_units", "sum"),
        )
        .reset_index()
    )

    eligible = history.loc[history["observed_months"] >= 24].copy()
    eligible = eligible.sort_values("total_demand", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "forecast_product_shortlist.csv"
    eligible.to_csv(output_file, index=False)

    print(f"Products in dataset: {len(history)}")
    print(f"Products with at least 24 observed months: {len(eligible)}")
    print(f"Saved shortlist: {output_file}")

    print("\nTop 10 eligible products by total demand:")
    print(eligible.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
