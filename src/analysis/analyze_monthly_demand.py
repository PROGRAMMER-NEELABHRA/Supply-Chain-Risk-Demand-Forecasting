
"""Analyze monthly demand coverage before forecasting."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "monthly_demand_observed.csv"
)


def main() -> None:
    """Summarize monthly coverage and demand patterns by product."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["month"])

    coverage = (
        df.groupby("product_id")
        .agg(
            observed_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            total_demand=("demand_units", "sum"),
            average_observed_monthly_demand=("demand_units", "mean"),
        )
        .reset_index()
    )

    print("\n=== MONTHLY HISTORY PER PRODUCT ===")
    print(coverage["observed_months"].describe().to_string())

    print("\n=== PRODUCTS WITH THE SHORTEST HISTORY ===")
    print(
        coverage.sort_values("observed_months")
        .head(10)
        .to_string(index=False)
    )

    print("\n=== PRODUCTS WITH THE LONGEST HISTORY ===")
    print(
        coverage.sort_values("observed_months", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\n=== DEMAND DISTRIBUTION ===")
    print(df["demand_units"].describe().to_string())

    print("\n=== ZERO OR NEGATIVE OBSERVED DEMAND ===")
    print(f"Zero-demand records: {(df['demand_units'] == 0).sum()}")
    print(f"Negative-demand records: {(df['demand_units'] < 0).sum()}")

    print("\n=== MISSING MONTHS WITHIN EACH PRODUCT'S HISTORY ===")
    gaps = []
    for product_id, group in df.groupby("product_id"):
        group = group.sort_values("month")
        expected = pd.period_range(
            group["month"].min(),
            group["month"].max(),
            freq="M",
        )
        observed = group["month"].dt.to_period("M")
        gaps.append(
            {
                "product_id": product_id,
                "missing_months": len(expected.difference(observed)),
            }
        )

    gaps_df = pd.DataFrame(gaps)
    print(gaps_df["missing_months"].describe().to_string())
    print(f"Products with internal gaps: {(gaps_df['missing_months'] > 0).sum()}")


if __name__ == "__main__":
    main()
