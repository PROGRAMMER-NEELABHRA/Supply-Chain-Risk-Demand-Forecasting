
"""Find products with at least 24 consecutive observed months."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)


def main() -> None:
    """Identify products with long, uninterrupted monthly histories."""
    df = pd.read_csv(INPUT_FILE, parse_dates=["month"])
    df["month"] = df["month"].dt.to_period("M")

    results = []

    for product_id, group in df.groupby("product_id"):
        months = sorted(group["month"].unique())

        if not months:
            continue

        # Find the longest uninterrupted sequence of observed months.
        longest_run = 1
        current_run = 1

        for previous, current in zip(months, months[1:]):
            if current.ordinal - previous.ordinal == 1:
                current_run += 1
            else:
                current_run = 1

            longest_run = max(longest_run, current_run)

        results.append(
            {
                "product_id": product_id,
                "observed_months": len(months),
                "longest_consecutive_months": longest_run,
                "first_month": str(months[0]),
                "last_month": str(months[-1]),
                "total_demand": group["demand_units"].sum(),
            }
        )

    summary = pd.DataFrame(results)
    eligible = summary[
        summary["longest_consecutive_months"] >= 24
    ].sort_values("total_demand", ascending=False)

    print(f"Total products: {len(summary)}")
    print(f"Products with at least 24 consecutive months: {len(eligible)}")
    print("\nTop candidates:")
    print(eligible.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
