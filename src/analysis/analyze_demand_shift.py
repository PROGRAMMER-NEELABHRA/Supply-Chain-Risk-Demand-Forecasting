
"""Analyze demand shifts between training and testing periods."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2017-04-01")
TEST_START = pd.Timestamp("2017-05-01")
TEST_END = pd.Timestamp("2017-10-01")


def load_data() -> pd.DataFrame:
    """Load and validate the monthly demand dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["month"])

    required = {"month", "product_id", "demand_units"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[
        df["month"].between(TRAIN_START, TEST_END),
        ["month", "product_id", "demand_units"],
    ].copy()

    df["demand_units"] = pd.to_numeric(
        df["demand_units"], errors="raise"
    )

    if df[list(required)].isna().any().any():
        raise ValueError("Required fields contain missing values.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    if (df["demand_units"] < 0).any():
        raise ValueError("Negative demand found.")

    return df.sort_values(["product_id", "month"])


def summarize_period(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prefix: str,
) -> pd.DataFrame:
    """Calculate demand summaries for a specified period."""
    period = df.loc[df["month"].between(start, end)]

    summary = (
        period.groupby("product_id")
        .agg(
            observed_months=("month", "nunique"),
            total_demand=("demand_units", "sum"),
            average_monthly_demand=("demand_units", "mean"),
            demand_std=("demand_units", "std"),
            minimum_monthly_demand=("demand_units", "min"),
            maximum_monthly_demand=("demand_units", "max"),
        )
        .add_prefix(f"{prefix}_")
    )

    return summary


def main() -> None:
    """Compare product demand between training and testing periods."""
    df = load_data()

    training = summarize_period(
        df, TRAIN_START, TRAIN_END, "train"
    )
    testing = summarize_period(
        df, TEST_START, TEST_END, "test"
    )

    comparison = training.join(testing, how="outer").reset_index()

    comparison["average_demand_change"] = (
        comparison["test_average_monthly_demand"]
        - comparison["train_average_monthly_demand"]
    )

    comparison["average_demand_change_pct"] = (
        comparison["average_demand_change"]
        / comparison["train_average_monthly_demand"].where(
            comparison["train_average_monthly_demand"] != 0
        )
        * 100
    )

    comparison["demand_shift"] = "Stable or modest change"

    comparison.loc[
        comparison["average_demand_change_pct"] >= 25,
        "demand_shift",
    ] = "Demand increased by at least 25%"

    comparison.loc[
        comparison["average_demand_change_pct"] <= -25,
        "demand_shift",
    ] = "Demand decreased by at least 25%"

    comparison.loc[
        comparison["train_average_monthly_demand"].isna(),
        "demand_shift",
    ] = "No training-period observations"

    comparison.loc[
        comparison["test_average_monthly_demand"].isna(),
        "demand_shift",
    ] = "No testing-period observations"

    comparison = comparison.sort_values(
        "average_demand_change_pct",
        ascending=False,
        na_position="last",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "demand_shift_analysis.csv"
    comparison.to_csv(output_file, index=False)

    print("\n=== DEMAND SHIFT ANALYSIS ===")
    print(f"Training period: {TRAIN_START:%Y-%m} to {TRAIN_END:%Y-%m}")
    print(f"Testing period: {TEST_START:%Y-%m} to {TEST_END:%Y-%m}")
    print(f"Products analyzed: {comparison['product_id'].nunique()}")

    print("\n=== OVERALL PERIOD SUMMARY ===")
    for label, start, end in [
        ("Training", TRAIN_START, TRAIN_END),
        ("Testing", TEST_START, TEST_END),
    ]:
        period = df.loc[df["month"].between(start, end)]
        print(
            f"{label}: "
            f"{period['product_id'].nunique()} products, "
            f"{period['month'].nunique()} calendar months, "
            f"{period['demand_units'].sum():,.0f} total units"
        )

    print("\n=== LARGEST DEMAND INCREASES ===")
    print(
        comparison.nlargest(10, "average_demand_change_pct")[
            [
                "product_id",
                "train_average_monthly_demand",
                "test_average_monthly_demand",
                "average_demand_change_pct",
                "demand_shift",
            ]
        ].to_string(index=False)
    )

    print("\n=== LARGEST DEMAND DECREASES ===")
    decreases = comparison.dropna(
        subset=["average_demand_change_pct"]
    ).nsmallest(10, "average_demand_change_pct")

    print(
        decreases[
            [
                "product_id",
                "train_average_monthly_demand",
                "test_average_monthly_demand",
                "average_demand_change_pct",
                "demand_shift",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved detailed results to: {output_file}")


if __name__ == "__main__":
    main()
