
"""Analyze demand shifts for products in the later-period evaluation."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)
METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
    / "later_period_per_product_metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2017-04-01")
TEST_START = pd.Timestamp("2017-05-01")
TEST_END = pd.Timestamp("2017-10-01")


def main() -> None:
    """Compare average demand for the evaluated products across periods."""
    for file_path in (DATA_FILE, METRICS_FILE):
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    demand = pd.read_csv(DATA_FILE, parse_dates=["month"])
    metrics = pd.read_csv(METRICS_FILE)

    required_demand = {"month", "product_id", "demand_units"}
    missing_demand = required_demand - set(demand.columns)
    if missing_demand:
        raise ValueError(
            f"Demand dataset is missing: {sorted(missing_demand)}"
        )

    if "product_id" not in metrics.columns:
        raise ValueError("Metrics file must contain product_id.")

    product_ids = metrics["product_id"].dropna().unique()

    if len(product_ids) != 9:
        print(
            "WARNING: Expected 9 products, but found "
            f"{len(product_ids)} unique product IDs."
        )

    demand = demand.loc[
        demand["product_id"].isin(product_ids)
        & demand["month"].between(TRAIN_START, TEST_END)
    ].copy()

    if demand.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    if demand["demand_units"].isna().any():
        raise ValueError("Demand contains missing values.")

    train = demand.loc[
        demand["month"].between(TRAIN_START, TRAIN_END)
    ]
    test = demand.loc[
        demand["month"].between(TEST_START, TEST_END)
    ]

    train_summary = (
        train.groupby("product_id")
        .agg(
            train_months=("month", "nunique"),
            train_average_demand=("demand_units", "mean"),
            train_total_demand=("demand_units", "sum"),
        )
    )

    test_summary = (
        test.groupby("product_id")
        .agg(
            test_months=("month", "nunique"),
            test_average_demand=("demand_units", "mean"),
            test_total_demand=("demand_units", "sum"),
        )
    )

    result = train_summary.join(test_summary, how="outer").reset_index()

    result["average_demand_change"] = (
        result["test_average_demand"]
        - result["train_average_demand"]
    )

    result["average_demand_change_pct"] = (
        result["average_demand_change"]
        / result["train_average_demand"].where(
            result["train_average_demand"] != 0
        )
        * 100
    )

    result["demand_change_label"] = "Change below ±25%"

    result.loc[
        result["average_demand_change_pct"] >= 25,
        "demand_change_label",
    ] = "Demand increased by at least 25%"

    result.loc[
        result["average_demand_change_pct"] <= -25,
        "demand_change_label",
    ] = "Demand decreased by at least 25%"

    result.loc[
        result["train_months"].isna(),
        "demand_change_label",
    ] = "No training-period observations"

    result.loc[
        result["test_months"].isna(),
        "demand_change_label",
    ] = "No testing-period observations"

    result = result.merge(
        metrics.drop_duplicates("product_id"),
        on="product_id",
        how="left",
        validate="one_to_one",
    )

    result = result.sort_values(
        "average_demand_change_pct",
        ascending=False,
        na_position="last",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "later_period_demand_shift.csv"
    result.to_csv(output_file, index=False)

    print("\n=== LATER-PERIOD DEMAND SHIFT ===")
    print(f"Products analyzed: {result['product_id'].nunique()}")
    print(f"Training period: {TRAIN_START:%Y-%m} to {TRAIN_END:%Y-%m}")
    print(f"Testing period: {TEST_START:%Y-%m} to {TEST_END:%Y-%m}")

    print("\n=== PRODUCT-LEVEL RESULTS ===")
    display_columns = [
        "product_id",
        "train_months",
        "test_months",
        "train_average_demand",
        "test_average_demand",
        "average_demand_change_pct",
        "demand_change_label",
    ]

    print(result[display_columns].round(2).to_string(index=False))

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
