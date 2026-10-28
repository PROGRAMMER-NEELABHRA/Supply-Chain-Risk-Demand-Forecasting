
"""Compare naive and moving-average forecasts for one product."""

from pathlib import Path

import pandas as pd

from src.forecasting.baseline import naive_forecast, moving_average_forecast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)

PRODUCT_ID = 365
TEST_MONTHS = 6


def main() -> None:
    """Run a chronological baseline experiment."""
    df = pd.read_csv(DATA_FILE, parse_dates=["month"])

    product = (
        df.loc[df["product_id"] == PRODUCT_ID]
        .sort_values("month")
        .copy()
    )

    if product.empty:
        raise ValueError(f"No data found for product {PRODUCT_ID}.")

    expected_months = pd.date_range(
        product["month"].min(),
        product["month"].max(),
        freq="MS",
    )

    if not product["month"].reset_index(drop=True).equals(
        pd.Series(expected_months, name="month")
    ):
        raise ValueError(
            "Product history is not continuous. Resolve missing months first."
        )

    if len(product) <= TEST_MONTHS + 3:
        raise ValueError("Insufficient history for this experiment.")

    values = product["demand_units"].astype(float).reset_index(drop=True)
    train = values.iloc[:-TEST_MONTHS]
    actual = values.iloc[-TEST_MONTHS:]

    naive_predictions = naive_forecast(train, len(actual))
    moving_predictions = moving_average_forecast(
        train, horizon=len(actual), window=3
    )

    results = pd.DataFrame(
        {
            "month": product["month"].iloc[-TEST_MONTHS:].to_numpy(),
            "actual": actual.to_numpy(),
            "naive_forecast": naive_predictions,
            "moving_average_forecast": moving_predictions,
        }
    )

    results["naive_absolute_error"] = (
        results["actual"] - results["naive_forecast"]
    ).abs()
    results["moving_average_absolute_error"] = (
        results["actual"] - results["moving_average_forecast"]
    ).abs()

    output_file = PROJECT_ROOT / "outputs" / "forecasts" / "product_365_baseline.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print("\n=== BASELINE RESULTS ===")
    print(results.to_string(index=False))
    print("\nMAE:")
    print(f"Naive: {results['naive_absolute_error'].mean():.2f}")
    print(
        "Moving average: "
        f"{results['moving_average_absolute_error'].mean():.2f}"
    )
    print(f"\nSaved results: {output_file}")


if __name__ == "__main__":
    main()
