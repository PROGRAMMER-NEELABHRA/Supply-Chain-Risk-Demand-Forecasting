
"""Compare naive and moving-average forecasts for Product 365."""

from pathlib import Path

import pandas as pd

from src.forecasting.baseline import (
    naive_forecast,
    moving_average_forecast,
)
from src.forecasting.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "monthly_demand_observed.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
    / "product_365_baseline.csv"
)

PRODUCT_ID = 365
TEST_MONTHS = 6
MOVING_AVERAGE_WINDOW = 3


def main() -> None:
    """Evaluate naive and moving-average forecasts chronologically."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["month"])

    product = (
        df.loc[df["product_id"] == PRODUCT_ID]
        .sort_values("month")
        .reset_index(drop=True)
    )

    if product.empty:
        raise ValueError(f"No records found for product {PRODUCT_ID}.")

    if product["month"].duplicated().any():
        raise ValueError(
            f"Product {PRODUCT_ID} has duplicate monthly records."
        )

    expected_months = pd.date_range(
        start=product["month"].min(),
        end=product["month"].max(),
        freq="MS",
    )

    if not product["month"].reset_index(drop=True).equals(
        pd.Series(expected_months, name="month")
    ):
        raise ValueError(
            "Product history contains missing months. "
            "Investigate the gaps before forecasting."
        )

    demand = pd.to_numeric(product["demand_units"], errors="coerce")

    if demand.isna().any():
        raise ValueError("Demand contains missing or non-numeric values.")

    if not demand.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all():
        raise ValueError("Demand contains non-finite values.")

    if len(demand) <= TEST_MONTHS + MOVING_AVERAGE_WINDOW:
        raise ValueError("Not enough history for this evaluation.")

    # Reserve the final six observations for chronological testing.
    split_index = len(demand) - TEST_MONTHS

    actual = demand.iloc[split_index:].reset_index(drop=True)
    test_months = product["month"].iloc[split_index:].reset_index(drop=True)

    naive_predictions = []
    moving_average_predictions = []

    # Rolling one-step-ahead evaluation:
    # each prediction uses only demand available before that month.
    for index in range(split_index, len(demand)):
        history = demand.iloc[:index]

        naive_value = naive_forecast(history).iloc[-1]

        moving_average_value = moving_average_forecast(
            history,
            window=MOVING_AVERAGE_WINDOW,
        ).iloc[-1]

        naive_predictions.append(float(naive_value))
        moving_average_predictions.append(float(moving_average_value))

    results = pd.DataFrame(
        {
            "month": test_months,
            "actual": actual,
            "naive_forecast": naive_predictions,
            "moving_average_forecast": moving_average_predictions,
        }
    )

    results["naive_absolute_error"] = (
        results["actual"] - results["naive_forecast"]
    ).abs()

    results["moving_average_absolute_error"] = (
        results["actual"] - results["moving_average_forecast"]
    ).abs()

    # Save detailed predictions and errors.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    # Evaluate each model over the complete test period.
    actual_values = results["actual"]

    print("\n=== MODEL EVALUATION: FULL TEST PERIOD ===")

    for model_name, prediction_column in [
        ("Naive", "naive_forecast"),
        ("Moving Average", "moving_average_forecast"),
    ]:
        predicted_values = results[prediction_column]

        print(f"\n{model_name}")
        print(
            f"MAE:  "
            f"{mean_absolute_error(actual_values, predicted_values):.2f}"
        )
        print(
            f"RMSE: "
            f"{root_mean_squared_error(actual_values, predicted_values):.2f}"
        )
        print(
            f"WAPE: "
            f"{weighted_absolute_percentage_error(actual_values, predicted_values):.2f}%"
        )

    # Diagnostic comparison before October 2017.
    # This is not the official full-test score.
    results["month"] = pd.to_datetime(results["month"])

    before_october = results.loc[
        results["month"] < pd.Timestamp("2017-10-01")
    ]

    print("\n=== DIAGNOSTIC: BEFORE OCTOBER 2017 ===")

    if before_october.empty:
        print("No test observations before October 2017.")
    else:
        for model_name, prediction_column in [
            ("Naive", "naive_forecast"),
            ("Moving Average", "moving_average_forecast"),
        ]:
            print(
                f"{model_name} MAE: "
                f"{mean_absolute_error(before_october['actual'], before_october[prediction_column]):.2f}"
            )

    print("\n=== TEST PREDICTIONS ===")
    print(results.to_string(index=False))

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
