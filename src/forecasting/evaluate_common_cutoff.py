
"""Compare baseline forecasts using a common historical test period."""

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
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

START_MONTH = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2016-10-01")
TEST_START = pd.Timestamp("2016-11-01")
TEST_END = pd.Timestamp("2017-04-01")
MOVING_AVERAGE_WINDOW = 3


def main() -> None:
    """Evaluate eligible products over the same six test months."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["month"])

    required = {"month", "product_id", "demand_units"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["demand_units"] = pd.to_numeric(df["demand_units"], errors="raise")

    if df[list(required)].isna().any().any():
        raise ValueError("Required fields contain missing values.")

    if not df["demand_units"].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all():
        raise ValueError("Demand contains non-finite values.")

    if (df["demand_units"] < 0).any():
        raise ValueError("Negative demand found; investigate before forecasting.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    expected_months = pd.date_range(START_MONTH, TEST_END, freq="MS")
    train_months = pd.date_range(START_MONTH, TRAIN_END, freq="MS")
    test_months = pd.date_range(TEST_START, TEST_END, freq="MS")

    records = []
    predictions = []

    period_df = df.loc[
        df["month"].between(START_MONTH, TEST_END)
    ].copy()

    for product_id, group in period_df.groupby("product_id"):
        group = group.sort_values("month").reset_index(drop=True)

        # Only compare products with a complete common historical window.
        if (
            len(group) != len(expected_months)
            or not group["month"].reset_index(drop=True).equals(
                pd.Series(expected_months, name="month")
            )
        ):
            continue

        demand = group["demand_units"].astype(float)
        train = demand.iloc[:len(train_months)].reset_index(drop=True)
        actual = demand.iloc[len(train_months):].reset_index(drop=True)

        naive_values = []
        moving_values = []

        # Rolling one-step forecasts: only earlier observations are used.
        history = train.copy()

        for actual_value in actual:
            naive_values.append(float(naive_forecast(history).iloc[-1]))
            moving_values.append(
                float(
                    moving_average_forecast(
                        history,
                        window=MOVING_AVERAGE_WINDOW,
                    ).iloc[-1]
                )
            )

            # Once a month has occurred, it becomes available for the next forecast.
            history = pd.concat(
                [history, pd.Series([actual_value])],
                ignore_index=True,
            )

        naive_series = pd.Series(naive_values)
        moving_series = pd.Series(moving_values)

        records.append(
            {
                "product_id": product_id,
                "train_months": len(train),
                "test_months": len(actual),
                "naive_mae": mean_absolute_error(actual, naive_series),
                "moving_average_mae": mean_absolute_error(
                    actual, moving_series
                ),
                "naive_rmse": root_mean_squared_error(
                    actual, naive_series
                ),
                "moving_average_rmse": root_mean_squared_error(
                    actual, moving_series
                ),
                "naive_wape_pct": weighted_absolute_percentage_error(
                    actual, naive_series
                ),
                "moving_average_wape_pct": weighted_absolute_percentage_error(
                    actual, moving_series
                ),
            }
        )

        for index, month in enumerate(test_months):
            predictions.append(
                {
                    "product_id": product_id,
                    "month": month,
                    "actual": float(actual.iloc[index]),
                    "naive_forecast": naive_values[index],
                    "moving_average_forecast": moving_values[index],
                }
            )

    if not records:
        raise ValueError(
            "No products have complete history over the common evaluation window."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(records)
    prediction_df = pd.DataFrame(predictions)

    results.to_csv(
        OUTPUT_DIR / "common_cutoff_model_comparison.csv",
        index=False,
    )
    prediction_df.to_csv(
        OUTPUT_DIR / "common_cutoff_predictions.csv",
        index=False,
    )

    print("\n=== COMMON-CUTOFF EVALUATION ===")
    print(f"Eligible products: {len(results)}")
    print(f"Training period: {START_MONTH:%Y-%m} to {TRAIN_END:%Y-%m}")
    print(f"Testing period: {TEST_START:%Y-%m} to {TEST_END:%Y-%m}")

    print("\n=== MEAN METRICS PER PRODUCT ===")
    for model, mae, rmse, wape in [
        (
            "Naive",
            "naive_mae",
            "naive_rmse",
            "naive_wape_pct",
        ),
        (
            "Moving Average",
            "moving_average_mae",
            "moving_average_rmse",
            "moving_average_wape_pct",
        ),
    ]:
        print(f"\n{model}")
        print(f"Mean MAE:  {results[mae].mean():.2f}")
        print(f"Mean RMSE: {results[rmse].mean():.2f}")
        print(f"Mean WAPE: {results[wape].mean():.2f}%")

    moving_wins = (
        results["moving_average_mae"] < results["naive_mae"]
    ).sum()
    naive_wins = (
        results["naive_mae"] < results["moving_average_mae"]
    ).sum()

    print("\n=== PRODUCT-LEVEL MAE WINS ===")
    print(f"Moving average wins: {moving_wins}")
    print(f"Naive wins: {naive_wins}")
    print(f"Ties: {len(results) - moving_wins - naive_wins}")

    print(f"\nSaved comparison: {OUTPUT_DIR / 'common_cutoff_model_comparison.csv'}")
    print(f"Saved predictions: {OUTPUT_DIR / 'common_cutoff_predictions.csv'}")


if __name__ == "__main__":
    main()
