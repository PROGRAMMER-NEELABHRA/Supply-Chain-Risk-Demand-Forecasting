
"""Evaluate naive and moving-average baselines across multiple products."""

from pathlib import Path

import pandas as pd

from src.data.loaders import load_monthly_demand
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

MIN_CONSECUTIVE_MONTHS = 24
TEST_MONTHS = 6
MOVING_AVERAGE_WINDOW = 3


def longest_consecutive_run(months: pd.Series) -> int:
    """Return the longest run of consecutive calendar months."""
    periods = sorted(months.dt.to_period("M").unique())

    if not periods:
        return 0

    longest = 1
    current = 1

    for previous, current_period in zip(periods, periods[1:]):
        if current_period.ordinal - previous.ordinal == 1:
            current += 1
        else:
            current = 1

        longest = max(longest, current)

    return longest


def forecast_test_period(
    demand: pd.Series,
    split_index: int,
) -> tuple[list[float], list[float]]:
    """Generate rolling one-step forecasts without using future observations."""
    naive_predictions = []
    moving_average_predictions = []

    for index in range(split_index, len(demand)):
        history = demand.iloc[:index]

        naive_predictions.append(
            float(naive_forecast(history).iloc[-1])
        )

        moving_average_predictions.append(
            float(
                moving_average_forecast(
                    history,
                    window=MOVING_AVERAGE_WINDOW,
                ).iloc[-1]
            )
        )

    return naive_predictions, moving_average_predictions


def main() -> None:
    """Evaluate baseline models across eligible products."""
    df = load_monthly_demand(DATA_FILE)

    product_results = []
    prediction_results = []
    skipped_products = []

    for product_id, group in df.groupby("product_id"):
        group = group.sort_values("month").reset_index(drop=True)

        consecutive_months = longest_consecutive_run(group["month"])

        if consecutive_months < MIN_CONSECUTIVE_MONTHS:
            skipped_products.append(
                {
                    "product_id": product_id,
                    "reason": "Insufficient consecutive monthly history",
                    "longest_consecutive_months": consecutive_months,
                }
            )
            continue

        # Select the longest consecutive segment for valid time-series testing.
        periods = group["month"].dt.to_period("M")
        best_start = 0
        current_start = 0
        best_length = 1
        current_length = 1

        for index in range(1, len(group)):
            if periods.iloc[index].ordinal - periods.iloc[index - 1].ordinal == 1:
                current_length += 1
            else:
                current_start = index
                current_length = 1

            if current_length > best_length:
                best_start = current_start
                best_length = current_length

        segment = group.iloc[best_start:best_start + best_length].reset_index(
            drop=True
        )

        if len(segment) <= TEST_MONTHS + MOVING_AVERAGE_WINDOW:
            skipped_products.append(
                {
                    "product_id": product_id,
                    "reason": "Insufficient history after selecting continuous segment",
                    "longest_consecutive_months": len(segment),
                }
            )
            continue

        demand = segment["demand_units"].astype(float)
        split_index = len(segment) - TEST_MONTHS

        naive_predictions, moving_predictions = forecast_test_period(
            demand,
            split_index,
        )

        test = segment.iloc[split_index:].copy()

        actual = test["demand_units"].reset_index(drop=True)
        naive_series = pd.Series(naive_predictions)
        moving_series = pd.Series(moving_predictions)

        naive_mae = mean_absolute_error(actual, naive_series)
        moving_mae = mean_absolute_error(actual, moving_series)

        naive_rmse = root_mean_squared_error(actual, naive_series)
        moving_rmse = root_mean_squared_error(actual, moving_series)

        naive_wape = weighted_absolute_percentage_error(
            actual,
            naive_series,
        )
        moving_wape = weighted_absolute_percentage_error(
            actual,
            moving_series,
        )

        # Flag a test period ending in a product's final observed month.
        last_observed_month = group["month"].max()
        final_test_month = test["month"].max()
        ends_at_last_observation = final_test_month == last_observed_month

        product_results.append(
            {
                "product_id": product_id,
                "continuous_months": len(segment),
                "train_months": split_index,
                "test_months": len(test),
                "first_test_month": test["month"].min(),
                "last_test_month": final_test_month,
                "test_ends_at_last_observation": ends_at_last_observation,
                "naive_mae": naive_mae,
                "moving_average_mae": moving_mae,
                "naive_rmse": naive_rmse,
                "moving_average_rmse": moving_rmse,
                "naive_wape_pct": naive_wape,
                "moving_average_wape_pct": moving_wape,
            }
        )

        for row_index, (_, row) in enumerate(test.iterrows()):
            prediction_results.append(
                {
                    "product_id": product_id,
                    "month": row["month"],
                    "actual": float(row["demand_units"]),
                    "naive_forecast": naive_predictions[row_index],
                    "moving_average_forecast": moving_predictions[row_index],
                    "test_ends_at_last_observation": ends_at_last_observation,
                }
            )

    if not product_results:
        raise ValueError(
            "No eligible products found. Check the monthly history and thresholds."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(product_results)
    predictions = pd.DataFrame(prediction_results)
    skipped = pd.DataFrame(skipped_products)

    results.to_csv(OUTPUT_DIR / "multi_product_model_comparison.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "multi_product_predictions.csv", index=False)

    if not skipped.empty:
        skipped.to_csv(OUTPUT_DIR / "skipped_products.csv", index=False)

    print("\n=== MULTI-PRODUCT EVALUATION ===")
    print(f"Products evaluated: {len(results)}")
    print(f"Products skipped: {len(skipped)}")
    print(
        "Products whose test period ends at their final observed month: "
        f"{results['test_ends_at_last_observation'].sum()}"
    )

    print("\n=== AVERAGE METRICS ACROSS PRODUCTS ===")
    for model_name, mae_col, rmse_col, wape_col in [
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
        print(f"\n{model_name}")
        print(f"Mean product MAE: {results[mae_col].mean():.2f}")
        print(f"Mean product RMSE: {results[rmse_col].mean():.2f}")
        print(f"Mean product WAPE: {results[wape_col].mean():.2f}%")

    print("\n=== BEST 10 PRODUCTS BY MOVING-AVERAGE MAE ===")
    print(
        results.sort_values("moving_average_mae")
        .head(10)[
            [
                "product_id",
                "continuous_months",
                "moving_average_mae",
                "moving_average_wape_pct",
                "test_ends_at_last_observation",
            ]
        ]
        .to_string(index=False)
    )

    print(f"\nSaved model comparison: {OUTPUT_DIR / 'multi_product_model_comparison.csv'}")
    print(f"Saved predictions: {OUTPUT_DIR / 'multi_product_predictions.csv'}")


if __name__ == "__main__":
    main()
