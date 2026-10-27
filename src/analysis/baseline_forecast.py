
"""Compare naive, seasonal-naive, and moving-average demand forecasts."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

INPUT_PATH = DATA_DIR / "forecasting_dataset.csv"
PREDICTIONS_PATH = DATA_DIR / "baseline_predictions.csv"
METRICS_PATH = DATA_DIR / "baseline_metrics.csv"

TEST_MONTHS = 6
SEASONAL_LAG_MONTHS = 12
MOVING_AVERAGE_WINDOW = 3


def load_forecasting_data(path: Path) -> pd.DataFrame:
    """Load and validate product-month demand records."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Forecasting dataset not found: {path}"
        )

    data = pd.read_csv(path)

    required_columns = {"month", "product_id", "demand_units"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data["demand_units"] = pd.to_numeric(
        data["demand_units"], errors="raise"
    )

    if data["product_id"].isna().any():
        raise ValueError("Product IDs contain missing values.")

    if data["demand_units"].isna().any():
        raise ValueError("Demand values contain missing values.")

    if data.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    if (data["demand_units"] < 0).any():
        raise ValueError("Demand values cannot be negative.")

    # The forecasting dataset should contain a continuous monthly
    # series for each product. Missing months must be investigated
    # rather than silently treated as zero.
    for product_id, group in data.groupby("product_id"):
        product_months = sorted(group["month"])
        expected_months = pd.period_range(
            start=product_months[0],
            end=product_months[-1],
            freq="M",
        )

        if list(product_months) != list(expected_months):
            raise ValueError(
                f"Missing monthly observations for product {product_id}."
            )

    return data.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def create_predictions(
    data: pd.DataFrame,
    test_months: int,
) -> pd.DataFrame:
    """Create forecasts for the chronological test period."""
    months = sorted(data["month"].unique())

    if test_months < 1 or test_months >= len(months):
        raise ValueError(
            "test_months must be at least 1 and less than total months."
        )

    if len(months) <= SEASONAL_LAG_MONTHS:
        raise ValueError(
            "Insufficient history for seasonal-naive forecasting."
        )

    test_periods = set(months[-test_months:])
    work = data.copy()

    grouped_demand = work.groupby("product_id")["demand_units"]

    # Forecast using the previous month's actual demand.
    work["naive_forecast"] = grouped_demand.shift(1)

    # Forecast using demand from the same month one year earlier.
    work["seasonal_forecast"] = grouped_demand.shift(
        SEASONAL_LAG_MONTHS
    )

    # Forecast using the average of the previous three months.
    work["moving_average_forecast"] = (
        grouped_demand.transform(
            lambda series: series.shift(1).rolling(
                window=MOVING_AVERAGE_WINDOW,
                min_periods=MOVING_AVERAGE_WINDOW,
            ).mean()
        )
    )

    test_data = work.loc[
        work["month"].isin(test_periods)
    ].copy()

    forecast_methods = [
        ("naive_previous_month", "naive_forecast"),
        ("seasonal_naive_previous_year", "seasonal_forecast"),
        ("moving_average_3_month", "moving_average_forecast"),
    ]

    prediction_frames = []

    for model_name, forecast_column in forecast_methods:
        model_data = test_data.dropna(
            subset=[forecast_column]
        ).copy()

        model_data["model"] = model_name
        model_data["forecast"] = model_data[forecast_column]
        model_data["month"] = model_data["month"].astype(str)

        prediction_frames.append(
            model_data[
                [
                    "model",
                    "month",
                    "product_id",
                    "demand_units",
                    "forecast",
                ]
            ].rename(
                columns={"demand_units": "actual"}
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if predictions.empty:
        raise ValueError("No forecasts could be generated.")

    return predictions


def calculate_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate MAE, RMSE, and WAPE for each forecasting method."""
    results = []

    for model_name, group in predictions.groupby("model"):
        errors = group["actual"] - group["forecast"]
        total_actual = group["actual"].sum()

        wape = (
            errors.abs().sum() / total_actual * 100
            if total_actual > 0
            else float("nan")
        )

        results.append(
            {
                "model": model_name,
                "test_months": group["month"].nunique(),
                "evaluated_records": len(group),
                "MAE": errors.abs().mean(),
                "RMSE": errors.pow(2).mean() ** 0.5,
                "WAPE_percent": wape,
            }
        )

    return pd.DataFrame(results).sort_values(
        "WAPE_percent"
    ).reset_index(drop=True)


def main() -> None:
    """Run all baselines, evaluate them, and save results."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = load_forecasting_data(INPUT_PATH)

    predictions = create_predictions(
        data=data,
        test_months=TEST_MONTHS,
    )

    metrics = calculate_metrics(predictions)

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    test_months = sorted(predictions["month"].unique())

    print("=== BASELINE MODEL COMPARISON ===")
    print(
        f"Test period: {test_months[0]} to {test_months[-1]}"
    )
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
