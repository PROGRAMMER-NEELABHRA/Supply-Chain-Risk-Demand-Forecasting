
"""Evaluate baseline demand forecasts across multiple monthly test windows."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

INPUT_PATH = DATA_DIR / "forecasting_dataset.csv"
OUTPUT_PATH = DATA_DIR / "rolling_backtest_metrics.csv"

INITIAL_TRAIN_MONTHS = 24
TEST_MONTHS = 9
SEASONAL_LAG_MONTHS = 12
MOVING_AVERAGE_WINDOW = 3


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the monthly product demand dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)

    required = {"month", "product_id", "demand_units"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data["demand_units"] = pd.to_numeric(
        data["demand_units"], errors="raise"
    )

    if data.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    if data[["product_id", "month", "demand_units"]].isna().any().any():
        raise ValueError("Dataset contains missing required values.")

    if (data["demand_units"] < 0).any():
        raise ValueError("Demand cannot be negative.")

    return data.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def create_rolling_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """Generate one-month-ahead predictions over a chronological test window."""
    months = sorted(data["month"].unique())

    if len(months) <= INITIAL_TRAIN_MONTHS:
        raise ValueError("Not enough months for the initial training period.")

    test_periods = months[
        INITIAL_TRAIN_MONTHS:
        INITIAL_TRAIN_MONTHS + TEST_MONTHS
    ]

    work = data.copy()
    grouped = work.groupby("product_id")["demand_units"]

    work["naive_previous_month"] = grouped.shift(1)
    work["seasonal_naive_previous_year"] = grouped.shift(
        SEASONAL_LAG_MONTHS
    )
    work["moving_average_3_month"] = grouped.transform(
        lambda series: series.shift(1).rolling(
            window=MOVING_AVERAGE_WINDOW,
            min_periods=MOVING_AVERAGE_WINDOW,
        ).mean()
    )

    model_columns = [
        "naive_previous_month",
        "seasonal_naive_previous_year",
        "moving_average_3_month",
    ]

    test_data = work[work["month"].isin(test_periods)]
    frames = []

    for model in model_columns:
        eligible = test_data.dropna(subset=[model]).copy()
        eligible["model"] = model
        eligible["forecast"] = eligible[model]
        eligible["month"] = eligible["month"].astype(str)

        frames.append(
            eligible[
                [
                    "model",
                    "month",
                    "product_id",
                    "demand_units",
                    "forecast",
                ]
            ].rename(columns={"demand_units": "actual"})
        )

    predictions = pd.concat(frames, ignore_index=True)

    if predictions.empty:
        raise ValueError("No predictions were generated.")

    return predictions


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall and monthly forecast metrics for each model."""
    results = []

    for (model, month), group in predictions.groupby(["model", "month"]):
        errors = group["actual"] - group["forecast"]
        total_actual = group["actual"].sum()

        results.append(
            {
                "model": model,
                "month": month,
                "records": len(group),
                "actual_units": total_actual,
                "MAE": errors.abs().mean(),
                "RMSE": errors.pow(2).mean() ** 0.5,
                "WAPE_percent": (
                    errors.abs().sum() / total_actual * 100
                    if total_actual > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    """Run the rolling backtest and save monthly results."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(INPUT_PATH)
    predictions = create_rolling_predictions(data)
    metrics = calculate_metrics(predictions)

    metrics.to_csv(OUTPUT_PATH, index=False)

    print("=== ROLLING BACKTEST: MONTHLY RESULTS ===")
    print(
        metrics.sort_values(["month", "model"]).to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\n=== ROLLING BACKTEST: OVERALL SUMMARY ===")
    summary = (
        metrics.groupby("model")
        .agg(
            test_months=("month", "nunique"),
            evaluated_records=("records", "sum"),
            mean_monthly_MAE=("MAE", "mean"),
            mean_monthly_RMSE=("RMSE", "mean"),
            mean_monthly_WAPE_percent=("WAPE_percent", "mean"),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print(f"\nMetrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
