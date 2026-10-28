
"""Analyze forecast errors consistently across evaluation periods."""

from pathlib import Path

import pandas as pd

from src.forecasting.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

FILES = {
    "First period": OUTPUT_DIR / "random_forest_predictions.csv",
    "Later period": OUTPUT_DIR / "later_period_predictions.csv",
}


def analyze_period(path: Path, period_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate per-product and summary metrics from one prediction file."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path, parse_dates=["month"])

    required = {
        "product_id",
        "month",
        "actual",
        "naive_forecast",
        "moving_average_forecast",
        "random_forest_forecast",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month predictions found.")

    models = {
        "Naive": "naive_forecast",
        "Moving Average": "moving_average_forecast",
        "Random Forest": "random_forest_forecast",
    }

    rows = []

    for product_id, group in df.groupby("product_id"):
        actual = group["actual"]

        for model_name, prediction_column in models.items():
            predicted = group[prediction_column]

            rows.append(
                {
                    "period": period_name,
                    "product_id": product_id,
                    "model": model_name,
                    "mae": mean_absolute_error(actual, predicted),
                    "rmse": root_mean_squared_error(actual, predicted),
                    "wape_pct": weighted_absolute_percentage_error(
                        actual, predicted
                    ),
                }
            )

    product_metrics = pd.DataFrame(rows)

    summary = (
        product_metrics.groupby(["period", "model"])[
            ["mae", "rmse", "wape_pct"]
        ]
        .mean()
        .reset_index()
    )

    return product_metrics, summary


def main() -> None:
    """Report and save consistent model evaluation metrics."""
    all_product_metrics = []
    all_summaries = []

    for period_name, path in FILES.items():
        product_metrics, summary = analyze_period(path, period_name)
        all_product_metrics.append(product_metrics)
        all_summaries.append(summary)

    product_metrics = pd.concat(all_product_metrics, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)

    print("\n=== MEAN PER-PRODUCT METRICS BY PERIOD ===")
    print(
        summary.pivot(
            index=["period", "model"],
            columns=[],
            values=["mae", "rmse", "wape_pct"],
        ).round(2).to_string()
    )

    print("\n=== PRODUCT-LEVEL RANDOM FOREST MAE WINS ===")

    for period_name, group in product_metrics.groupby("period"):
        mae_table = group.pivot(
            index="product_id",
            columns="model",
            values="mae",
        )

        print(f"\n{period_name}")
        print(f"Products evaluated: {len(mae_table)}")
        print(
            "Random Forest wins vs moving average:",
            int(
                (
                    mae_table["Random Forest"]
                    < mae_table["Moving Average"]
                ).sum()
            ),
        )
        print(
            "Random Forest wins vs naive:",
            int(
                (
                    mae_table["Random Forest"]
                    < mae_table["Naive"]
                ).sum()
            ),
        )

    product_metrics.to_csv(
        OUTPUT_DIR / "forecast_error_by_product.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "forecast_error_summary.csv",
        index=False,
    )

    print("\nUpdated error-analysis files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
