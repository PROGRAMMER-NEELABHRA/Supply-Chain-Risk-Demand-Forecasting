
"""Evaluate baseline forecast errors by month and product."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

INPUT_PATH = DATA_DIR / "baseline_predictions.csv"
MONTHLY_OUTPUT_PATH = DATA_DIR / "baseline_monthly_metrics.csv"
PRODUCT_OUTPUT_PATH = DATA_DIR / "baseline_product_metrics.csv"


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions and validate required fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    data = pd.read_csv(path)

    required = {"model", "month", "product_id", "actual", "forecast"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for column in ("actual", "forecast"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[list(required)].isna().any().any():
        raise ValueError("Prediction data contains missing values.")

    if (data["actual"] < 0).any() or (data["forecast"] < 0).any():
        raise ValueError("Actual demand and forecasts must be non-negative.")

    return data


def summarize_errors(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Calculate error metrics for each model and group."""
    rows = []

    for keys, group in data.groupby(["model", group_column]):
        model_name, group_name = keys
        errors = group["actual"] - group["forecast"]
        actual_total = group["actual"].sum()

        rows.append(
            {
                "model": model_name,
                group_column: group_name,
                "records": len(group),
                "actual_units": group["actual"].sum(),
                "forecast_units": group["forecast"].sum(),
                "MAE": errors.abs().mean(),
                "RMSE": (errors.pow(2).mean()) ** 0.5,
                "WAPE_percent": (
                    errors.abs().sum() / actual_total * 100
                    if actual_total > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Save monthly and product-level evaluation summaries."""
    predictions = load_predictions(INPUT_PATH)

    monthly = summarize_errors(predictions, "month")
    products = summarize_errors(predictions, "product_id")

    monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
    products.to_csv(PRODUCT_OUTPUT_PATH, index=False)

    print("=== MONTHLY BASELINE PERFORMANCE ===")
    print(
        monthly.sort_values(["month", "model"]).to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )

    print("\n=== PRODUCTS WITH HIGHEST BASELINE MAE ===")
    for model_name, group in products.groupby("model"):
        print(f"\nModel: {model_name}")
        print(
            group.sort_values("MAE", ascending=False)
            .head(10)
            .to_string(index=False, float_format=lambda value: f"{value:.2f}")
        )

    print(f"\nMonthly report saved to: {MONTHLY_OUTPUT_PATH}")
    print(f"Product report saved to: {PRODUCT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
