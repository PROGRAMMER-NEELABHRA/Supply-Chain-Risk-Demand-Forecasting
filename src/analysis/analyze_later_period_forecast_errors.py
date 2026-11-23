
"""Analyze later-period forecast errors and systematic model bias."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
    / "later_period_model_predictions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"


def calculate_metrics(group: pd.DataFrame, column: str) -> dict:
    """Calculate forecast metrics for one product and model."""
    actual = group["actual"].astype(float)
    predicted = group[column].astype(float)
    errors = predicted - actual

    total_actual = actual.sum()
    wape = (
        errors.abs().sum() / abs(total_actual) * 100
        if total_actual != 0
        else float("nan")
    )

    return {
        "mae": errors.abs().mean(),
        "rmse": (errors.pow(2).mean()) ** 0.5,
        "bias": errors.mean(),
        "wape_pct": wape,
        "mean_actual_demand": actual.mean(),
        "mean_predicted_demand": predicted.mean(),
        "mean_actual_minus_predicted": (actual - predicted).mean(),
    }


def main() -> None:
    """Create product-level error analysis for both Random Forest models."""
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE, parse_dates=["month"])

    required = {
        "product_id",
        "month",
        "actual",
        "current_rf_forecast",
        "shallow_rf_forecast",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError("Prediction data contains missing values.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month predictions found.")

    model_columns = {
        "Current RF depth 10": "current_rf_forecast",
        "Shallow RF depth 5": "shallow_rf_forecast",
    }

    rows = []

    for product_id, product_group in df.groupby("product_id"):
        for model_name, prediction_column in model_columns.items():
            metrics = calculate_metrics(
                product_group.sort_values("month"),
                prediction_column,
            )
            rows.append(
                {
                    "product_id": product_id,
                    "model": model_name,
                    **metrics,
                }
            )

    per_product = pd.DataFrame(rows)

    wide = per_product.pivot(
        index="product_id",
        columns="model",
        values=["mae", "rmse", "bias", "wape_pct"],
    )
    wide.columns = [
        f"{metric}_{model.lower().replace(' ', '_')}"
        for metric, model in wide.columns
    ]
    wide = wide.reset_index()

    wide["mae_improvement_depth5"] = (
        wide["mae_current_rf_depth_10"]
        - wide["mae_shallow_rf_depth_5"]
    )

    wide["better_by_mae"] = wide["mae_improvement_depth5"].map(
        lambda value: (
            "Depth 5"
            if value > 0
            else "Depth 10"
            if value < 0
            else "Tie"
        )
    )

    # Positive prediction bias means overforecasting; negative means underforecasting.
    wide["depth10_bias_direction"] = wide[
        "bias_current_rf_depth_10"
    ].map(
        lambda value: (
            "Overforecast"
            if value > 0
            else "Underforecast"
            if value < 0
            else "No average bias"
        )
    )

    wide = wide.sort_values("mae_current_rf_depth_10", ascending=False)

    summary_rows = []

    for model_name, prediction_column in model_columns.items():
        actual = df["actual"].astype(float)
        predicted = df[prediction_column].astype(float)
        errors = predicted - actual

        summary_rows.append(
            {
                "model": model_name,
                "overall_mae": errors.abs().mean(),
                "overall_rmse": (errors.pow(2).mean()) ** 0.5,
                "overall_wape_pct": (
                    errors.abs().sum() / actual.abs().sum() * 100
                    if actual.abs().sum() != 0
                    else float("nan")
                ),
                "overall_bias": errors.mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_product_file = OUTPUT_DIR / "later_period_error_by_product.csv"
    summary_file = OUTPUT_DIR / "later_period_error_model_summary.csv"

    wide.to_csv(per_product_file, index=False)
    summary.to_csv(summary_file, index=False)

    print("\n=== LATER-PERIOD OVERALL ERROR SUMMARY ===")
    print(summary.round(2).to_string(index=False))

    print("\n=== PRODUCTS WITH LARGEST DEPTH-10 MAE ===")
    print(
        wide[
            [
                "product_id",
                "mae_current_rf_depth_10",
                "bias_current_rf_depth_10",
                "mae_shallow_rf_depth_5",
                "mae_improvement_depth5",
                "better_by_mae",
                "depth10_bias_direction",
            ]
        ].head(10).round(2).to_string(index=False)
    )

    print("\n=== MODEL WINS BY PRODUCT ===")
    print(wide["better_by_mae"].value_counts().to_string())

    print(f"\nSaved product errors: {per_product_file}")
    print(f"Saved model summary: {summary_file}")


if __name__ == "__main__":
    main()
