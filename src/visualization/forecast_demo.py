"""Create reproducible forecast and model-comparison demonstration charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.data.validators import validate_forecast_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "quality_review"
    / "monthly_demand_observed_rebuilt.csv"
)
FORECAST_DIR = PROJECT_ROOT / "outputs" / "forecasts"


def create_forecast_visualization(
    monthly: pd.DataFrame,
    forecasts: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot observed history and examples from each available forecast tier."""
    validate_forecast_output(forecasts)
    method_styles = {
        "random_forest_recursive": ("tab:blue", "Random Forest"),
        "moving_average_baseline": ("tab:orange", "Moving Average baseline"),
        "naive_last_observation_baseline": ("tab:green", "Naive baseline"),
    }
    selected_products = []
    for method in method_styles:
        product_totals = (
            forecasts.loc[forecasts["forecast_method"].eq(method)]
            .groupby("product_id")["forecast_demand"]
            .sum()
            .nlargest(2)
        )
        selected_products.extend(product_totals.index.tolist())
    product_count = len(selected_products)
    if product_count == 0:
        raise ValueError("No products are available for the forecast chart.")

    rows = (product_count + 1) // 2
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(14, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for axis, product_id in zip(axes.flat, selected_products):
        history = monthly.loc[
            monthly["product_id"].eq(product_id)
        ].sort_values("month").tail(12)
        future = forecasts.loc[
            forecasts["product_id"].eq(product_id)
        ].sort_values("forecast_month")

        method = future["forecast_method"].iloc[0]
        confidence = future["forecast_confidence"].iloc[0]
        color, method_label = method_styles.get(
            method, ("tab:gray", method)
        )
        axis.scatter(
            history["month"],
            history["demand_units"],
            color="black",
            marker="o",
            label="Observed actual records",
        )
        axis.plot(
            future["forecast_month"],
            future["forecast_demand"],
            marker="o",
            linestyle="--",
            color=color,
            label=f"{method_label} ({confidence.lower()} confidence)",
        )
        axis.axvline(
            future["forecast_origin_month"].iloc[0],
            color="gray",
            linestyle=":",
            label="Forecast origin",
        )
        axis.set_title(
            f"Product {product_id} — {method_label}\n"
            f"{future['data_quality_status'].iloc[0]}"
        )
        axis.set_ylabel("Demand units")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=35)
        axis.legend()

    for axis in axes.flat[product_count:]:
        axis.set_visible(False)

    figure.suptitle(
        "Tiered three-month forecasts: Random Forest and baseline estimates\n"
        "Only products with actual observations are shown; unknown months are not filled"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def create_model_comparison_visualization(
    comparison: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot pooled MAE, RMSE, and WAPE from the recursive backtest."""
    required = {"model", "mae", "rmse", "wape_pct"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(
            f"Model comparison is missing columns: {sorted(missing)}"
        )
    if comparison.empty:
        raise ValueError("Model comparison data cannot be empty.")

    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis, metric, label in zip(
        axes,
        ["mae", "rmse", "wape_pct"],
        ["MAE (demand units)", "RMSE (demand units)", "WAPE (%)"],
    ):
        axis.bar(comparison["model"], comparison[metric])
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=25)

    figure.suptitle(
        "Recursive three-month backtest\n"
        "Historical pre-regime-change period; not evidence for late-2017"
    )
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def main() -> None:
    """Load pipeline outputs and write project-demonstration charts."""
    monthly = pd.read_csv(DATA_FILE, parse_dates=["month"])
    forecasts = pd.read_csv(
        FORECAST_DIR / "future_demand_forecasts.csv",
        parse_dates=["forecast_month", "forecast_origin_month"],
    )
    comparison = pd.read_csv(
        FORECAST_DIR / "random_forest_model_comparison.csv"
    )

    forecast_chart = FORECAST_DIR / "forecast_demonstration.png"
    comparison_chart = FORECAST_DIR / "model_comparison.png"
    create_forecast_visualization(monthly, forecasts, forecast_chart)
    create_model_comparison_visualization(comparison, comparison_chart)
    print(f"Saved forecast chart: {forecast_chart}")
    print(f"Saved model comparison chart: {comparison_chart}")


if __name__ == "__main__":
    main()
