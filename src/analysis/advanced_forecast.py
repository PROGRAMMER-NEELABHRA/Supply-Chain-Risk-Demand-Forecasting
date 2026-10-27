
"""Train and evaluate a Random Forest demand forecasting model."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

INPUT_PATH = DATA_DIR / "forecasting_dataset.csv"
PREDICTIONS_PATH = DATA_DIR / "advanced_predictions.csv"
METRICS_PATH = DATA_DIR / "advanced_metrics.csv"

TEST_MONTHS = 6
RANDOM_STATE = 42
N_ESTIMATORS = 200

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate product-month demand data."""
    if not path.is_file():
        raise FileNotFoundError(f"Forecasting dataset not found: {path}")

    data = pd.read_csv(path)

    required = {"month", "product_id", "demand_units"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data["demand_units"] = pd.to_numeric(
        data["demand_units"], errors="raise"
    )

    if data[["month", "product_id", "demand_units"]].isna().any().any():
        raise ValueError("Required data contains missing values.")

    if data.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    if (data["demand_units"] < 0).any():
        raise ValueError("Demand cannot be negative.")

    return data.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling features using past demand only."""
    result = data.copy()
    grouped = result.groupby("product_id", sort=False)["demand_units"]

    for lag in (1, 2, 3, 6, 12):
        result[f"lag_{lag}"] = grouped.shift(lag)

    result["rolling_mean_3"] = grouped.transform(
        lambda series: series.shift(1).rolling(3).mean()
    )
    result["rolling_mean_6"] = grouped.transform(
        lambda series: series.shift(1).rolling(6).mean()
    )

    return result


def calculate_metrics(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict:
    """Calculate MAE, RMSE and WAPE."""
    errors = actual - forecast
    actual_total = actual.sum()

    return {
        "evaluated_records": len(actual),
        "MAE": mean_absolute_error(actual, forecast),
        "RMSE": mean_squared_error(actual, forecast) ** 0.5,
        "WAPE_percent": (
            errors.abs().sum() / actual_total * 100
            if actual_total > 0
            else float("nan")
        ),
    }


def main() -> None:
    """Train the model, evaluate it chronologically and save outputs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(INPUT_PATH)
    featured = create_features(data)

    months = sorted(featured["month"].unique())
    if len(months) <= TEST_MONTHS:
        raise ValueError("Not enough months for the requested test period.")

    test_periods = months[-TEST_MONTHS:]
    test_start = test_periods[0]

    # Use only rows with all required historical features.
    model_data = featured.dropna(subset=FEATURE_COLUMNS).copy()

    train = model_data[model_data["month"] < test_start].copy()
    test = model_data[model_data["month"].isin(test_periods)].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test dataset is empty.")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(train[FEATURE_COLUMNS], train["demand_units"])
    test["forecast"] = model.predict(test[FEATURE_COLUMNS]).clip(min=0)

    predictions = test[
        ["month", "product_id", "demand_units", "forecast"]
    ].rename(columns={"demand_units": "actual"})

    predictions["month"] = predictions["month"].astype(str)
    predictions["model"] = "random_forest"

    metrics = calculate_metrics(
        predictions["actual"],
        predictions["forecast"],
    )
    metrics["model"] = "random_forest"
    metrics["test_start"] = str(test_start)
    metrics["test_end"] = str(test_periods[-1])

    predictions = predictions[
        ["model", "month", "product_id", "actual", "forecast"]
    ]
    metrics_frame = pd.DataFrame([metrics])

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    metrics_frame.to_csv(METRICS_PATH, index=False)

    print("=== RANDOM FOREST FORECAST RESULTS ===")
    print(f"Training through: {str(train['month'].max())}")
    print(f"Test period: {test_start} to {test_periods[-1]}")
    print(
        metrics_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
