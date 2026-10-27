
"""Train and evaluate a feature-enhanced Random Forest demand model."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

DEMAND_PATH = DATA_DIR / "forecasting_dataset.csv"
PRODUCT_PATH = DATA_DIR / "products.csv"
PREDICTIONS_PATH = DATA_DIR / "advanced_v2_predictions.csv"
METRICS_PATH = DATA_DIR / "advanced_v2_metrics.csv"

TEST_MONTHS = 6
RANDOM_STATE = 42
N_ESTIMATORS = 200

LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6)

CATEGORICAL_FEATURES = [
    "product_id",
    "product_category_id",
    "department_id",
    "month_of_year",
]

NUMERIC_FEATURES = [
    *(f"lag_{lag}" for lag in LAGS),
    *(f"rolling_mean_{window}" for window in ROLLING_WINDOWS),
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate demand and product data."""
    for path in (DEMAND_PATH, PRODUCT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    demand = pd.read_csv(DEMAND_PATH)
    products = pd.read_csv(PRODUCT_PATH)

    required_demand = {"month", "product_id", "demand_units"}
    missing_demand = required_demand - set(demand.columns)
    if missing_demand:
        raise ValueError(
            f"Demand data is missing columns: {sorted(missing_demand)}"
        )

    required_products = {
        "product_id",
        "product_category_id",
        "department_id",
    }
    missing_products = required_products - set(products.columns)
    if missing_products:
        raise ValueError(
            f"Product data is missing columns: {sorted(missing_products)}"
        )

    demand["month"] = pd.PeriodIndex(demand["month"], freq="M")
    demand["demand_units"] = pd.to_numeric(
        demand["demand_units"], errors="raise"
    )

    if demand[["month", "product_id", "demand_units"]].isna().any().any():
        raise ValueError("Demand data contains missing required values.")

    if demand.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month demand records found.")

    if (demand["demand_units"] < 0).any():
        raise ValueError("Demand values cannot be negative.")

    if products["product_id"].duplicated().any():
        raise ValueError("Product master contains duplicate product IDs.")

    return demand.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True), products


def create_features(
    demand: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical demand, calendar, and product features."""
    data = demand.copy()
    grouped = data.groupby("product_id", sort=False)["demand_units"]

    for lag in LAGS:
        data[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        data[f"rolling_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(
                window=window,
                min_periods=window,
            ).mean()
        )

    data["month_of_year"] = data["month"].dt.month.astype(str)

    product_features = products[
        ["product_id", "product_category_id", "department_id"]
    ].copy()

    data = data.merge(
        product_features,
        on="product_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if (data["_merge"] != "both").any():
        missing_ids = data.loc[
            data["_merge"] != "both", "product_id"
        ].unique().tolist()
        raise ValueError(
            f"Products missing from product master: {missing_ids[:10]}"
        )

    data = data.drop(columns="_merge")

    if data[CATEGORICAL_FEATURES].isna().any().any():
        raise ValueError("Categorical features contain missing values.")

    return data


def calculate_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """Calculate MAE, RMSE, and WAPE."""
    errors = actual - forecast
    total_actual = actual.sum()

    return {
        "evaluated_records": len(actual),
        "MAE": mean_absolute_error(actual, forecast),
        "RMSE": mean_squared_error(actual, forecast) ** 0.5,
        "WAPE_percent": (
            errors.abs().sum() / total_actual * 100
            if total_actual > 0
            else float("nan")
        ),
    }


def main() -> None:
    """Train, evaluate, and save the enhanced forecasting model."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    demand, products = load_data()
    data = create_features(demand, products)

    months = sorted(data["month"].unique())
    if len(months) <= TEST_MONTHS:
        raise ValueError("Not enough months for the test period.")

    test_periods = months[-TEST_MONTHS:]
    test_start = test_periods[0]

    # Fit on earlier months only. Test features use past actual demand,
    # appropriate for a rolling one-month-ahead forecasting process.
    model_data = data.dropna(subset=FEATURE_COLUMNS).copy()
    train = model_data[model_data["month"] < test_start].copy()
    test = model_data[model_data["month"].isin(test_periods)].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test data is empty.")

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", transformer),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(train[FEATURE_COLUMNS], train["demand_units"])

    forecasts = model.predict(test[FEATURE_COLUMNS]).clip(min=0)

    predictions = pd.DataFrame(
        {
            "model": "random_forest_with_product_calendar",
            "month": test["month"].astype(str).to_numpy(),
            "product_id": test["product_id"].to_numpy(),
            "actual": test["demand_units"].to_numpy(),
            "forecast": forecasts,
        }
    )

    metrics = calculate_metrics(
        predictions["actual"],
        predictions["forecast"],
    )
    metrics.update(
        {
            "model": "random_forest_with_product_calendar",
            "test_start": str(test_start),
            "test_end": str(test_periods[-1]),
            "training_end": str(train["month"].max()),
        }
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    print("=== ENHANCED RANDOM FOREST RESULTS ===")
    print(f"Training through: {train['month'].max()}")
    print(f"Test period: {test_start} to {test_periods[-1]}")
    print(
        pd.DataFrame([metrics]).to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
