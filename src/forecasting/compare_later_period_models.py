
"""Compare current and shallower Random Forest on a later test period."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.forecasting.baseline import (
    naive_forecast,
    moving_average_forecast,
)
from src.forecasting.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.preprocessing.feature_engineering import create_demand_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

START_MONTH = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2017-04-01")
TEST_START = pd.Timestamp("2017-05-01")
TEST_END = pd.Timestamp("2017-10-01")

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "month_number",
    "quarter",
]


def load_data() -> pd.DataFrame:
    """Load and validate the monthly demand dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["month"])

    required = {"month", "product_id", "demand_units"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[
        df["month"].between(START_MONTH, TEST_END),
        ["month", "product_id", "demand_units"],
    ].copy()

    df["demand_units"] = pd.to_numeric(
        df["demand_units"], errors="raise"
    )

    if df[list(required)].isna().any().any():
        raise ValueError("Required fields contain missing values.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    if (df["demand_units"] < 0).any():
        raise ValueError("Negative demand found.")

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def select_complete_products(df: pd.DataFrame) -> pd.DataFrame:
    """Select products with complete monthly history through October 2017."""
    expected_months = pd.date_range(
        START_MONTH,
        TEST_END,
        freq="MS",
    )

    eligible_ids = []

    for product_id, group in df.groupby("product_id"):
        observed = (
            group["month"]
            .sort_values()
            .reset_index(drop=True)
        )

        if observed.equals(pd.Series(expected_months, name="month")):
            eligible_ids.append(product_id)

    if not eligible_ids:
        raise ValueError(
            "No products have complete history through October 2017."
        )

    return df.loc[df["product_id"].isin(eligible_ids)].copy()


def main() -> None:
    """Compare both Random Forest configurations and baseline models."""
    df = load_data()
    eligible = select_complete_products(df)

    features = create_demand_features(eligible)

    train = features.loc[features["month"] <= TRAIN_END].copy()
    test = features.loc[
        features["month"].between(TEST_START, TEST_END)
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test data is empty.")

    missing_features = set(FEATURE_COLUMNS) - set(features.columns)
    if missing_features:
        raise ValueError(
            f"Missing engineered features: {sorted(missing_features)}"
        )

    if (
        train[FEATURE_COLUMNS].isna().any().any()
        or test[FEATURE_COLUMNS].isna().any().any()
    ):
        raise ValueError("Feature data contains missing values.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["demand_units"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["demand_units"]

    configurations = {
        "Current RF depth 10": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 1,
        },
        "Shallow RF depth 5": {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_leaf": 1,
        },
    }

    results = test[
        ["product_id", "month", "demand_units"]
    ].rename(columns={"demand_units": "actual"}).reset_index(drop=True)

    predictions = {}

    for name, settings in configurations.items():
        model = RandomForestRegressor(
            **settings,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    # Calculate one-step-ahead baselines from each product's prior history.
    baseline_rows = []

    for product_id, test_group in test.groupby("product_id"):
        history_df = (
            eligible.loc[eligible["product_id"] == product_id]
            .sort_values("month")
        )

        for _, test_row in test_group.sort_values("month").iterrows():
            history = history_df.loc[
                history_df["month"] < test_row["month"],
                "demand_units",
            ]

            baseline_rows.append(
                {
                    "product_id": product_id,
                    "month": test_row["month"],
                    "naive_forecast": float(
                        naive_forecast(history).iloc[-1]
                    ),
                    "moving_average_forecast": float(
                        moving_average_forecast(history, window=3).iloc[-1]
                    ),
                }
            )

    baseline_df = pd.DataFrame(baseline_rows)

    results = results.merge(
        baseline_df,
        on=["product_id", "month"],
        how="left",
        validate="one_to_one",
    )

    results["current_rf_forecast"] = predictions["Current RF depth 10"]
    results["shallow_rf_forecast"] = predictions["Shallow RF depth 5"]

    model_columns = {
        "Naive": "naive_forecast",
        "Moving Average": "moving_average_forecast",
        "Current RF depth 10": "current_rf_forecast",
        "Shallow RF depth 5": "shallow_rf_forecast",
    }

    # Calculate metrics per product, then average across products.
    metric_rows = []

    for product_id, group in results.groupby("product_id"):
        for model_name, column in model_columns.items():
            actual = group["actual"]
            predicted = group[column]

            metric_rows.append(
                {
                    "product_id": product_id,
                    "model": model_name,
                    "mae": mean_absolute_error(actual, predicted),
                    "rmse": root_mean_squared_error(actual, predicted),
                    "wape_pct": weighted_absolute_percentage_error(
                        actual, predicted
                    ),
                }
            )

    per_product = pd.DataFrame(metric_rows)
    summary = (
        per_product.groupby("model")[["mae", "rmse", "wape_pct"]]
        .mean()
        .reindex(list(model_columns))
    )

    print("\n=== LATER-PERIOD MODEL COMPARISON ===")
    print(f"Eligible products: {eligible['product_id'].nunique()}")
    print(f"Training: {START_MONTH:%Y-%m} to {TRAIN_END:%Y-%m}")
    print(f"Testing: {TEST_START:%Y-%m} to {TEST_END:%Y-%m}")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")

    print("\n=== MEAN METRICS PER PRODUCT ===")
    print(summary.round(2).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "later_period_model_predictions.csv",
        index=False,
    )
    per_product.to_csv(
        OUTPUT_DIR / "later_period_per_product_metrics.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "later_period_model_comparison.csv",
        index_label="model",
    )

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
