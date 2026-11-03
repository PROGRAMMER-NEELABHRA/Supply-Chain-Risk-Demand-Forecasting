
"""Compare current and shallower Random Forest on a later test period."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data.loaders import load_monthly_demand
from src.forecasting.baseline import (
    naive_forecast,
    moving_average_forecast,
)
from src.forecasting.evaluation import (
    calculate_product_metrics,
    select_products_with_complete_history,
)
from src.preprocessing.feature_engineering import (
    DEFAULT_FEATURE_COLUMNS as FEATURE_COLUMNS,
    create_demand_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

START_MONTH = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2017-04-01")
TEST_START = pd.Timestamp("2017-05-01")
TEST_END = pd.Timestamp("2017-10-01")

def load_data() -> pd.DataFrame:
    """Load and validate the monthly demand dataset."""
    df = load_monthly_demand(DATA_FILE)
    df = df.loc[
        df["month"].between(START_MONTH, TEST_END),
        ["month", "product_id", "demand_units"],
    ].copy()

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def select_complete_products(df: pd.DataFrame) -> pd.DataFrame:
    """Select products with complete monthly history through October 2017."""
    eligible = select_products_with_complete_history(
        df,
        START_MONTH,
        TEST_END,
    )
    if eligible.empty:
        raise ValueError(
            "No products have complete history through October 2017."
        )

    return eligible


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

    per_product = calculate_product_metrics(results, model_columns)
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
