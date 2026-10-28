
"""Compare Random Forest configurations on a fixed chronological split."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
TRAIN_END = pd.Timestamp("2016-10-01")
TEST_START = pd.Timestamp("2016-11-01")
TEST_END = pd.Timestamp("2017-04-01")

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

CONFIGURATIONS = [
    {
        "name": "Current baseline",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "Shallower trees",
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "Deeper trees",
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "name": "Larger leaf size",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 3,
        "max_features": 1.0,
    },
    {
        "name": "More trees",
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
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
        raise ValueError("Required data contains missing values.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month observations found.")

    if (df["demand_units"] < 0).any():
        raise ValueError("Negative demand found.")

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def select_complete_products(df: pd.DataFrame) -> pd.DataFrame:
    """Select products with complete monthly history through the test end."""
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
        raise ValueError("No products have complete history for this period.")

    return df.loc[df["product_id"].isin(eligible_ids)].copy()


def main() -> None:
    """Evaluate fixed Random Forest configurations on the same test rows."""
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

    if train[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Training features contain missing values.")

    if test[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Test features contain missing values.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["demand_units"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["demand_units"]

    results = []
    predictions = test[
        ["product_id", "month", "demand_units"]
    ].rename(columns={"demand_units": "actual"}).reset_index(drop=True)

    print("\n=== RANDOM FOREST CONFIGURATION EXPERIMENT ===")
    print(f"Eligible products: {eligible['product_id'].nunique()}")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(f"Training ends: {TRAIN_END:%Y-%m}")
    print(f"Testing: {TEST_START:%Y-%m} to {TEST_END:%Y-%m}")

    for config in CONFIGURATIONS:
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        mae = mean_absolute_error(y_test, predicted)
        rmse = root_mean_squared_error(y_test, predicted)
        wape = weighted_absolute_percentage_error(y_test, predicted)

        results.append(
            {
                **config,
                "mae": mae,
                "rmse": rmse,
                "wape_pct": wape,
            }
        )

        prediction_column = (
            "prediction_"
            + config["name"].lower().replace(" ", "_")
        )
        predictions[prediction_column] = predicted

        print(f"\n{config['name']}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"WAPE: {wape:.2f}%")

    comparison = pd.DataFrame(results).sort_values("mae")

    print("\n=== CONFIGURATIONS RANKED BY MAE ===")
    print(
        comparison[
            [
                "name",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "mae",
                "rmse",
                "wape_pct",
            ]
        ].round(2).to_string(index=False)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        OUTPUT_DIR / "random_forest_tuning_comparison.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT_DIR / "random_forest_tuning_predictions.csv",
        index=False,
    )

    print(f"\nSaved comparison to: {OUTPUT_DIR / 'random_forest_tuning_comparison.csv'}")
    print(f"Saved predictions to: {OUTPUT_DIR / 'random_forest_tuning_predictions.csv'}")


if __name__ == "__main__":
    main()
