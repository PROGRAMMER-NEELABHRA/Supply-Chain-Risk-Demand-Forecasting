
"""Compare recursive Random Forest and baseline forecasts."""

from pathlib import Path

import pandas as pd

from src.forecasting.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.forecasting.models import (
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
    train_random_forest,
)
from src.preprocessing.feature_engineering import (
    DEFAULT_FEATURE_COLUMNS as FEATURE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "quality_review"
    / "monthly_demand_observed_rebuilt.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

START_MONTH = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2016-10-01")
TEST_START = pd.Timestamp("2016-11-01")
TEST_END = pd.Timestamp("2017-01-01")

MIN_HISTORY_MONTHS = 12

MODEL_COLUMNS = {
    "Naive": "naive_forecast",
    "Moving Average": "moving_average_forecast",
    "Random Forest": "random_forest_forecast",
}


def load_data() -> pd.DataFrame:
    """Load and validate observed monthly demand."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["month"])

    required = {"month", "product_id", "demand_units"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[["month", "product_id", "demand_units"]].copy()

    df["month"] = (
        df["month"].dt.to_period("M").dt.to_timestamp()
    )

    df["demand_units"] = pd.to_numeric(
        df["demand_units"], errors="raise"
    )

    if df[list(required)].isna().any().any():
        raise ValueError("Required data contains missing values.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    if (df["demand_units"] < 0).any():
        raise ValueError("Negative demand found.")

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)



def select_complete_products(df: pd.DataFrame) -> pd.DataFrame:
    """Select products with sufficient consecutive observed history."""

    eligible_ids = []

    for product_id, group in df.groupby("product_id"):
        group = group.sort_values("month")

        # Require at least 12 consecutive months before the test.
        training = group.loc[
            group["month"] <= TRAIN_END
        ]

        test = group.loc[
            group["month"].between(TEST_START, TEST_END)
        ]

        if len(training) < MIN_HISTORY_MONTHS:
            continue

        if len(test) != 3:
            continue

        # Verify consecutive monthly observations.
        expected_training = pd.date_range(
            START_MONTH,
            TRAIN_END,
            freq="MS",
        )

        expected_test = pd.date_range(
            TEST_START,
            TEST_END,
            freq="MS",
        )

        if not pd.DatetimeIndex(training["month"]).equals(
            expected_training
        ):
            continue

        if not pd.DatetimeIndex(test["month"]).equals(
            expected_test
        ):
            continue

        eligible_ids.append(product_id)

    if not eligible_ids:
        raise ValueError(
            "No products satisfy the historical evaluation requirements."
        )

    return df.loc[
        df["product_id"].isin(eligible_ids)
    ].copy()


def create_feature_row(
    history: list[float],
    forecast_month: pd.Timestamp,
) -> pd.DataFrame:
    """Create the same nine features used by the forecast generator."""

    return pd.DataFrame(
        [{
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_3": history[-3],
            "lag_6": history[-6],
            "lag_12": history[-12],
            "rolling_mean_3": sum(history[-3:]) / 3,
            "rolling_mean_6": sum(history[-6:]) / 6,
            "month_number": forecast_month.month,
            "quarter": forecast_month.quarter,
        }],
        columns=FEATURE_COLUMNS,
    )


def forecast_one_product(
    product: pd.DataFrame,
) -> list[dict]:
    """Train and recursively forecast one product."""

    product = product.sort_values("month")

    training_history = product.loc[
        product["month"] <= TRAIN_END,
        "demand_units",
    ].astype(float).tolist()

    test_data = product.loc[
        product["month"].between(TEST_START, TEST_END)
    ].sort_values("month")

    if len(training_history) < MIN_HISTORY_MONTHS:
        raise ValueError("Insufficient training history.")

    # Build supervised training rows using past observations only.
    feature_rows = []
    targets = []

    for index in range(MIN_HISTORY_MONTHS, len(training_history)):
        month = pd.date_range(
            product.loc[product["month"] <= TRAIN_END, "month"].min(),
            TRAIN_END,
            freq="MS",
        )[index]

        feature_rows.append(
            create_feature_row(
                training_history[:index],
                month,
            )
        )

        targets.append(training_history[index])

    if not feature_rows:
        raise ValueError("No usable training rows.")

    X_train = pd.concat(
        feature_rows,
        ignore_index=True,
    )

    y_train = pd.Series(targets)

    model = train_random_forest(X_train, y_train)

    # All three models forecast recursively.
    rf_history = training_history.copy()
    naive_history = training_history.copy()
    moving_history = training_history.copy()

    records = []

    for _, test_row in test_data.iterrows():

        month = test_row["month"]

        features = create_feature_row(
            rf_history,
            month,
        )

        rf_prediction = max(
            0.0,
            float(model.predict(features)[0]),
        )

        naive_prediction = naive_history[-1]

        moving_prediction = sum(
            moving_history[-3:]
        ) / 3

        records.append({
            "product_id": int(test_row["product_id"]),
            "month": month,
            "actual": float(test_row["demand_units"]),
            "naive_forecast": float(naive_prediction),
            "moving_average_forecast": float(moving_prediction),
            "random_forest_forecast": rf_prediction,
        })

        # Append predictions, not actual test values.
        rf_history.append(rf_prediction)
        naive_history.append(naive_prediction)
        moving_history.append(moving_prediction)

    return records


def calculate_product_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics separately for each product and model."""

    records = []

    for product_id, group in results.groupby("product_id"):

        actual = group["actual"]

        for model_name, prediction_column in MODEL_COLUMNS.items():

            predicted = group[prediction_column]

            records.append({
                "product_id": product_id,
                "model": model_name,
                "mae": mean_absolute_error(actual, predicted),
                "rmse": root_mean_squared_error(actual, predicted),
                "wape_pct": weighted_absolute_percentage_error(
                    actual,
                    predicted,
                ),
            })

    return pd.DataFrame(records)


def main() -> None:
    """Run the recursive three-month model comparison."""

    df = load_data()

    eligible = select_complete_products(df)

    all_records = []

    for _, product in eligible.groupby("product_id"):
        all_records.extend(forecast_one_product(product))

    results = pd.DataFrame(all_records)

    if results.empty:
        raise ValueError("No evaluation predictions were generated.")

    product_metrics = calculate_product_metrics(results)

    summary_rows = []
    for model_name, prediction_column in MODEL_COLUMNS.items():
        predicted = results[prediction_column]
        summary_rows.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(results["actual"], predicted),
                "rmse": root_mean_squared_error(results["actual"], predicted),
                "wape_pct": weighted_absolute_percentage_error(
                    results["actual"], predicted
                ),
                "training_start": START_MONTH,
                "training_end": TRAIN_END,
                "test_start": TEST_START,
                "test_end": TEST_END,
                "forecast_strategy": "recursive_3_month",
                "random_forest_n_estimators": (
                    RANDOM_FOREST_N_ESTIMATORS
                    if model_name == "Random Forest"
                    else pd.NA
                ),
                "random_forest_max_depth": (
                    RANDOM_FOREST_MAX_DEPTH
                    if model_name == "Random Forest"
                    else pd.NA
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    print("\n=== RECURSIVE RANDOM FOREST VS BASELINES ===")
    print(f"Eligible products: {eligible['product_id'].nunique()}")
    print(
        f"Training period: {START_MONTH:%Y-%m} "
        f"to {TRAIN_END:%Y-%m}"
    )
    print(
        f"Testing period: {TEST_START:%Y-%m} "
        f"to {TEST_END:%Y-%m}"
    )
    print("Forecast strategy: Recursive three-month prediction")
    print(
        "Random Forest: "
        f"{RANDOM_FOREST_N_ESTIMATORS} trees, "
        f"maximum depth {RANDOM_FOREST_MAX_DEPTH}"
    )
    print(
        "Limitation: this pre-regime-change test does not establish "
        "performance under the late-2017 data regime change."
    )
    print(f"Training observations: {len(eligible[eligible['month'] <= TRAIN_END])}")
    print(f"Test observations: {len(results)}")

    print("\n=== POOLED METRICS ACROSS TEST ROWS ===")
    print(
        summary[["model", "mae", "rmse", "wape_pct"]]
        .round(2)
        .to_string(index=False)
    )

    mae_table = product_metrics.pivot(
        index="product_id",
        columns="model",
        values="mae",
    )

    print("\n=== PRODUCT-LEVEL MAE WINS ===")
    print(
        "Random Forest wins vs Naive: "
        f"{(mae_table['Random Forest'] < mae_table['Naive']).sum()}"
    )
    print(
        "Random Forest wins vs Moving Average: "
        f"{(mae_table['Random Forest'] < mae_table['Moving Average']).sum()}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "random_forest_model_comparison.csv",
        index=False,
    )

    results.to_csv(
        OUTPUT_DIR / "random_forest_predictions.csv",
        index=False,
    )

    product_metrics.to_csv(
        OUTPUT_DIR / "random_forest_per_product_metrics.csv",
        index=False,
    )

    print("\nSaved model comparison.")
    print("Saved recursive predictions.")
    print("Saved per-product metrics.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
