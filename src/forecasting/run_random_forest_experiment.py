
"""Compare Random Forest and baseline models on a common test period."""

from pathlib import Path

import pandas as pd

from src.forecasting.baseline import (
    naive_forecast,
    moving_average_forecast,
)
from src.forecasting.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.forecasting.models import train_random_forest
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

MODEL_COLUMNS = {
    "Naive": "naive_forecast",
    "Moving Average": "moving_average_forecast",
    "Random Forest": "random_forest_forecast",
}


def load_and_validate_data() -> pd.DataFrame:
    """Load demand data and validate required fields and records."""
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

    if not pd.api.types.is_numeric_dtype(df["demand_units"]):
        raise ValueError("Demand must be numeric.")

    if not df["demand_units"].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all():
        raise ValueError("Demand contains non-finite values.")

    if (df["demand_units"] < 0).any():
        raise ValueError(
            "Negative demand found. Investigate before forecasting."
        )

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def select_complete_products(df: pd.DataFrame) -> pd.DataFrame:
    """Keep products with a complete monthly history in the shared period."""
    expected_months = pd.date_range(
        START_MONTH,
        TEST_END,
        freq="MS",
    )

    eligible_ids = []

    for product_id, group in df.groupby("product_id"):
        observed_months = (
            group["month"]
            .sort_values()
            .reset_index(drop=True)
        )

        if observed_months.equals(
            pd.Series(expected_months, name="month")
        ):
            eligible_ids.append(product_id)

    if not eligible_ids:
        raise ValueError(
            "No products have complete history in the common period."
        )

    return df.loc[df["product_id"].isin(eligible_ids)].copy()


def generate_baseline_predictions(
    eligible: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Generate rolling one-step baseline predictions for each product."""
    records = []

    for product_id, test_group in test.groupby("product_id"):
        product_history = (
            eligible.loc[eligible["product_id"] == product_id]
            .sort_values("month")
            .reset_index(drop=True)
        )

        for _, test_row in test_group.sort_values("month").iterrows():
            history = product_history.loc[
                product_history["month"] < test_row["month"],
                "demand_units",
            ]

            if history.empty:
                raise ValueError(
                    f"No historical demand for product {product_id}."
                )

            records.append(
                {
                    "product_id": product_id,
                    "month": test_row["month"],
                    "naive_forecast": float(
                        naive_forecast(history).iloc[-1]
                    ),
                    "moving_average_forecast": float(
                        moving_average_forecast(
                            history,
                            window=3,
                        ).iloc[-1]
                    ),
                }
            )

    return pd.DataFrame(records)


def calculate_product_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate evaluation metrics separately for every product and model."""
    records = []

    for product_id, group in results.groupby("product_id"):
        actual = group["actual"]

        for model_name, prediction_column in MODEL_COLUMNS.items():
            predicted = group[prediction_column]

            records.append(
                {
                    "product_id": product_id,
                    "model": model_name,
                    "mae": mean_absolute_error(actual, predicted),
                    "rmse": root_mean_squared_error(actual, predicted),
                    "wape_pct": weighted_absolute_percentage_error(
                        actual,
                        predicted,
                    ),
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    """Train Random Forest and compare it with both baseline models."""
    df = load_and_validate_data()
    eligible = select_complete_products(df)

    features = create_demand_features(eligible)

    train = features.loc[features["month"] <= TRAIN_END].copy()
    test = features.loc[
        features["month"].between(TEST_START, TEST_END)
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or test data is empty after feature generation."
        )

    if train[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Training features contain missing values.")

    if test[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Test features contain missing values.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["demand_units"]
    X_test = test[FEATURE_COLUMNS]

    model = train_random_forest(X_train, y_train)
    rf_predictions = model.predict(X_test)

    baseline_df = generate_baseline_predictions(eligible, test)

    results = test[
        ["product_id", "month", "demand_units"]
    ].rename(columns={"demand_units": "actual"}).reset_index(drop=True)

    results["random_forest_forecast"] = rf_predictions

    results = results.merge(
        baseline_df,
        on=["product_id", "month"],
        how="left",
        validate="one_to_one",
    )

    prediction_columns = list(MODEL_COLUMNS.values())
    if results[prediction_columns].isna().any().any():
        raise ValueError("Some predictions are missing.")

    product_metrics = calculate_product_metrics(results)

    # Mean per-product metrics, consistent with the baseline comparison.
    summary = (
        product_metrics.groupby("model")[["mae", "rmse", "wape_pct"]]
        .mean()
        .reindex(list(MODEL_COLUMNS))
    )

    print("\n=== RANDOM FOREST VS BASELINES ===")
    print(f"Eligible products: {eligible['product_id'].nunique()}")
    print(f"Training period: {START_MONTH:%Y-%m} to {TRAIN_END:%Y-%m}")
    print(f"Testing period: {TEST_START:%Y-%m} to {TEST_END:%Y-%m}")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")

    print("\n=== MEAN METRICS PER PRODUCT ===")
    print(summary.round(2).to_string())

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
    print(
        "Random Forest ties vs Moving Average: "
        f"{(mae_table['Random Forest'] == mae_table['Moving Average']).sum()}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "random_forest_model_comparison.csv",
        index_label="model",
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
    print("Saved predictions.")
    print("Saved per-product metrics.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
