
"""Evaluate forecasting models on a later chronological test period."""

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

MODEL_COLUMNS = {
    "Naive": "naive_forecast",
    "Moving Average": "moving_average_forecast",
    "Random Forest": "random_forest_forecast",
}


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

    if not df["demand_units"].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all():
        raise ValueError("Demand contains non-finite values.")

    if (df["demand_units"] < 0).any():
        raise ValueError("Negative demand found.")

    if df.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate product-month records found.")

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def select_eligible_products(df: pd.DataFrame) -> pd.DataFrame:
    """Select products with complete history through the test period."""
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
            "No products have complete history through October 2017. "
            "Inspect product coverage before changing the evaluation period."
        )

    return df.loc[df["product_id"].isin(eligible_ids)].copy()


def build_baseline_predictions(
    eligible: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Create one-step-ahead baseline predictions for each product."""
    rows = []

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

            if history.empty:
                raise ValueError(
                    f"No prior history for product {product_id}."
                )

            rows.append(
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

    return pd.DataFrame(rows)


def main() -> None:
    """Train and compare models on the later historical period."""
    df = load_data()
    eligible = select_eligible_products(df)

    features = create_demand_features(eligible)

    train = features.loc[features["month"] <= TRAIN_END].copy()
    test = features.loc[
        features["month"].between(TEST_START, TEST_END)
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test dataset is empty.")

    model = train_random_forest(
        train[FEATURE_COLUMNS],
        train["demand_units"],
    )

    test = test.reset_index(drop=True)
    rf_predictions = model.predict(test[FEATURE_COLUMNS])

    baseline_df = build_baseline_predictions(eligible, test)

    results = test[
        ["product_id", "month", "demand_units"]
    ].rename(columns={"demand_units": "actual"})

    results["random_forest_forecast"] = rf_predictions

    results = results.merge(
        baseline_df,
        on=["product_id", "month"],
        how="left",
        validate="one_to_one",
    )

    if results[list(MODEL_COLUMNS.values())].isna().any().any():
        raise ValueError("Missing predictions found.")

    # Average metrics across products, matching the first experiment.
    product_metrics = []

    for product_id, group in results.groupby("product_id"):
        for model_name, column in MODEL_COLUMNS.items():
            actual = group["actual"]
            predicted = group[column]

            product_metrics.append(
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

    product_metrics_df = pd.DataFrame(product_metrics)

    summary = (
        product_metrics_df.groupby("model")[["mae", "rmse", "wape_pct"]]
        .mean()
        .reindex(list(MODEL_COLUMNS))
    )

    print("\n=== LATER-PERIOD EVALUATION ===")
    print(f"Eligible products: {eligible['product_id'].nunique()}")
    print(f"Training: {START_MONTH:%Y-%m} to {TRAIN_END:%Y-%m}")
    print(f"Testing: {TEST_START:%Y-%m} to {TEST_END:%Y-%m}")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")

    print("\n=== MEAN METRICS PER PRODUCT ===")
    print(summary.round(2).to_string())

    mae_table = product_metrics_df.pivot(
        index="product_id",
        columns="model",
        values="mae",
    )

    print("\n=== RANDOM FOREST MAE WINS ===")
    print(
        "Against naive:",
        int(
            (mae_table["Random Forest"] < mae_table["Naive"]).sum()
        ),
    )
    print(
        "Against moving average:",
        int(
            (
                mae_table["Random Forest"]
                < mae_table["Moving Average"]
            ).sum()
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "later_period_predictions.csv",
        index=False,
    )
    product_metrics_df.to_csv(
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
