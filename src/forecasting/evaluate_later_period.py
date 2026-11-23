
"""Evaluate forecasting models on a later chronological test period."""

from pathlib import Path

import pandas as pd

from src.data.loaders import load_monthly_demand
from src.forecasting.baseline import (
    naive_forecast,
    moving_average_forecast,
)
from src.forecasting.evaluation import (
    calculate_product_metrics,
    select_products_with_complete_history,
)
from src.forecasting.models import train_random_forest
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

MODEL_COLUMNS = {
    "Naive": "naive_forecast",
    "Moving Average": "moving_average_forecast",
    "Random Forest": "random_forest_forecast",
}


def load_data() -> pd.DataFrame:
    """Load and validate the monthly demand dataset."""
    df = load_monthly_demand(DATA_FILE)
    df = df.loc[
        df["month"] >= START_MONTH,
        ["month", "product_id", "demand_units"],
    ].copy()

    if df.empty:
        raise ValueError("No monthly demand observations are available.")

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def select_eligible_products(
    df: pd.DataFrame,
    test_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Select products with complete history through the test period."""
    if df.empty:
        raise ValueError("No monthly demand observations are available.")
    test_end = pd.Timestamp(
        test_end if test_end is not None else df["month"].max()
    )
    eligible = select_products_with_complete_history(
        df,
        START_MONTH,
        test_end,
    )
    if eligible.empty:
        raise ValueError(
            f"No products have complete history through {test_end:%Y-%m}. "
            "The later-period evaluation is unavailable for this source."
        )

    return eligible


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
    test_end = pd.Timestamp(df["month"].max())
    eligible = select_eligible_products(df, test_end)

    features = create_demand_features(eligible)

    train = features.loc[features["month"] <= TRAIN_END].copy()
    test = features.loc[
        features["month"].between(TEST_START, test_end)
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
    product_metrics_df = calculate_product_metrics(
        results,
        MODEL_COLUMNS,
    )

    summary = (
        product_metrics_df.groupby("model")[["mae", "rmse", "wape_pct"]]
        .mean()
        .reindex(list(MODEL_COLUMNS))
    )

    print("\n=== LATER-PERIOD EVALUATION ===")
    print(f"Eligible products: {eligible['product_id'].nunique()}")
    print(f"Training: {START_MONTH:%Y-%m} to {TRAIN_END:%Y-%m}")
    print(f"Testing: {TEST_START:%Y-%m} to {test_end:%Y-%m}")
    print(
        "Protocol: one-step-ahead rolling evaluation; actual demand from "
        "completed test months is available for the next forecast step."
    )
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
