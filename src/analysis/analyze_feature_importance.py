
"""Analyze Random Forest feature importance for demand forecasting."""

from pathlib import Path

import pandas as pd

from src.forecasting.models import train_random_forest
from src.preprocessing.feature_engineering import create_demand_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

START_MONTH = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2016-10-01")
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


def load_training_data() -> pd.DataFrame:
    """Load demand history and validate its required columns."""
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


def main() -> None:
    """Train the existing model and export its feature importances."""
    df = load_training_data()

    # Match the first experiment's complete-history product selection.
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

    eligible = df.loc[df["product_id"].isin(eligible_ids)].copy()

    features = create_demand_features(eligible)

    train = features.loc[features["month"] <= TRAIN_END].copy()

    if train.empty:
        raise ValueError("No training rows available.")

    missing_features = set(FEATURE_COLUMNS) - set(train.columns)
    if missing_features:
        raise ValueError(
            f"Feature engineering did not create: {sorted(missing_features)}"
        )

    if train[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Training features contain missing values.")

    model = train_random_forest(
        train[FEATURE_COLUMNS],
        train["demand_units"],
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance["importance_pct"] = importance["importance"] * 100

    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
    print(f"Eligible products: {len(eligible_ids)}")
    print(f"Training observations: {len(train)}")
    print(importance.round(4).to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "random_forest_feature_importance.csv"
    importance.to_csv(output_file, index=False)

    print(f"\nSaved feature importance to: {output_file}")


if __name__ == "__main__":
    main()
