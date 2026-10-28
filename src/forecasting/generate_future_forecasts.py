import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.preprocessing.feature_engineering import create_demand_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "monthly_demand_observed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

FORECAST_HORIZON = 3
MIN_HISTORY_MONTHS = 12

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
        raise ValueError(
            "Duplicate product-month records found. "
            "Resolve them before forecasting."
        )

    if (df["demand_units"] < 0).any():
        raise ValueError(
            "Negative demand found. Investigate before forecasting."
        )

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def build_monthly_history(
    product: pd.DataFrame,
    cutoff_month: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Build monthly history through the common cutoff.

    Missing months are filled with zero under the explicit assumption
    that no recorded orders means zero demand.
    """
    product = product.sort_values("month").copy()

    product_id = int(product["product_id"].iloc[0])
    first_month = product["month"].min()

    monthly_index = pd.date_range(
        start=first_month,
        end=cutoff_month,
        freq="MS",
    )

    observed = (
        product.set_index("month")["demand_units"]
        .reindex(monthly_index)
    )

    missing_months = int(observed.isna().sum())
    monthly_demand = observed.fillna(0.0)

    history = pd.DataFrame(
        {
            "month": monthly_index,
            "product_id": product_id,
            "demand_units": monthly_demand.to_numpy(),
        }
    )

    return history, missing_months


def forecast_one_product(
    product: pd.DataFrame,
    cutoff_month: pd.Timestamp,
) -> tuple[list[dict], str | None]:
    """Train one product model and forecast beyond the common cutoff."""
    product_id = int(product["product_id"].iloc[0])

    first_recorded_month = product["month"].min()
    last_recorded_month = product["month"].max()

    calendar_months = (
        (cutoff_month.year - first_recorded_month.year) * 12
        + cutoff_month.month
        - first_recorded_month.month
        + 1
    )

    if calendar_months < MIN_HISTORY_MONTHS:
        return [], (
            f"Insufficient history: {calendar_months} calendar months; "
            f"at least {MIN_HISTORY_MONTHS} required"
        )

    history, missing_months = build_monthly_history(
        product,
        cutoff_month,
    )

    demand_features = create_demand_features(history)

    usable = demand_features.dropna(
        subset=FEATURE_COLUMNS + ["demand_units"]
    ).copy()

    if usable.empty:
        return [], "No usable training rows after feature engineering"

    X_train = usable[FEATURE_COLUMNS]
    y_train = usable["demand_units"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    demand_history = history["demand_units"].astype(float).tolist()
    forecasts = []

    for step in range(1, FORECAST_HORIZON + 1):
        forecast_month = cutoff_month + pd.offsets.MonthBegin(step)

        features = pd.DataFrame(
            [{
                "lag_1": demand_history[-1],
                "lag_2": demand_history[-2],
                "lag_3": demand_history[-3],
                "lag_6": demand_history[-6],
                "lag_12": demand_history[-12],
                "rolling_mean_3": sum(demand_history[-3:]) / 3,
                "rolling_mean_6": sum(demand_history[-6:]) / 6,
                "month_number": forecast_month.month,
                "quarter": forecast_month.quarter,
            }],
            columns=FEATURE_COLUMNS,
        )

        prediction = max(
            0.0,
            float(model.predict(features)[0]),
        )

        months_since_record = (
            (cutoff_month.year - last_recorded_month.year) * 12
            + (cutoff_month.month - last_recorded_month.month)
        )

        forecasts.append(
            {
                "product_id": product_id,
                "forecast_month": forecast_month,
                "forecast_demand": round(prediction, 2),
                "model_name": "Random Forest",
                "model_max_depth": 5,
                "calendar_history_months": len(history),
                "first_recorded_month": first_recorded_month,
                "last_recorded_month": last_recorded_month,
                "forecast_origin_month": cutoff_month,
                "months_since_last_record": int(months_since_record),
                "missing_months_assumed_zero": missing_months,
                "forecast_horizon_month": step,
            }
        )

        # Use the prediction as input when forecasting the next month.
        demand_history.append(prediction)

    warning_parts = []

    if missing_months > 0:
        warning_parts.append(
            f"{missing_months} missing product-month records "
            "were treated as zero demand"
        )

    if last_recorded_month < cutoff_month:
        warning_parts.append(
            f"Last recorded demand was {last_recorded_month:%Y-%m}; "
            f"common cutoff is {cutoff_month:%Y-%m}"
        )

    warning = "; ".join(warning_parts) if warning_parts else None

    return forecasts, warning


def main() -> None:
    """Generate future forecasts and save results and warnings."""
    df = load_data()

    cutoff_month = (
        df["month"].max().to_period("M").to_timestamp()
    )

    all_forecasts = []
    skipped_products = []
    warnings = []

    for product_id, product in df.groupby("product_id"):
        try:
            product_forecasts, warning = forecast_one_product(
                product,
                cutoff_month,
            )

            if not product_forecasts:
                skipped_products.append(
                    {
                        "product_id": product_id,
                        "reason": warning or "No forecasts generated",
                    }
                )
                continue

            all_forecasts.extend(product_forecasts)

            if warning:
                warnings.append(
                    {
                        "product_id": product_id,
                        "warning": warning,
                    }
                )

        except (ValueError, KeyError) as exc:
            skipped_products.append(
                {
                    "product_id": product_id,
                    "reason": str(exc),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skipped_file = OUTPUT_DIR / "forecast_products_skipped.csv"
    warnings_file = OUTPUT_DIR / "forecast_data_warnings.csv"

    pd.DataFrame(
        skipped_products,
        columns=["product_id", "reason"],
    ).to_csv(skipped_file, index=False)

    pd.DataFrame(
        warnings,
        columns=["product_id", "warning"],
    ).to_csv(warnings_file, index=False)

    if not all_forecasts:
        print("\n=== NO FORECASTS GENERATED ===")
        print(f"Dataset cutoff: {cutoff_month:%Y-%m}")
        print(f"Products in dataset: {df['product_id'].nunique()}")
        print(f"Products skipped: {len(skipped_products)}")
        print("\nFirst skip reasons:")
        print(pd.DataFrame(skipped_products).head(20).to_string(index=False))
        raise ValueError(
            "No forecasts generated. Review the skip report."
        )

    forecasts = pd.DataFrame(all_forecasts).sort_values(
        ["product_id", "forecast_month"]
    )

    forecasts_file = OUTPUT_DIR / "future_demand_forecasts.csv"
    forecasts.to_csv(forecasts_file, index=False)

    print("\n=== FUTURE DEMAND FORECASTS ===")
    print(f"Dataset cutoff: {cutoff_month:%Y-%m}")
    print(
        f"Forecast months: "
        f"{', '.join(sorted(forecasts['forecast_month'].dt.strftime('%Y-%m').unique()))}"
    )
    print(f"Products forecast: {forecasts['product_id'].nunique()}")
    print(f"Forecast rows: {len(forecasts)}")
    print(f"Products skipped: {len(skipped_products)}")
    print(f"Products with data warnings: {len(warnings)}")

    print("\n=== SAMPLE FORECASTS ===")
    print(forecasts.head(12).to_string(index=False))

    print(f"\nSaved forecasts: {forecasts_file}")
    print(f"Saved skipped-product report: {skipped_file}")
    print(f"Saved data warnings: {warnings_file}")


if __name__ == "__main__":
    main()
