
"""Generate quality-aware monthly demand forecasts."""

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feature_engineering import (
    DEFAULT_FEATURE_COLUMNS as FEATURE_COLUMNS,
)
from src.data.validators import (
    validate_forecast_output,
    validate_monthly_demand,
)
from src.forecasting.models import (
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_RANDOM_STATE,
    train_random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "quality_review"
    / "monthly_demand_observed_rebuilt.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

FORECAST_HORIZON = 3
MIN_HISTORY_MONTHS = 12
WARNING_CATEGORIES = (
    "STALE_HISTORY",
    "BASELINE_METHOD",
    "INTERNAL_MISSING_MONTHS",
)
WARNING_PRIORITY = (
    "STALE_HISTORY",
    "INTERNAL_MISSING_MONTHS",
    "BASELINE_METHOD",
)
WARNING_COLUMNS = [
    "product_id",
    "warning_category",
    "warning_message",
    "last_recorded_month",
    "months_since_last_record",
    "forecast_method",
    "forecast_confidence",
]


def load_data(file_path: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load and validate observed monthly demand."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["month"])

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
    validate_monthly_demand(df)

    return df.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)


def create_features(
    history: list[float],
    forecast_month: pd.Timestamp,
) -> pd.DataFrame:
    """Create one forecast row using actual or predicted history."""
    if len(history) < 12:
        raise ValueError("At least 12 prior observations are required.")

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


def _consecutive_observation_runs(product: pd.DataFrame) -> list[pd.DataFrame]:
    """Split observations whenever their calendar months are not adjacent."""
    ordered = product.sort_values("month").reset_index(drop=True)
    month_ordinals = ordered["month"].dt.year * 12 + ordered["month"].dt.month
    run_ids = month_ordinals.diff().ne(1).cumsum()
    return [
        run.reset_index(drop=True)
        for _, run in ordered.groupby(run_ids, sort=False)
    ]


def _random_forest_training_data(
    runs: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe training rows only within consecutive actual runs."""
    feature_rows = []
    targets = []

    for run in runs:
        history = run["demand_units"].astype(float).tolist()
        months = run["month"].tolist()
        for index in range(MIN_HISTORY_MONTHS, len(history)):
            features = create_features(history[:index], months[index])
            feature_rows.append(features)
            targets.append(history[index])

    if not feature_rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float)

    return pd.concat(feature_rows, ignore_index=True), pd.Series(targets)


def forecast_one_product(
    product: pd.DataFrame,
    cutoff_month: pd.Timestamp,
    *,
    forecast_horizon: int = FORECAST_HORIZON,
    min_history_months: int = MIN_HISTORY_MONTHS,
) -> tuple[list[dict], str | None]:
    """Forecast a product using RF, moving-average, or naive tiers."""
    if forecast_horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 month.")
    if min_history_months < 1:
        raise ValueError("Minimum history must be at least 1 month.")
    if product.empty:
        return [], "No observed history is available"

    product = product.sort_values("month").copy()
    product_id = int(product["product_id"].iloc[0])
    cutoff_month = pd.Timestamp(cutoff_month).to_period("M").to_timestamp()
    first_month = product["month"].min()
    last_month = product["month"].max()
    months_since_last = (
        (cutoff_month.year - last_month.year) * 12
        + cutoff_month.month - last_month.month
    )

    if months_since_last < 0:
        raise ValueError("Forecast cutoff cannot precede the last observation.")

    expected_months = pd.date_range(
        start=first_month,
        end=last_month,
        freq="MS",
    )
    observed = product.set_index("month")["demand_units"]
    internal_gaps = expected_months.difference(observed.index)
    runs = _consecutive_observation_runs(product)
    latest_run = runs[-1]
    latest_history = latest_run["demand_units"].astype(float).tolist()
    observed_history_months = len(product)

    if len(latest_history) > max(min_history_months, 12):
        training_features, targets = _random_forest_training_data(runs)
        model = train_random_forest(
            training_features[FEATURE_COLUMNS],
            targets,
        )
        forecast_method = "random_forest_recursive"
        model_name = "Random Forest"
        forecast_history = latest_history.copy()
        base_prediction = None
        confidence = "HIGH"
    elif observed_history_months >= 2:
        forecast_method = "moving_average_baseline"
        model_name = "Moving Average Baseline"
        base_prediction = sum(product["demand_units"].tail(3)) / min(
            observed_history_months, 3
        )
        forecast_history = []
        model = None
        confidence = "MEDIUM"
    else:
        forecast_method = "naive_last_observation_baseline"
        model_name = "Naive Baseline"
        base_prediction = float(product["demand_units"].iloc[-1])
        forecast_history = []
        model = None
        confidence = "LOW"

    if months_since_last > 0:
        confidence = "LOW"
        data_quality_status = "STALE_HISTORY_LOW_CONFIDENCE"
    elif len(internal_gaps) > 0:
        confidence = "MEDIUM" if confidence == "HIGH" else confidence
        data_quality_status = "INTERNAL_GAPS_WARNING"
    else:
        data_quality_status = "OBSERVED_HISTORY"

    warning_parts = []
    if months_since_last > 0:
        warning_parts.append(
            f"Stale history: last observation {last_month:%Y-%m}; "
            f"{months_since_last} months before forecast cutoff. "
            "The estimate is low confidence; unknown months were not filled."
        )
    if len(internal_gaps) > 0:
        warning_parts.append(
            f"{len(internal_gaps)} internal missing months in observed "
            "lifecycle; unknown months were not filled."
        )
    if forecast_method != "random_forest_recursive":
        warning_parts.append(
            f"{model_name} is a baseline estimate, not a Random Forest "
            "prediction."
        )

    forecasts = []
    for step in range(1, forecast_horizon + 1):
        forecast_month = cutoff_month + pd.offsets.MonthBegin(step)
        if model is not None:
            features = create_features(forecast_history, forecast_month)
            prediction = max(0.0, float(model.predict(features)[0]))
            forecast_history.append(prediction)
        else:
            prediction = max(0.0, float(base_prediction))

        forecasts.append(
            {
                "product_id": product_id,
                "forecast_month": forecast_month,
                "forecast_demand": round(prediction, 2),
                "model_name": model_name,
                "forecast_method": forecast_method,
                "forecast_confidence": confidence,
                "data_quality_status": data_quality_status,
                "first_recorded_month": first_month,
                "last_recorded_month": last_month,
                "observed_history_months": observed_history_months,
                "months_since_last_record": months_since_last,
                "internal_missing_months": len(internal_gaps),
                "forecast_origin_month": cutoff_month,
                "forecast_horizon_month": step,
                "model_n_estimators": (
                    RANDOM_FOREST_N_ESTIMATORS if model is not None else pd.NA
                ),
                "model_max_depth": (
                    RANDOM_FOREST_MAX_DEPTH if model is not None else pd.NA
                ),
                "model_random_state": (
                    RANDOM_FOREST_RANDOM_STATE if model is not None else pd.NA
                ),
            }
        )

    warning = "; ".join(warning_parts) if warning_parts else None
    return forecasts, warning


def _write_method_summary(forecasts: pd.DataFrame, output_file: Path) -> None:
    """Write product counts by method and forecast-row counts by confidence."""
    rows = []
    methods = [
        "random_forest_recursive",
        "moving_average_baseline",
        "naive_last_observation_baseline",
    ]
    for method in methods:
        group = forecasts.loc[forecasts["forecast_method"].eq(method)]
        rows.append(
            {
                "summary_type": "forecast_method",
                "category": method,
                "product_count": group["product_id"].nunique(),
                "forecast_count": len(group),
            }
        )

    for confidence in ["HIGH", "MEDIUM", "LOW"]:
        group = forecasts.loc[
            forecasts["forecast_confidence"].eq(confidence)
        ]
        rows.append(
            {
                "summary_type": "forecast_confidence",
                "category": confidence,
                "product_count": group["product_id"].nunique(),
                "forecast_count": len(group),
            }
        )
    pd.DataFrame(rows).to_csv(output_file, index=False)


def build_warning_records(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Create one filterable warning row per product, retaining all details."""
    records = []
    product_forecasts = forecasts.drop_duplicates("product_id")

    for row in product_forecasts.to_dict(orient="records"):
        months_since_last = int(row["months_since_last_record"])
        internal_missing_months = int(row["internal_missing_months"])
        method = row["forecast_method"]
        common = {
            "product_id": row["product_id"],
            "last_recorded_month": row["last_recorded_month"],
            "months_since_last_record": months_since_last,
            "forecast_method": method,
            "forecast_confidence": row["forecast_confidence"],
        }
        messages = []
        categories = []

        if months_since_last > 0:
            categories.append("STALE_HISTORY")
            messages.append(
                f"Last observed "
                f"{pd.Timestamp(row['last_recorded_month']):%Y-%m}; "
                f"{months_since_last} months before forecast cutoff. "
                "This estimate is low confidence; unknown months were not filled."
            )

        if method != "random_forest_recursive":
            categories.append("BASELINE_METHOD")
            messages.append(
                f"Forecast uses {method}, a baseline estimate rather than a "
                "Random Forest prediction; it has not been validated as a "
                "Random Forest forecast."
            )

        if internal_missing_months > 0:
            categories.append("INTERNAL_MISSING_MONTHS")
            messages.append(
                f"{internal_missing_months} internal missing months within "
                "the observed lifecycle; unknown months were not filled or "
                "used as actual demand."
            )

        if not categories:
            continue

        # A single primary category keeps product-level dashboard counts
        # mutually exclusive; warning_message retains all applicable details.
        primary_category = next(
            category
            for category in WARNING_PRIORITY
            if category in categories
        )
        records.append(
            {
                **common,
                "warning_category": primary_category,
                "warning_message": " ".join(messages),
            }
        )

    if not records:
        return pd.DataFrame(columns=WARNING_COLUMNS)

    warnings = pd.DataFrame.from_records(records, columns=WARNING_COLUMNS)
    return warnings.sort_values(
        ["product_id", "warning_category"]
    ).reset_index(drop=True)


def build_warning_summary(warnings: pd.DataFrame) -> pd.DataFrame:
    """Count warning rows and affected products by stable warning category."""
    if warnings.empty:
        return pd.DataFrame(
            [
                {
                    "warning_category": category,
                    "product_count": 0,
                    "warning_count": 0,
                }
                for category in WARNING_CATEGORIES
            ]
        )

    summary = (
        warnings.groupby("warning_category")
        .agg(
            product_count=("product_id", "nunique"),
            warning_count=("product_id", "size"),
        )
    )
    return (
        summary.reindex(WARNING_CATEGORIES, fill_value=0)
        .rename_axis("warning_category")
        .reset_index()
    )


def main() -> None:
    """Generate forecasts and save quality reports."""

    df = load_data()

    cutoff_month = (
        df["month"].max().to_period("M").to_timestamp()
    )

    all_forecasts = []
    skipped_products = []
    observed_products = set(df["product_id"].unique())

    for product_id, product in df.groupby("product_id"):

        product_forecasts, _warning = forecast_one_product(
            product,
            cutoff_month,
        )

        if not product_forecasts:
            skipped_products.append({
                "product_id": product_id,
                "reason": warning or "No forecast generated",
            })
            continue

        all_forecasts.extend(product_forecasts)

    if not all_forecasts:
        raise ValueError("No products have observed history to forecast.")

    forecasts = pd.DataFrame(all_forecasts).sort_values(
        ["product_id", "forecast_month"]
    )
    forecasted_products = set(forecasts["product_id"].unique())
    for product_id in sorted(observed_products - forecasted_products):
        if not any(row["product_id"] == product_id for row in skipped_products):
            skipped_products.append(
                {"product_id": product_id, "reason": "No forecast generated"}
            )

    counts = forecasts.groupby("product_id").size()
    invalid_counts = counts[counts != FORECAST_HORIZON]
    if not invalid_counts.empty:
        raise ValueError(
            "Forecast row count must equal the configured horizon for every "
            f"product: {invalid_counts.to_dict()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skipped_file = (
        OUTPUT_DIR / "forecast_products_skipped.csv"
    )

    warnings_file = (
        OUTPUT_DIR / "forecast_data_warnings.csv"
    )
    warning_summary_file = (
        OUTPUT_DIR / "forecast_warning_summary.csv"
    )

    forecasts_file = (
        OUTPUT_DIR / "future_demand_forecasts.csv"
    )
    summary_file = OUTPUT_DIR / "forecast_method_summary.csv"

    pd.DataFrame(
        skipped_products,
        columns=["product_id", "reason"],
    ).to_csv(skipped_file, index=False)

    validate_forecast_output(forecasts)
    warnings = build_warning_records(forecasts)
    warning_summary = build_warning_summary(warnings)
    forecasts.to_csv(forecasts_file, index=False)
    _write_method_summary(forecasts, summary_file)
    warnings.to_csv(warnings_file, index=False)
    warning_summary.to_csv(warning_summary_file, index=False)

    print("\n=== QUALITY-AWARE DEMAND FORECASTS ===")
    print(f"Dataset cutoff: {cutoff_month:%Y-%m}")
    print(f"Products in dataset: {df['product_id'].nunique()}")
    print(f"Products forecast: {forecasts['product_id'].nunique()}")
    print(f"Forecast rows: {len(forecasts)}")
    print(f"Products skipped: {len(skipped_products)}")
    stale_product_count = forecasts.loc[
        forecasts["data_quality_status"].eq("STALE_HISTORY_LOW_CONFIDENCE"),
        "product_id",
    ].nunique()
    print(f"Products with stale history: {stale_product_count}")
    print(f"Structured warnings: {len(warnings)}")
    print(f"Warning summary: {warning_summary_file}")
    print("\n=== FORECAST METHODS ===")
    print(pd.read_csv(summary_file).query(
        "summary_type == 'forecast_method'"
    )[["category", "product_count"]].to_string(index=False))
    print("\n=== FORECAST CONFIDENCE COUNTS ===")
    print(pd.read_csv(summary_file).query(
        "summary_type == 'forecast_confidence'"
    )[["category", "forecast_count"]].to_string(index=False))

    print("\n=== MONTHLY FORECAST TOTALS ===")
    print(
        forecasts.groupby("forecast_month")[
            "forecast_demand"
        ].sum().round(2).to_string()
    )

    print("\n=== SKIP REASONS ===")
    if skipped_products:
        print(
            pd.Series(
                [item["reason"] for item in skipped_products]
            ).value_counts().to_string()
        )

    print(f"\nSaved forecasts: {forecasts_file}")
    print(f"Saved skipped products: {skipped_file}")
    print(f"Saved warnings: {warnings_file}")
    print(f"Saved warning summary: {warning_summary_file}")
    print(f"Saved method summary: {summary_file}")


if __name__ == "__main__":
    main()
