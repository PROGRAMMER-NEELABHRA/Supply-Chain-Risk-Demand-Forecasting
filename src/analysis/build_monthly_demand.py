
"""Aggregate order records into monthly product demand."""

from datetime import date, datetime
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATACO_RAW_DATA_FILE
from src.data.loaders import load_csv
from src.data.validators import (
    parse_raw_order_dates,
    validate_monthly_demand,
    validate_raw_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = DATACO_RAW_DATA_FILE
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
QUALITY_OUTPUT_DIR = PROJECT_ROOT / "data" / "quality_review"
FORECAST_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"


def aggregate_monthly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid raw order items without inferring absent months."""
    validate_raw_data(df)

    transactions = df.copy()
    transactions["order_date"] = parse_raw_order_dates(
        transactions["order_date"]
    )
    if transactions["order_date"].isna().any():
        raise ValueError("Raw data contains invalid order dates.")
    transactions["quantity"] = pd.to_numeric(
        transactions["quantity"], errors="raise"
    )
    transactions["month"] = (
        transactions["order_date"].dt.to_period("M").dt.to_timestamp()
    )

    aggregations = {
        "demand_units": ("quantity", "sum"),
        "transaction_rows": ("quantity", "size"),
    }
    if "order_id" in transactions.columns:
        aggregations["unique_orders"] = ("order_id", "nunique")

    return (
        transactions.groupby(
            ["month", "product_id"], as_index=False
        )
        .agg(**aggregations)
        .sort_values(["product_id", "month"])
        .reset_index(drop=True)
    )


def build_source_metadata(
    raw: pd.DataFrame,
    source_file: str | Path,
) -> dict[str, int | str]:
    """Describe the validated raw source for the quality summary."""
    validate_raw_data(raw)
    dates = parse_raw_order_dates(raw["order_date"])

    return {
        "source_filename": Path(source_file).name,
        "source_row_count": len(raw),
        "source_product_count": int(
            pd.to_numeric(raw["product_id"]).nunique()
        ),
        "source_first_order_date": dates.min().date().isoformat(),
        "source_last_order_date": dates.max().date().isoformat(),
    }


def build_quality_artifacts(
    monthly: pd.DataFrame,
    *,
    source_metadata: dict[str, int | str] | None = None,
) -> dict[str, int | str]:
    """Classify absent product-months and isolate labelled gap estimates."""
    validate_monthly_demand(monthly)
    if "transaction_rows" not in monthly.columns:
        raise ValueError(
            "Missing monthly demand columns: ['transaction_rows']"
        )

    transaction_rows = monthly["transaction_rows"]
    if transaction_rows.isna().any():
        raise ValueError("transaction_rows must not contain null values.")
    if not transaction_rows.map(
        lambda value: isinstance(value, Number)
        and not isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("transaction_rows must contain numeric values.")

    transaction_rows = pd.to_numeric(transaction_rows, errors="raise")
    if not np.isfinite(transaction_rows).all():
        raise ValueError("transaction_rows must contain finite numeric values.")
    if (transaction_rows <= 0).any() or (
        transaction_rows % 1 != 0
    ).any():
        raise ValueError("transaction_rows must contain positive integers.")

    supported_month_types = (str, date, datetime, np.datetime64)
    if not monthly["month"].map(
        lambda value: isinstance(value, supported_month_types)
    ).all():
        raise ValueError(
            "Monthly month values must be date strings or datetime values."
        )
    try:
        normalized_months = pd.to_datetime(
            monthly["month"],
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Monthly month values must be valid date strings or datetime values."
        ) from exc
    if not pd.api.types.is_datetime64_any_dtype(normalized_months.dtype):
        raise ValueError(
            "Monthly month values must use consistent timezone-naive dates."
        )
    if normalized_months.dt.tz is not None:
        raise ValueError(
            "Monthly month values must be timezone-naive."
        )

    monthly = monthly.copy()
    monthly["month"] = normalized_months
    validate_monthly_demand(monthly)

    products = monthly["product_id"].drop_duplicates().sort_values()
    months = pd.date_range(
        monthly["month"].min(),
        monthly["month"].max(),
        freq="MS",
    )
    panel = pd.MultiIndex.from_product(
        [products, months],
        names=["product_id", "month"],
    ).to_frame(index=False)
    panel = panel.merge(monthly, on=["product_id", "month"], how="left")

    lifecycle = (
        monthly.groupby("product_id")["month"]
        .agg(first_observed_month="min", last_observed_month="max")
        .reset_index()
    )
    panel = panel.merge(lifecycle, on="product_id", how="left")

    observed = panel["demand_units"].notna()
    before_first = panel["month"] < panel["first_observed_month"]
    after_last = panel["month"] > panel["last_observed_month"]
    panel["record_status"] = np.select(
        [observed, before_first, after_last],
        [
            "observed",
            "before_first_observation_not_assumed_zero",
            "after_last_observation_not_assumed_zero",
        ],
        default="missing_inside_observed_lifecycle_review_required",
    )
    panel["demand_units_imputed"] = np.nan
    panel["imputation_method"] = pd.NA

    internal_gap_frames = []
    for product_id, group in panel.groupby("product_id", sort=False):
        gap_mask = group["record_status"].eq(
            "missing_inside_observed_lifecycle_review_required"
        )
        if not gap_mask.any():
            continue

        indexed_demand = group.set_index("month")["demand_units"]
        interpolated = indexed_demand.interpolate(
            method="linear", limit_area="inside"
        )
        gap_indices = group.index[gap_mask]
        panel.loc[gap_indices, "demand_units_imputed"] = (
            interpolated.loc[group.loc[gap_mask, "month"]].to_numpy()
        )
        panel.loc[gap_indices, "imputation_method"] = (
            "linear_interpolation_synthetic_not_observed"
        )
        internal_gap_frames.append(
            panel.loc[
                gap_indices,
                [
                    "product_id",
                    "month",
                    "demand_units_imputed",
                    "imputation_method",
                ],
            ]
        )

    observed_rows = panel.loc[observed].copy()
    observed_rows["demand_units_imputed"] = observed_rows["demand_units"]
    observed_rows["imputation_method"] = pd.NA
    imputed_rows = panel.loc[panel["demand_units_imputed"].notna()].copy()
    observed_and_labelled = pd.concat(
        [
            observed_rows[
                [
                    "product_id",
                    "month",
                    "demand_units",
                    "demand_units_imputed",
                    "record_status",
                    "imputation_method",
                ]
            ],
            imputed_rows.assign(record_status="synthetic_internal_gap")[
                [
                    "product_id",
                    "month",
                    "demand_units",
                    "demand_units_imputed",
                    "record_status",
                    "imputation_method",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values(["product_id", "month"])

    QUALITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rebuilt_columns = [
        "product_id",
        "month",
        "demand_units",
        "transaction_rows",
    ]
    if "unique_orders" in monthly.columns:
        rebuilt_columns.append("unique_orders")
    monthly[rebuilt_columns].to_csv(
        QUALITY_OUTPUT_DIR / "monthly_demand_observed_rebuilt.csv",
        index=False,
    )
    panel.to_csv(
        QUALITY_OUTPUT_DIR / "product_month_quality_panel.csv",
        index=False,
    )
    gap_report = (
        pd.concat(internal_gap_frames, ignore_index=True)
        if internal_gap_frames
        else pd.DataFrame(
            columns=[
                "product_id",
                "month",
                "demand_units_imputed",
                "imputation_method",
            ]
        )
    )
    gap_report.to_csv(
        QUALITY_OUTPUT_DIR / "internal_gaps_imputed_review.csv",
        index=False,
    )
    observed_and_labelled.to_csv(
        QUALITY_OUTPUT_DIR
        / "monthly_demand_observed_plus_labelled_imputation.csv",
        index=False,
    )

    summary = {
        "raw_transaction_records": int(monthly["transaction_rows"].sum()),
        "unique_products": int(monthly["product_id"].nunique()),
        "observed_calendar_months": int(monthly["month"].nunique()),
        "first_observed_month": monthly["month"].min().strftime("%Y-%m"),
        "last_observed_month": monthly["month"].max().strftime("%Y-%m"),
        "observed_product_months": int(observed.sum()),
        "expected_full_grid_product_months": int(len(panel)),
        "absent_product_months": int((~observed).sum()),
        "internal_missing_months": int(len(gap_report)),
        "products_with_internal_gaps": int(
            gap_report["product_id"].nunique()
        ),
        "synthetic_imputed_values": int(len(gap_report)),
    }
    if source_metadata is not None:
        summary.update(source_metadata)
        summary["source_coverage_limit"] = (
            f"Raw transactions end {source_metadata['source_last_order_date']}; "
            "later demand regimes are not observed or validated."
        )

    pd.DataFrame(
        [
            {"metric": name, "value": value}
            for name, value in summary.items()
        ]
    ).to_csv(
        FORECAST_OUTPUT_DIR / "data_quality_summary.csv",
        index=False,
    )
    return summary


def main() -> None:
    """Create a monthly demand table for each product."""

    df = load_csv(INPUT_FILE)
    source_metadata = build_source_metadata(df, INPUT_FILE)
    monthly = aggregate_monthly_demand(df)
    quality_summary = build_quality_artifacts(
        monthly,
        source_metadata=source_metadata,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_file = OUTPUT_DIR / "monthly_demand_observed.csv"
    monthly[["month", "product_id", "demand_units"]].to_csv(
        processed_file, index=False
    )

    print("\n=== MONTHLY DEMAND AGGREGATION ===")
    print(f"Input records: {len(df):,}")
    print(
        "Source date coverage: "
        f"{source_metadata['source_first_order_date']} to "
        f"{source_metadata['source_last_order_date']}"
    )
    print(f"Monthly records: {len(monthly):,}")
    print(f"Products: {monthly['product_id'].nunique():,}")
    print(f"Months: {monthly['month'].nunique():,}")
    print(f"Total demand: {monthly['demand_units'].sum():,.0f}")
    print(
        "Internal missing months classified (not used as observations): "
        f"{quality_summary['internal_missing_months']}"
    )
    print(f"Saved observed demand: {processed_file}")
    print(f"Saved verified rebuild: {QUALITY_OUTPUT_DIR}")
    print(
        "Saved data-quality summary: "
        f"{FORECAST_OUTPUT_DIR / 'data_quality_summary.csv'}"
    )


if __name__ == "__main__":
    main()
