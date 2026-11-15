
"""Assess product history and status for demand forecasting."""

from pathlib import Path

import pandas as pd

from config import DATACO_RAW_DATA_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMAND_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "monthly_demand_observed.csv"
)

RAW_FILE = DATACO_RAW_DATA_FILE

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"
OUTPUT_FILE = OUTPUT_DIR / "forecast_eligibility.csv"

MIN_HISTORY_MONTHS = 12
STALE_DEMAND_MONTHS = 3


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the processed demand and raw order data."""
    for file_path in (DEMAND_FILE, RAW_FILE):
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    demand = pd.read_csv(DEMAND_FILE)
    raw = pd.read_csv(RAW_FILE)

    required_demand = {
        "month",
        "product_id",
        "demand_units",
    }
    required_raw = {
        "product_id",
        "product_status",
        "order_date",
    }

    missing_demand = required_demand - set(demand.columns)
    missing_raw = required_raw - set(raw.columns)

    if missing_demand:
        raise ValueError(
            f"Demand data missing columns: {sorted(missing_demand)}"
        )

    if missing_raw:
        raise ValueError(
            f"Raw data missing columns: {sorted(missing_raw)}"
        )

    demand["month"] = pd.to_datetime(
        demand["month"], errors="coerce"
    )
    demand["product_id"] = pd.to_numeric(
        demand["product_id"], errors="coerce"
    )
    demand["demand_units"] = pd.to_numeric(
        demand["demand_units"], errors="coerce"
    )

    if demand[
        ["month", "product_id", "demand_units"]
    ].isna().any().any():
        raise ValueError(
            "Demand data contains invalid or missing required values."
        )

    if demand.duplicated(["product_id", "month"]).any():
        raise ValueError(
            "Duplicate product-month records found in demand data."
        )

    raw["product_id"] = pd.to_numeric(
        raw["product_id"], errors="coerce"
    )
    raw["order_date"] = pd.to_datetime(
        raw["order_date"],
        errors="coerce",
        dayfirst=True,
    )

    raw = raw.dropna(subset=["product_id"]).copy()

    return demand, raw


def build_history_report(
    demand: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Combine demand history, product status, and raw order dates."""
    cutoff = demand["month"].max()

    history = (
        demand.groupby("product_id")
        .agg(
            first_recorded_month=("month", "min"),
            last_recorded_month=("month", "max"),
            recorded_months=("month", "nunique"),
            total_demand=("demand_units", "sum"),
        )
        .reset_index()
    )

    history["calendar_months"] = (
        (
            history["last_recorded_month"].dt.year
            - history["first_recorded_month"].dt.year
        )
        * 12
        + history["last_recorded_month"].dt.month
        - history["first_recorded_month"].dt.month
        + 1
    )

    history["missing_months_within_history"] = (
        history["calendar_months"]
        - history["recorded_months"]
    )

    history["months_since_last_record"] = (
        (
            cutoff.year
            - history["last_recorded_month"].dt.year
        )
        * 12
        + cutoff.month
        - history["last_recorded_month"].dt.month
    )

    # Identify products whose status changes across raw records.
    status_counts = (
        raw.groupby("product_id")["product_status"].nunique()
    )
    conflicting_ids = set(
        status_counts[status_counts > 1].index
    )

    # Preserve one non-null status value per product.
    status = (
        raw.groupby("product_id")["product_status"]
        .agg(
            lambda values: (
                values.dropna().iloc[0]
                if not values.dropna().empty
                else pd.NA
            )
        )
        .rename("product_status")
        .reset_index()
    )

    # Find the latest raw order date for each product.
    valid_order_dates = raw.dropna(
        subset=["order_date"]
    ).copy()

    last_orders = (
        valid_order_dates.groupby("product_id")
        .agg(
            last_raw_order_date=("order_date", "max"),
            raw_order_rows=("order_date", "size"),
        )
        .reset_index()
    )

    report = history.merge(
        status,
        on="product_id",
        how="left",
    )

    report = report.merge(
        last_orders,
        on="product_id",
        how="left",
    )

    report["status_conflict"] = report[
        "product_id"
    ].isin(conflicting_ids)

    report["status_missing"] = report[
        "product_status"
    ].isna()

    # Conservative eligibility checks:
    # adequate history, no gaps, recent demand, and consistent status.
    report["history_eligible"] = (
        (report["recorded_months"] >= MIN_HISTORY_MONTHS)
        & (report["missing_months_within_history"] == 0)
        & (report["months_since_last_record"] == 0)
        & (~report["status_conflict"])
        & (~report["status_missing"])
    )

    def explain_eligibility(row: pd.Series) -> str:
        """Explain why a product passes or fails history checks."""
        reasons = []

        if row["recorded_months"] < MIN_HISTORY_MONTHS:
            reasons.append("Fewer than 12 recorded months")

        if row["missing_months_within_history"] > 0:
            reasons.append("Gaps within recorded history")

        if row["months_since_last_record"] > 0:
            reasons.append(
                "No recorded demand near dataset cutoff"
            )

        if row["status_conflict"]:
            reasons.append("Conflicting product status")

        if row["status_missing"]:
            reasons.append("Missing product status")

        return (
            "; ".join(reasons)
            if reasons
            else "Eligible by history checks"
        )

    report["eligibility_reason"] = report.apply(
        explain_eligibility,
        axis=1,
    )

    # Flag stale demand where a later raw order exists.
    report["raw_order_after_last_demand"] = (
        report["last_raw_order_date"].notna()
        & (
            report["last_raw_order_date"]
            > report["last_recorded_month"]
        )
    )

    report["stale_demand_with_later_raw_order"] = (
        (
            report["months_since_last_record"]
            >= STALE_DEMAND_MONTHS
        )
        & report["raw_order_after_last_demand"]
    )

    return report, cutoff


def print_summary(
    report: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> None:
    """Print eligibility and stale-demand diagnostic summaries."""
    print("\n=== FORECAST ELIGIBILITY SUMMARY ===")
    print(f"Dataset cutoff: {cutoff:%Y-%m}")
    print(f"Products analyzed: {len(report)}")
    print(
        "Eligible by history checks: "
        f"{int(report['history_eligible'].sum())}"
    )
    print(
        "Not eligible by history checks: "
        f"{int((~report['history_eligible']).sum())}"
    )

    print("\nEligibility reasons:")
    print(
        report["eligibility_reason"]
        .value_counts()
        .to_string()
    )

    print("\n=== PRODUCTS WITH STALE DEMAND BUT LATER RAW ORDERS ===")

    recent_raw_orders = report.loc[
        report["stale_demand_with_later_raw_order"],
        [
            "product_id",
            "product_status",
            "last_recorded_month",
            "last_raw_order_date",
            "months_since_last_record",
        ],
    ].sort_values(
        "last_raw_order_date",
        ascending=False,
    )

    print(
        "Products matching this condition: "
        f"{len(recent_raw_orders)}"
    )

    if recent_raw_orders.empty:
        print(
            "No products matched. This does not prove that "
            "the products were discontinued."
        )
    else:
        print(
            recent_raw_orders.head(20).to_string(index=False)
        )

    print("\n=== PRODUCT STATUS SUMMARY ===")
    print(
        report["product_status"]
        .value_counts(dropna=False)
        .to_string()
    )


def main() -> None:
    """Run the eligibility analysis and save its report."""
    demand, raw = load_data()
    report, cutoff = build_history_report(demand, raw)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.sort_values(
        ["history_eligible", "recorded_months"],
        ascending=[False, False],
    ).to_csv(OUTPUT_FILE, index=False)

    print_summary(report, cutoff)
    print(f"\nSaved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
