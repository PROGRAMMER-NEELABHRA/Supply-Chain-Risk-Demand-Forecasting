
"""Analyze delivery and operational risk using integrated DataCo data."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"
INPUT_FILE = DATA_DIR / "integrated_supply_chain.csv"
OUTPUT_DIR = DATA_DIR / "risk_analysis"

REQUIRED_COLUMNS = [
    "order_item_id",
    "order_date",
    "product_id",
    "quantity",
    "order_region",
    "actual_shipping_days",
    "scheduled_shipping_days",
    "delivery_status",
    "late_delivery_risk",
    "shipping_mode",
    "market",
    "order_country",
]


def main():
    """Generate delivery risk summaries without modifying source data."""
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    # Parse dates and create a delivery-delay measure.
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["shipping_delay_days"] = (
        df["actual_shipping_days"] - df["scheduled_shipping_days"]
    )
    df["is_late_by_days"] = (df["shipping_delay_days"] > 0).astype(int)
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)

    # Overall delivery summary.
    overall = pd.DataFrame([{
        "order_items": len(df),
        "late_delivery_risk_flagged_items": int(
            (df["late_delivery_risk"] == 1).sum()
        ),
        "late_delivery_risk_flag_percent": round(
            (df["late_delivery_risk"] == 1).mean() * 100, 2
        ),
        "items_shipping_later_than_schedule": int(
            (df["shipping_delay_days"] > 0).sum()
        ),
        "actual_late_percent": round(
            (df["shipping_delay_days"] > 0).mean() * 100, 2
        ),
        "average_actual_shipping_days": round(
            df["actual_shipping_days"].mean(), 2
        ),
        "average_scheduled_shipping_days": round(
            df["scheduled_shipping_days"].mean(), 2
        ),
        "average_shipping_delay_days": round(
            df["shipping_delay_days"].mean(), 2
        ),
    }])
    overall.to_csv(OUTPUT_DIR / "overall_delivery_risk.csv", index=False)

    # Risk by shipping mode.
    shipping = (
        df.groupby("shipping_mode", dropna=False)
        .agg(
            order_items=("order_item_id", "count"),
            risk_flagged_items=("late_delivery_risk", "sum"),
            actual_late_items=("is_late_by_days", "sum"),
            avg_actual_days=("actual_shipping_days", "mean"),
            avg_scheduled_days=("scheduled_shipping_days", "mean"),
            avg_delay_days=("shipping_delay_days", "mean"),
        )
        .reset_index()
    )
    shipping["risk_flag_percent"] = (
        shipping["risk_flagged_items"] / shipping["order_items"] * 100
    ).round(2)
    shipping["actual_late_percent"] = (
        shipping["actual_late_items"] / shipping["order_items"] * 100
    ).round(2)
    shipping = shipping.round(2).sort_values(
        "actual_late_percent", ascending=False
    )
    shipping.to_csv(
        OUTPUT_DIR / "delivery_risk_by_shipping_mode.csv", index=False
    )

    # Risk by region.
    region = (
        df.groupby("order_region", dropna=False)
        .agg(
            order_items=("order_item_id", "count"),
            risk_flagged_items=("late_delivery_risk", "sum"),
            actual_late_items=("is_late_by_days", "sum"),
            avg_delay_days=("shipping_delay_days", "mean"),
        )
        .reset_index()
    )
    region["risk_flag_percent"] = (
        region["risk_flagged_items"] / region["order_items"] * 100
    ).round(2)
    region["actual_late_percent"] = (
        region["actual_late_items"] / region["order_items"] * 100
    ).round(2)
    region = region.round(2).sort_values(
        "actual_late_percent", ascending=False
    )
    region.to_csv(OUTPUT_DIR / "delivery_risk_by_region.csv", index=False)

    # Risk by country.
    country = (
        df.groupby("order_country", dropna=False)
        .agg(
            order_items=("order_item_id", "count"),
            risk_flagged_items=("late_delivery_risk", "sum"),
            actual_late_items=("is_late_by_days", "sum"),
            avg_delay_days=("shipping_delay_days", "mean"),
        )
        .reset_index()
    )
    country["risk_flag_percent"] = (
        country["risk_flagged_items"] / country["order_items"] * 100
    ).round(2)
    country["actual_late_percent"] = (
        country["actual_late_items"] / country["order_items"] * 100
    ).round(2)
    country = country.round(2).sort_values(
        ["actual_late_percent", "order_items"], ascending=[False, False]
    )
    country.to_csv(OUTPUT_DIR / "delivery_risk_by_country.csv", index=False)

    # Risk by month to identify changes over time.
    monthly = (
        df.dropna(subset=["order_date"])
        .groupby("order_month")
        .agg(
            order_items=("order_item_id", "count"),
            risk_flagged_items=("late_delivery_risk", "sum"),
            actual_late_items=("is_late_by_days", "sum"),
            avg_delay_days=("shipping_delay_days", "mean"),
        )
        .reset_index()
    )
    monthly["risk_flag_percent"] = (
        monthly["risk_flagged_items"] / monthly["order_items"] * 100
    ).round(2)
    monthly["actual_late_percent"] = (
        monthly["actual_late_items"] / monthly["order_items"] * 100
    ).round(2)
    monthly = monthly.round(2)
    monthly.to_csv(OUTPUT_DIR / "delivery_risk_by_month.csv", index=False)

    print("\n=== OVERALL DELIVERY RISK ===")
    print(overall.to_string(index=False))

    print("\n=== DELIVERY RISK BY SHIPPING MODE ===")
    print(shipping.to_string(index=False))

    print("\n=== FIVE REGIONS WITH HIGHEST ACTUAL LATE RATE ===")
    print(region.head(5).to_string(index=False))

    print("\n=== FIVE COUNTRIES WITH HIGHEST ACTUAL LATE RATE ===")
    print(country.head(5).to_string(index=False))

    print("\n=== RISK ANALYSIS COMPLETE ===")
    print(f"Input rows analyzed: {len(df):,}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("The original integrated dataset was not modified.")


if __name__ == "__main__":
    main()
