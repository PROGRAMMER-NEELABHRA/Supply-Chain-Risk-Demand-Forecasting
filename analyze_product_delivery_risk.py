
"""Analyze product-level delivery risk from the integrated dataset."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"
INPUT_FILE = DATA_DIR / "integrated_supply_chain.csv"
OUTPUT_DIR = DATA_DIR / "risk_analysis"


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    required = [
        "order_item_id",
        "product_id",
        "product_name",
        "quantity",
        "order_profit",
        "actual_shipping_days",
        "scheduled_shipping_days",
        "late_delivery_risk",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["shipping_delay_days"] = (
        df["actual_shipping_days"] - df["scheduled_shipping_days"]
    )
    df["actually_late"] = (df["shipping_delay_days"] > 0).astype(int)

    product_risk = (
        df.groupby(["product_id", "product_name"], dropna=False)
        .agg(
            order_items=("order_item_id", "nunique"),
            units_sold=("quantity", "sum"),
            risk_flagged_items=("late_delivery_risk", "sum"),
            actually_late_items=("actually_late", "sum"),
            average_delay_days=("shipping_delay_days", "mean"),
            total_order_profit=("order_profit", "sum"),
        )
        .reset_index()
    )

    product_risk["risk_flag_percent"] = (
        product_risk["risk_flagged_items"]
        / product_risk["order_items"] * 100
    ).round(2)

    product_risk["actual_late_percent"] = (
        product_risk["actually_late_items"]
        / product_risk["order_items"] * 100
    ).round(2)

    product_risk["average_delay_days"] = (
        product_risk["average_delay_days"].round(2)
    )
    product_risk["total_order_profit"] = (
        product_risk["total_order_profit"].round(2)
    )

    product_risk = product_risk.sort_values(
        ["actual_late_percent", "order_items"],
        ascending=[False, False],
    )

    output_file = OUTPUT_DIR / "delivery_risk_by_product.csv"
    product_risk.to_csv(output_file, index=False)

    # Require at least 100 order items for the higher-volume shortlist.
    higher_volume = product_risk[product_risk["order_items"] >= 100]
    higher_volume = higher_volume.sort_values(
        ["actual_late_percent", "order_items"],
        ascending=[False, False],
    )
    higher_volume.to_csv(
        OUTPUT_DIR / "higher_volume_product_delivery_risk.csv",
        index=False,
    )

    print("\n=== PRODUCTS WITH HIGHEST ACTUAL LATE RATE ===")
    print(product_risk.head(10).to_string(index=False))

    print("\n=== HIGHER-VOLUME PRODUCTS WITH HIGHEST ACTUAL LATE RATE ===")
    if higher_volume.empty:
        print("No products meet the minimum of 100 order items.")
    else:
        print(higher_volume.head(10).to_string(index=False))

    print("\n=== OUTPUTS CREATED ===")
    print(output_file)
    print(OUTPUT_DIR / "higher_volume_product_delivery_risk.csv")
    print(f"Products analyzed: {len(product_risk)}")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
