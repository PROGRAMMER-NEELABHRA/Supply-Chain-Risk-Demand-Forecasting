
"""Build a consolidated, evidence-based supply-chain risk report."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"
RISK_DIR = DATA_DIR / "risk_analysis"
OUTPUT_FILE = RISK_DIR / "risk_priority_report.csv"


def main():
    product_file = RISK_DIR / "higher_volume_product_delivery_risk.csv"
    region_file = RISK_DIR / "delivery_risk_by_region.csv"
    shipping_file = RISK_DIR / "delivery_risk_by_shipping_mode.csv"
    country_file = RISK_DIR / "delivery_risk_by_country.csv"

    required_files = [
        product_file,
        region_file,
        shipping_file,
        country_file,
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required report not found: {path}\n"
                "Run the previous risk-analysis scripts first."
            )

    products = pd.read_csv(product_file)
    regions = pd.read_csv(region_file)
    shipping = pd.read_csv(shipping_file)
    countries = pd.read_csv(country_file)

    records = []

    # Product priorities: minimum 100 order items.
    for _, row in products.iterrows():
        records.append({
            "risk_area": "Product delivery",
            "entity": str(row["product_name"]),
            "order_items": int(row["order_items"]),
            "actual_late_percent": float(row["actual_late_percent"]),
            "average_delay_days": float(row["average_delay_days"]),
            "priority_basis": (
                "High-volume product with observed late deliveries"
            ),
            "interpretation": (
                "Investigate fulfillment and shipping patterns; "
                "this association does not establish causation."
            ),
        })

    # Regional priorities: minimum 100 order items to avoid tiny samples.
    for _, row in regions.iterrows():
        count = int(row["order_items"])
        if count < 100:
            continue
        records.append({
            "risk_area": "Regional delivery",
            "entity": str(row["order_region"]),
            "order_items": count,
            "actual_late_percent": float(row["actual_late_percent"]),
            "average_delay_days": float(row["avg_delay_days"]),
            "priority_basis": (
                "Region with at least 100 items and observed late deliveries"
            ),
            "interpretation": (
                "Compare delivery performance with other regions and "
                "investigate operational differences."
            ),
        })

    # Shipping modes: these are comparison groups, not causal explanations.
    for _, row in shipping.iterrows():
        records.append({
            "risk_area": "Shipping mode",
            "entity": str(row["shipping_mode"]),
            "order_items": int(row["order_items"]),
            "actual_late_percent": float(row["actual_late_percent"]),
            "average_delay_days": float(row["avg_delay_days"]),
            "priority_basis": "Observed delivery performance by shipping mode",
            "interpretation": (
                "Review scheduled service targets and actual performance; "
                "shipping modes may have different delivery commitments."
            ),
        })

    # Country priorities: minimum 100 items.
    for _, row in countries.iterrows():
        count = int(row["order_items"])
        if count < 100:
            continue
        records.append({
            "risk_area": "Country delivery",
            "entity": str(row["order_country"]),
            "order_items": count,
            "actual_late_percent": float(row["actual_late_percent"]),
            "average_delay_days": float(row["avg_delay_days"]),
            "priority_basis": (
                "Country with at least 100 items and observed late deliveries"
            ),
            "interpretation": (
                "Use as a screening signal; examine sample size, shipping mode, "
                "and route mix before drawing conclusions."
            ),
        })

    report = pd.DataFrame(records)
    report = report.sort_values(
        ["actual_late_percent", "order_items"],
        ascending=[False, False],
    ).reset_index(drop=True)

    report.insert(0, "review_rank", range(1, len(report) + 1))
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n=== RISK PRIORITY REPORT ===")
    print(f"Total findings: {len(report)}")
    print("\nFindings with the highest observed late-delivery rates:")
    print(report.head(15).to_string(index=False))

    print("\n=== FINDINGS BY RISK AREA ===")
    print(report.groupby("risk_area").size().to_string())

    print(f"\nSaved report: {OUTPUT_FILE}")
    print("This is a screening report, not a causal risk model.")
    print("No source dataset was modified.")


if __name__ == "__main__":
    main()
