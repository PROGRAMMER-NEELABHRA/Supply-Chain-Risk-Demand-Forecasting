
"""Check relationships between processed DataCo datasets."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"


def main():
    """Validate dataset keys and product relationships."""

    sales = pd.read_csv(DATA_DIR / "sales.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    logistics = pd.read_csv(DATA_DIR / "logistics.csv")

    print("=== DATASET ROW COUNTS ===")
    print(f"Sales: {len(sales):,}")
    print(f"Products: {len(products):,}")
    print(f"Logistics: {len(logistics):,}")

    print("\n=== KEY UNIQUENESS ===")
    print(
        "Sales order_item_id unique:",
        sales["order_item_id"].is_unique,
    )
    print(
        "Product product_id unique:",
        products["product_id"].is_unique,
    )
    print(
        "Logistics order_item_id unique:",
        logistics["order_item_id"].is_unique,
    )

    # Check that every sales product ID exists in the product dataset.
    sales_product_ids = set(sales["product_id"].dropna())
    master_product_ids = set(products["product_id"].dropna())

    missing_products = sales_product_ids - master_product_ids

    print("\n=== SALES-TO-PRODUCT RELATIONSHIP ===")
    print(f"Distinct product IDs in sales: {len(sales_product_ids):,}")
    print(f"Products in product dataset: {len(master_product_ids):,}")
    print(f"Sales product IDs absent from product dataset: {len(missing_products):,}")

    if missing_products:
        print("Example missing product IDs:", list(missing_products)[:10])

    # Check that sales and logistics have matching order-item IDs.
    sales_item_ids = set(sales["order_item_id"].dropna())
    logistics_item_ids = set(logistics["order_item_id"].dropna())

    print("\n=== SALES-TO-LOGISTICS RELATIONSHIP ===")
    print(
        "Sales items missing from logistics:",
        len(sales_item_ids - logistics_item_ids),
    )
    print(
        "Logistics items missing from sales:",
        len(logistics_item_ids - sales_item_ids),
    )

    # Check whether the product join preserves the sales row count.
    merged = sales.merge(
        products[["product_id"]],
        on="product_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    print("\n=== JOIN VALIDATION ===")
    print(f"Sales rows before join: {len(sales):,}")
    print(f"Rows after product join: {len(merged):,}")
    print(
        "Rows without a matching product:",
        int((merged["_merge"] == "left_only").sum()),
    )

    if len(merged) != len(sales):
        print("WARNING: The product join changed the sales row count.")
    else:
        print("PASS: The product join preserved the sales row count.")


if __name__ == "__main__":
    main()
