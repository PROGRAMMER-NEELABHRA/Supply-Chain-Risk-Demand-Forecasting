
"""Test the DataCo sales, product, and logistics data pipeline."""

from pathlib import Path

from src.data.loaders import load_dataco_data
from src.data.cleaning import (
    prepare_sales_data,
    validate_sales_data,
)
from src.data.products import prepare_products_data
from src.data.logistics import prepare_logistics_data


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"


def main():
    """Prepare, validate, and save processed datasets."""

    # Load the raw dataset once.
    raw_data = load_dataco_data()

    # Prepare and validate sales data.
    sales_data = prepare_sales_data(raw_data)
    validate_sales_data(sales_data)

    # Prepare product and logistics datasets.
    products_data = prepare_products_data(raw_data)
    logistics_data = prepare_logistics_data(raw_data)

    # Create the output directory if necessary.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all processed datasets.
    sales_data.to_csv(OUTPUT_DIR / "sales.csv", index=False)
    products_data.to_csv(OUTPUT_DIR / "products.csv", index=False)
    logistics_data.to_csv(OUTPUT_DIR / "logistics.csv", index=False)

    # Display sales summary.
    print("\nSales data saved successfully.")
    print(f"Rows: {len(sales_data):,}")
    print(f"Columns: {len(sales_data.columns)}")

    # Display product summary.
    print("\nProduct data saved successfully.")
    print(f"Unique products: {len(products_data):,}")
    print(f"Columns: {len(products_data.columns)}")

    print("\nProduct dataset missing values:")
    print(products_data.isna().sum())

    # Display logistics summary.
    print("\nLogistics data saved successfully.")
    print(f"Rows: {len(logistics_data):,}")
    print(f"Columns: {len(logistics_data.columns)}")

    print("\nLogistics dataset missing values:")
    print(logistics_data.isna().sum())

    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
