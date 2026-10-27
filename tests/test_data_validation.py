
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"


def test_processed_datasets_exist():
    """Check that the three processed datasets are available."""
    for filename in ("sales.csv", "products.csv", "logistics.csv"):
        assert (DATA_DIR / filename).is_file(), (
            f"Missing processed dataset: {filename}"
        )


def test_sales_order_item_ids_are_unique():
    """Each sales row should represent a unique order item."""
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    assert "order_item_id" in sales.columns
    assert sales["order_item_id"].notna().all()
    assert sales["order_item_id"].is_unique


def test_products_have_unique_product_ids():
    """The product dimension should have one row per product."""
    products = pd.read_csv(DATA_DIR / "products.csv")
    assert "product_id" in products.columns
    assert products["product_id"].notna().all()
    assert products["product_id"].is_unique


def test_sales_products_and_logistics_relationships():
    """Every sales item should have a product and logistics record."""
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    logistics = pd.read_csv(DATA_DIR / "logistics.csv")

    assert "product_id" in sales.columns
    assert "product_id" in products.columns
    assert "order_item_id" in sales.columns
    assert "order_item_id" in logistics.columns

    assert sales["product_id"].isin(products["product_id"]).all()
    assert sales["order_item_id"].is_unique
    assert logistics["order_item_id"].is_unique
    assert set(sales["order_item_id"]) == set(logistics["order_item_id"])
