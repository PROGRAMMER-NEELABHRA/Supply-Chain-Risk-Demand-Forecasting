
"""Validate the integrated supply chain dataset before sharing."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

MASTER_FILE = DATA_DIR / "integrated_supply_chain.csv"
SALES_FILE = DATA_DIR / "sales.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
LOGISTICS_FILE = DATA_DIR / "logistics.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file after checking that it exists."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def main() -> None:
    """Check row counts, join keys, missing values, and duplicate IDs."""
    master = load_csv(MASTER_FILE)
    sales = load_csv(SALES_FILE)
    products = load_csv(PRODUCTS_FILE)
    logistics = load_csv(LOGISTICS_FILE)

    print("=== INTEGRATED DATASET VALIDATION ===")
    print(f"Master rows: {len(master):,}")
    print(f"Master columns: {len(master.columns)}")
    print(f"Sales rows: {len(sales):,}")
    print(f"Products rows: {len(products):,}")
    print(f"Logistics rows: {len(logistics):,}")

    checks = {
        "Master row count matches sales": len(master) == len(sales),
        "Master order_item_id is unique": (
            "order_item_id" in master.columns
            and master["order_item_id"].is_unique
        ),
        "Master has no missing product IDs": (
            "product_id" in master.columns
            and master["product_id"].notna().all()
        ),
        "Master has no duplicate rows": not master.duplicated().any(),
        "Every sales item exists in master": (
            set(sales["order_item_id"]) == set(master["order_item_id"])
        ),
        "Product IDs match the product table": (
            set(master["product_id"]) == set(products["product_id"])
        ),
        "Every sales item has logistics": (
            set(master["order_item_id"]) == set(logistics["order_item_id"])
        ),
    }

    print("\n=== CHECK RESULTS ===")
    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")

    print("\n=== MISSING VALUES ===")
    missing = master.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values.")
    else:
        print(missing.to_string())

    if all(checks.values()):
        print("\nPASS: Structural validation succeeded.")
    else:
        print("\nWARNING: Investigate failed checks before sharing.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error
