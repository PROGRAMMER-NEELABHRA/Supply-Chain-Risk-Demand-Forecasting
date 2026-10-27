
"""Audit available fields for supply-chain risk analysis."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

INPUT_FILES = {
    "integrated_master": DATA_DIR / "integrated_supply_chain.csv",
    "sales": DATA_DIR / "sales.csv",
    "products": DATA_DIR / "products.csv",
    "logistics": DATA_DIR / "logistics.csv",
}

RISK_KEYWORDS = (
    "supplier", "vendor", "inventory", "stock", "warehouse",
    "lead_time", "lead time", "procurement", "purchase",
    "reorder", "safety_stock", "safety stock", "risk",
    "shipping", "delivery", "late", "delay", "quantity",
    "product", "order", "date", "region", "country",
    "department", "category",
)


def audit_file(label: str, path: Path) -> None:
    """Print dataset dimensions, columns, and relevant field names."""
    print(f"\n{'=' * 72}")
    print(f"DATASET: {label}")
    print(f"FILE: {path}")

    if not path.is_file():
        print("STATUS: FILE NOT FOUND")
        return

    try:
        sample = pd.read_csv(path, nrows=5)
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        print(f"STATUS: COULD NOT READ FILE: {exc}")
        return

    print(f"Columns: {len(header)}")
    print("Available fields:")
    for column in header:
        print(f"  - {column}")

    relevant = [
        column for column in header
        if any(word in column.lower() for word in RISK_KEYWORDS)
    ]

    print("\nFields potentially relevant to risk analysis:")
    if relevant:
        for column in relevant:
            print(f"  - {column}")
    else:
        print("  No keyword matches found.")

    print("\nSample data types:")
    print(sample.dtypes.to_string())


def main() -> None:
    """Audit all configured processed datasets."""
    for label, path in INPUT_FILES.items():
        audit_file(label, path)

    print("\nAudit complete. No source data was modified.")


if __name__ == "__main__":
    main()
