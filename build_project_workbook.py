
"""Create an integrated master CSV and an Excel workbook of processed data."""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

OUTPUT_CSV = DATA_DIR / "integrated_supply_chain.csv"
OUTPUT_EXCEL = DATA_DIR / "supply_chain_data_workbook.xlsx"

CORE_FILES = {
    "sales": DATA_DIR / "sales.csv",
    "products": DATA_DIR / "products.csv",
    "logistics": DATA_DIR / "logistics.csv",
}


def load_csv(path: Path) -> pd.DataFrame:
    """Read a non-empty CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(f"CSV file is empty: {path.name}")

    return data


def build_master() -> pd.DataFrame:
    """Join sales, products, and logistics without multiplying sales rows."""
    sales = load_csv(CORE_FILES["sales"])
    products = load_csv(CORE_FILES["products"])
    logistics = load_csv(CORE_FILES["logistics"])

    required_columns = {
        "sales": {"order_item_id", "product_id"},
        "products": {"product_id"},
        "logistics": {"order_item_id"},
    }

    for name, frame in (
        ("sales", sales),
        ("products", products),
        ("logistics", logistics),
    ):
        missing = required_columns[name] - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name}.csv is missing required columns: {sorted(missing)}"
            )

    for frame, key, name in (
        (sales, "order_item_id", "sales.csv"),
        (sales, "product_id", "sales.csv"),
        (products, "product_id", "products.csv"),
        (logistics, "order_item_id", "logistics.csv"),
    ):
        if frame[key].isna().any():
            raise ValueError(f"{name} has missing values in {key}.")

    for frame, key, name in (
        (sales, "order_item_id", "sales.csv"),
        (products, "product_id", "products.csv"),
        (logistics, "order_item_id", "logistics.csv"),
    ):
        if frame[key].duplicated().any():
            raise ValueError(f"{name} has duplicate keys in {key}.")

    expected_rows = len(sales)

    master = sales.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
        indicator="_product_join",
        suffixes=("", "_product"),
    )

    unmatched_products = int((master["_product_join"] == "left_only").sum())
    master = master.drop(columns="_product_join")

    master = master.merge(
        logistics,
        on="order_item_id",
        how="left",
        validate="one_to_one",
        indicator="_logistics_join",
        suffixes=("", "_logistics"),
    )

    unmatched_logistics = int(
        (master["_logistics_join"] == "left_only").sum()
    )
    master = master.drop(columns="_logistics_join")

    if len(master) != expected_rows:
        raise ValueError(
            "The join changed the sales row count. Output was not saved."
        )

    if unmatched_products or unmatched_logistics:
        raise ValueError(
            "Some rows did not match. "
            f"Unmatched products: {unmatched_products}; "
            f"unmatched logistics: {unmatched_logistics}. "
            "Review the source files before sharing."
        )

    master.to_csv(OUTPUT_CSV, index=False)

    print(f"Master rows: {len(master):,}")
    print(f"Master columns: {len(master.columns)}")
    print(f"Saved master CSV: {OUTPUT_CSV}")

    return master


def make_sheet_name(filename: str, used_names: set[str]) -> str:
    """Return a unique, Excel-compatible worksheet name."""
    name = Path(filename).stem
    name = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "Data"

    candidate = name
    counter = 1

    while candidate.lower() in used_names:
        suffix = f"_{counter}"
        candidate = name[: 31 - len(suffix)] + suffix
        counter += 1

    used_names.add(candidate.lower())
    return candidate


def create_workbook(master: pd.DataFrame) -> None:
    """Write the integrated master and every processed CSV to Excel sheets."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    used_names: set[str] = set()
    sheet_count = 0

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="Integrated_Master", index=False)
        used_names.add("integrated_master".lower())
        sheet_count += 1

        for csv_path in csv_files:
            if csv_path.name.lower() == OUTPUT_CSV.name.lower():
                continue

            frame = pd.read_csv(csv_path)
            sheet_name = make_sheet_name(csv_path.name, used_names)

            if len(frame) + 1 > 1_048_576:
                print(
                    f"Skipped {csv_path.name}: exceeds Excel's row limit."
                )
                continue

            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            print(
                f"Added sheet '{sheet_name}': "
                f"{len(frame):,} rows, {len(frame.columns)} columns"
            )
            sheet_count += 1

    print(f"\nWorkbook sheets created: {sheet_count}")
    print(f"Saved Excel workbook: {OUTPUT_EXCEL}")


def main() -> None:
    """Build the integrated CSV and the all-data workbook."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        master = build_master()
        create_workbook(master)
    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
        ImportError,
    ) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
