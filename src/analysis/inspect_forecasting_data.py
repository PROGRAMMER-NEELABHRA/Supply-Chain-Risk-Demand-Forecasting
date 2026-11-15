import pandas as pd

from config import DATACO_RAW_DATA_FILE


DATA_FILE = DATACO_RAW_DATA_FILE


def main() -> None:
    """Report dataset structure and demand-related data quality."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            f"Place {DATA_FILE.name} in data/raw/."
        )

    df = pd.read_csv(DATA_FILE)

    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n=== COLUMN NAMES AND TYPES ===")
    print(df.dtypes.to_string())

    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).to_string()
          or "No missing values.")

    required = {"order_date", "product_id", "quantity"}
    missing_columns = required - set(df.columns)

    print("\n=== FORECASTING READINESS ===")
    if missing_columns:
        print(f"Missing expected columns: {sorted(missing_columns)}")
        return

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce", dayfirst=True
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    print(f"Invalid dates: {df['order_date'].isna().sum():,}")
    print(f"Invalid quantities: {df['quantity'].isna().sum():,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    print(f"Total quantity: {df['quantity'].sum():,.0f}")

    print("\n=== RECORDS PER PRODUCT ===")
    print(df.groupby("product_id").size().describe().to_string())


if __name__ == "__main__":
    main()
