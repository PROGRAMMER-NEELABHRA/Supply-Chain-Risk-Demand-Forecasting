
"""
Monthly demand aggregation utilities.

Aggregates order-item sales data into monthly product-level demand.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"
SALES_DATA_PATH = PROCESSED_DATA_DIR / "sales.csv"
MONTHLY_DEMAND_PATH = PROCESSED_DATA_DIR / "monthly_demand.csv"



def load_sales_data(file_path: Path = SALES_DATA_PATH) -> pd.DataFrame:
    """Load the processed sales dataset and validate required columns."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Sales dataset not found: {file_path}")

    data = pd.read_csv(file_path)

    required_columns = {"order_date", "product_id", "quantity"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Sales dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Sales dataset contains no records.")

    data["order_date"] = pd.to_datetime(
        data["order_date"], errors="coerce"
    )
    data["quantity"] = pd.to_numeric(
        data["quantity"], errors="coerce"
    )

    if data["order_date"].isna().any():
        raise ValueError("Sales data contains invalid order dates.")

    if data["quantity"].isna().any() or (data["quantity"] <= 0).any():
        raise ValueError("Sales data contains invalid quantities.")

    if data["product_id"].isna().any():
        raise ValueError("Sales data contains missing product IDs.")

    return data


def build_monthly_demand(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quantities and order activity by month and product."""
    required_columns = {"order_date", "product_id", "quantity"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Input data is missing columns: {sorted(missing_columns)}"
        )

    result = data.copy()
    result["month"] = result["order_date"].dt.to_period("M").astype(str)

    monthly = (
        result.groupby(["month", "product_id"], as_index=False)
        .agg(
            demand_units=("quantity", "sum"),
            order_item_rows=("quantity", "size"),
            active_days=("order_date", lambda values: values.dt.date.nunique()),
        )
        .sort_values(["month", "product_id"])
        .reset_index(drop=True)
    )

    return monthly


def save_monthly_demand(
    monthly_data: pd.DataFrame,
    output_path: Path = MONTHLY_DEMAND_PATH,
) -> None:
    """Save monthly product demand to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_data.to_csv(output_path, index=False)


def main() -> None:
    """Build and save monthly product-level demand."""
    sales = load_sales_data()
    monthly = build_monthly_demand(sales)
    save_monthly_demand(monthly)

    print("Monthly demand dataset created.")
    print(f"Source sales rows: {len(sales):,}")
    print(f"Monthly product records: {len(monthly):,}")
    print(f"Products represented: {monthly['product_id'].nunique():,}")
    print(
        f"Date range: {monthly['month'].min()} "
        f"to {monthly['month'].max()}"
    )
    print(f"Output: {MONTHLY_DEMAND_PATH}")


if __name__ == "__main__":
    main()
