
"""Build and validate the local SQLite supply chain database."""

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"
DATABASE_PATH = DATA_DIR / "supply_chain.db"


TABLE_COLUMNS = {
    "products": [
        "product_id",
        "product_category_id",
        "product_name",
        "product_price",
        "product_status",
        "category_id",
        "category_name",
        "department_id",
        "department_name",
    ],
    "sales": [
        "order_id",
        "order_item_id",
        "order_date",
        "product_id",
        "customer_id",
        "quantity",
        "unit_price",
        "discount",
        "discount_rate",
        "sales",
        "order_item_total",
        "order_profit",
        "order_region",
    ],
    "logistics": [
        "order_id",
        "order_item_id",
        "order_date",
        "shipping_date",
        "actual_shipping_days",
        "scheduled_shipping_days",
        "delivery_status",
        "late_delivery_risk",
        "shipping_mode",
        "market",
        "order_region",
        "order_country",
        "order_state",
        "order_city",
    ],
}


SCHEMA = """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_category_id INTEGER NOT NULL,
    product_name TEXT NOT NULL,
    product_price REAL NOT NULL,
    product_status INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    category_name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    department_name TEXT NOT NULL
);

CREATE TABLE sales (
    order_id INTEGER NOT NULL,
    order_item_id INTEGER PRIMARY KEY,
    order_date TEXT NOT NULL,
    product_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price REAL NOT NULL CHECK (unit_price >= 0),
    discount REAL NOT NULL,
    discount_rate REAL NOT NULL,
    sales REAL NOT NULL,
    order_item_total REAL NOT NULL,
    order_profit REAL NOT NULL,
    order_region TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE logistics (
    order_id INTEGER NOT NULL,
    order_item_id INTEGER PRIMARY KEY,
    order_date TEXT NOT NULL,
    shipping_date TEXT NOT NULL,
    actual_shipping_days INTEGER NOT NULL
        CHECK (actual_shipping_days >= 0),
    scheduled_shipping_days INTEGER NOT NULL
        CHECK (scheduled_shipping_days >= 0),
    delivery_status TEXT NOT NULL,
    late_delivery_risk INTEGER NOT NULL,
    shipping_mode TEXT NOT NULL,
    market TEXT NOT NULL,
    order_region TEXT NOT NULL,
    order_country TEXT NOT NULL,
    order_state TEXT NOT NULL,
    order_city TEXT NOT NULL,
    FOREIGN KEY (order_item_id) REFERENCES sales(order_item_id)
);
"""


def load_csv(connection, table_name):
    """Load a processed CSV into its corresponding SQL table."""

    file_path = DATA_DIR / f"{table_name}.csv"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required processed dataset not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    expected_columns = TABLE_COLUMNS[table_name]

    if list(data.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns in {file_path.name}. "
            f"Expected {expected_columns}; found {list(data.columns)}"
        )

    if data.empty:
        raise ValueError(f"Dataset contains no rows: {file_path}")

    if data.isna().any().any():
        raise ValueError(
            f"Missing values found in {file_path.name}. "
            "Resolve them before database loading."
        )

    placeholders = ", ".join(["?"] * len(expected_columns))
    columns = ", ".join(f'"{column}"' for column in expected_columns)

    insert_sql = (
        f'INSERT INTO "{table_name}" ({columns}) '
        f"VALUES ({placeholders})"
    )

    # Convert NumPy scalar values into native Python values for SQLite.
    rows = (
        tuple(
            value.item() if hasattr(value, "item") else value
            for value in row
        )
        for row in data.itertuples(index=False, name=None)
    )

    connection.executemany(insert_sql, rows)
    print(f"Loaded {len(data):,} rows into {table_name}.")


def validate_database(connection):
    """Check row counts, foreign keys, and primary-key uniqueness."""

    print("\n=== DATABASE VALIDATION ===")

    expected_counts = {
        "products": 118,
        "sales": 180519,
        "logistics": 180519,
    }

    for table_name, expected_count in expected_counts.items():
        actual_count = connection.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()[0]

        print(f"{table_name}: {actual_count:,} rows")

        if actual_count != expected_count:
            raise ValueError(
                f"{table_name} has {actual_count} rows; "
                f"expected {expected_count}. "
                "Update the expected count if the source data changed."
            )

    foreign_key_errors = connection.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    if foreign_key_errors:
        raise ValueError(
            f"Foreign key validation failed: {foreign_key_errors[:10]}"
        )

    print("Foreign key checks: PASS")

    print("\nDatabase tables:")
    tables = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    for (table_name,) in tables:
        print(f"- {table_name}")


def main():
    """Build the database and validate its contents."""

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {DATA_DIR}"
        )

    if DATABASE_PATH.exists():
        raise FileExistsError(
            f"Database already exists: {DATABASE_PATH}\n"
            "Back it up or deliberately remove it before rebuilding."
        )

    connection = None

    try:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.execute("PRAGMA foreign_keys = ON")

        connection.executescript(SCHEMA)

        # Parent table first, followed by dependent tables.
        load_csv(connection, "products")
        load_csv(connection, "sales")
        load_csv(connection, "logistics")

        validate_database(connection)
        connection.commit()

        print("\nDatabase created successfully.")
        print(f"Location: {DATABASE_PATH}")

    except (sqlite3.Error, OSError, ValueError) as exc:
        if connection is not None:
            connection.rollback()

        if DATABASE_PATH.exists():
            connection.close()
            connection = None
            DATABASE_PATH.unlink()

        raise RuntimeError(
            f"Database build failed: {exc}"
        ) from exc

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
