
"""Run basic business-analysis queries against the SQLite database."""

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataco"
    / "supply_chain.db"
)


def run_query(connection, title, query):
    """Execute a SQL query and display its results."""

    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)

    results = pd.read_sql_query(query, connection)
    print(results.to_string(index=False))


def main():
    """Run initial sales and logistics analysis queries."""

    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}. "
            "Run build_database.py first."
        )

    with sqlite3.connect(
        f"{DATABASE_PATH.as_uri()}?mode=ro",
        uri=True,
    ) as connection:

        # Query 1: Monthly demand and sales.
        run_query(
            connection,
            "1. MONTHLY DEMAND AND SALES",
            """
            SELECT
                strftime('%Y-%m', order_date) AS month,
                SUM(quantity) AS units_sold,
                ROUND(SUM(sales), 2) AS total_sales,
                COUNT(DISTINCT order_id) AS total_orders
            FROM sales
            GROUP BY strftime('%Y-%m', order_date)
            ORDER BY month;
            """,
        )

        # Query 2: Best-selling products.
        run_query(
            connection,
            "2. TOP 10 PRODUCTS BY QUANTITY SOLD",
            """
            SELECT
                p.product_id,
                p.product_name,
                SUM(s.quantity) AS units_sold,
                ROUND(SUM(s.sales), 2) AS total_sales
            FROM sales AS s
            JOIN products AS p
                ON s.product_id = p.product_id
            GROUP BY p.product_id, p.product_name
            ORDER BY units_sold DESC
            LIMIT 10;
            """,
        )

        # Query 3: Delivery performance by shipping mode.
        run_query(
            connection,
            "3. DELIVERY PERFORMANCE BY SHIPPING MODE",
            """
            SELECT
                shipping_mode,
                COUNT(*) AS order_items,
                ROUND(AVG(actual_shipping_days), 2)
                    AS avg_actual_days,
                ROUND(AVG(scheduled_shipping_days), 2)
                    AS avg_scheduled_days,
                SUM(
                    CASE
                        WHEN LOWER(TRIM(delivery_status))
                             = 'late delivery'
                        THEN 1
                        ELSE 0
                    END
                ) AS late_delivery_items
            FROM logistics
            GROUP BY shipping_mode
            ORDER BY order_items DESC;
            """,
        )

    print("\nSQL checks completed successfully.")


if __name__ == "__main__":
    main()
