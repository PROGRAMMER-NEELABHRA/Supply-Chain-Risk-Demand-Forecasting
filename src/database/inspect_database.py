
"""Inspect the existing SQLite database without modifying it."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "processed"
    / "dataco"
    / "supply_chain.db"
)


def main():
    if not DATABASE_FILE.exists():
        print(f"Database not found: {DATABASE_FILE}")
        return

    with sqlite3.connect(DATABASE_FILE) as connection:
        for table in ("products", "sales", "logistics"):
            print(f"\n--- {table.upper()} ---")
            columns = connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()

            for column in columns:
                print(f"{column[1]} ({column[2]})")


if __name__ == "__main__":
    main()
