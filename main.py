"""
Application entry point for the supply-chain analytics pipeline.
"""

import argparse
from pathlib import Path

from src.data.inspection import inspect_csv


def main() -> None:
    """Run the application."""
    parser = argparse.ArgumentParser(
        description="Supply-chain dataset inspection utility."
    )

    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the CSV dataset to inspect.",
    )

    args = parser.parse_args()

    results = inspect_csv(args.file)

    print("\nDataset Inspection")
    print("------------------")
    print(f"Rows: {results['row_count']}")
    print(f"Columns: {results['column_count']}")
    print(f"Duplicate rows: {results['duplicate_rows']}")

    print("\nColumns:")
    for column in results["columns"]:
        print(f"  - {column}")

    print("\nMissing values:")
    for column, count in results["missing_values"].items():
        print(f"  - {column}: {count}")


if __name__ == "__main__":
    main()