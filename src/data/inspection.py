"""
Utilities for inspecting incoming supply-chain datasets.
"""

from pathlib import Path

import pandas as pd


def inspect_dataframe(df: pd.DataFrame) -> dict:
    """
    Generate basic structural and data-quality information.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to inspect.

    Returns
    -------
    dict
        Dataset inspection results.
    """
    if df.empty:
        raise ValueError("Cannot inspect an empty DataFrame.")

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df.describe(
            include="number"
        ).to_dict(),
    }


def inspect_csv(file_path: Path) -> dict:
    """
    Load a CSV file and inspect its structure.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    dict
        Dataset inspection results.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("Expected a CSV file.")

    dataframe = pd.read_csv(file_path)

    return inspect_dataframe(dataframe)