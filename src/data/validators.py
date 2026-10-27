
"""
Validation and profiling utilities for the Supply Chain project.

These functions inspect datasets and report potential data-quality
issues without modifying the original data.
"""

import pandas as pd


def profile_dataset(data: pd.DataFrame) -> dict:
    """
    Generate a basic data-quality profile for a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    dict
        Summary of dataset dimensions, data types, missing values,
        and duplicate rows.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Cannot profile an empty dataset.")

    profile = {
        "row_count": len(data),
        "column_count": len(data.columns),
        "data_types": data.dtypes.astype(str).to_dict(),
        "missing_values": data.isna().sum().to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
    }

    return profile


def print_dataset_profile(data: pd.DataFrame) -> None:
    """
    Print a readable summary of dataset quality.
    """
    profile = profile_dataset(data)

    print("\n========== DATASET PROFILE ==========")
    print(f"Rows: {profile['row_count']:,}")
    print(f"Columns: {profile['column_count']}")
    print(f"Duplicate rows: {profile['duplicate_rows']:,}")

    print("\nMissing values by column:")
    missing_values = profile["missing_values"]
    columns_with_missing = {
        column: count
        for column, count in missing_values.items()
        if count > 0
    }

    if columns_with_missing:
        for column, count in columns_with_missing.items():
            percentage = count / profile["row_count"] * 100
            print(f"- {column}: {count:,} ({percentage:.2f}%)")
    else:
        print("No missing values found.")

    print("\nData types:")
    for column, data_type in profile["data_types"].items():
        print(f"- {column}: {data_type}")
