
"""Reusable data-loading utilities."""

from pathlib import Path

import pandas as pd

from src.data.validators import validate_monthly_demand


def load_csv(
    file_path: str | Path,
    *,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file with basic validation."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file: {path}")

    try:
        dataframe = pd.read_csv(
            path,
            parse_dates=parse_dates,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}") from exc

    if dataframe.empty:
        raise ValueError(f"CSV contains no records: {path}")

    return dataframe


def load_monthly_demand(file_path: str | Path) -> pd.DataFrame:
    """Load, validate, and sort observed monthly product demand."""
    dataframe = load_csv(file_path, parse_dates=["month"])
    validate_monthly_demand(dataframe)

    return dataframe.sort_values(
        ["product_id", "month"]
    ).reset_index(drop=True)
