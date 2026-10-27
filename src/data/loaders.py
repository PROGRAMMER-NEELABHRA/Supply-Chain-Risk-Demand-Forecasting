
"""
Data loading utilities for the Supply Chain Risk and Demand Forecasting project.

This module loads raw datasets from disk without modifying their contents.
"""

from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATACO_DATA_PATH = (
    RAW_DATA_DIR / "dataco" / "DataCoSupplyChainDataset.csv"
)


def load_dataco_data(
    file_path: Path = DATACO_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the raw DataCo Supply Chain dataset from a CSV file.

    Parameters
    ----------
    file_path : Path, optional
        Path to the DataCo CSV file.

    Returns
    -------
    pandas.DataFrame
        The raw DataCo dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the path is not a file, the CSV cannot be read,
        or the loaded dataset contains no rows.
    """
    file_path = Path(file_path)
   

    if not file_path.exists():
        raise FileNotFoundError(
            f"DataCo dataset was not found at: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"The specified path is not a file: {file_path}"
        )

    try:
        data = pd.read_csv(file_path, encoding="latin-1")
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(
            f"Unable to load DataCo dataset from {file_path}: {exc}"
        ) from exc

    if data.empty:
        raise ValueError(
            f"The DataCo dataset contains no rows: {file_path}"
        )

    return data
