import pandas as pd
import pytest

from src.data.validators import validate_forecasting_data


def test_valid_forecasting_data():
    dataframe = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "product_id": ["P001", "P001"],
            "quantity": [10, 15],
        }
    )

    dataframe["date"] = pd.to_datetime(dataframe["date"])

    validate_forecasting_data(dataframe)


def test_missing_required_column():
    dataframe = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "product_id": ["P001"],
        }
    )

    with pytest.raises(ValueError, match="quantity"):
        validate_forecasting_data(dataframe)


def test_non_numeric_quantity():
    dataframe = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "product_id": ["P001"],
            "quantity": ["ten"],
        }
    )

    with pytest.raises(ValueError, match="numeric"):
        validate_forecasting_data(dataframe)