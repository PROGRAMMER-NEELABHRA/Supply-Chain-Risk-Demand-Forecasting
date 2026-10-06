import pandas as pd
import pytest

from src.data.inspection import inspect_dataframe


def test_inspect_dataframe():
    dataframe = pd.DataFrame(
        {
            "product_id": ["P001", "P002", "P003"],
            "quantity": [10, 20, 30],
        }
    )

    result = inspect_dataframe(dataframe)

    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["duplicate_rows"] == 0
    assert result["missing_values"]["quantity"] == 0


def test_empty_dataframe():
    dataframe = pd.DataFrame()

    with pytest.raises(ValueError):
        inspect_dataframe(dataframe)