
import pandas as pd
import pytest

from src.data.loaders import load_csv
from src.data.validators import (
    validate_monthly_demand,
    validate_raw_data,
)


def sample_monthly_data():
    return pd.DataFrame({
        "month": pd.to_datetime(["2017-01-01", "2017-02-01"]),
        "product_id": [1, 1],
        "demand_units": [10, 15],
    })


def test_load_csv_success(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"value": [1, 2]}).to_csv(path, index=False)

    result = load_csv(path)

    assert len(result) == 2


def test_load_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv("missing.csv")


def test_validate_monthly_demand_success():
    validate_monthly_demand(sample_monthly_data())


def test_validate_monthly_demand_rejects_duplicates():
    df = sample_monthly_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_monthly_demand(df)


def test_validate_monthly_demand_rejects_negative_demand():
    df = sample_monthly_data()
    df.loc[0, "demand_units"] = -5

    with pytest.raises(ValueError, match="negative"):
        validate_monthly_demand(df)


def test_validate_raw_data_success():
    df = pd.DataFrame({
        "order_date": ["2017-01-01", "2017-02-01"],
        "product_id": [1, 1],
        "quantity": [10, 15],
    })

    validate_raw_data(df)
