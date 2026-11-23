
import pandas as pd
import pytest

from config import DATACO_RAW_DATA_FILE
from src.analysis import build_monthly_demand
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


def test_dataco_source_is_explicitly_configured():
    assert DATACO_RAW_DATA_FILE.name == (
        "integrated_supply_chain_until_aug2017.csv"
    )
    assert build_monthly_demand.INPUT_FILE == DATACO_RAW_DATA_FILE


def test_source_metadata_reports_august_2017_coverage():
    raw = pd.DataFrame(
        {
            "order_date": [
                "2015-01-01",
                "2017-08-30",
                "2017-08-31",
            ],
            "product_id": [1, 2, 2],
            "quantity": [1, 2, 3],
        }
    )

    metadata = build_monthly_demand.build_source_metadata(
        raw,
        DATACO_RAW_DATA_FILE,
    )

    assert metadata == {
        "source_filename": "integrated_supply_chain_until_aug2017.csv",
        "source_row_count": 3,
        "source_product_count": 2,
        "source_first_order_date": "2015-01-01",
        "source_last_order_date": "2017-08-31",
    }


def test_aggregation_sums_quantity_by_product_and_calendar_month():
    raw = pd.DataFrame(
        {
            "order_date": [
                "2017-01-02",
                "2017-01-31",
                "2017-01-15",
                "2017-02-01",
            ],
            "product_id": [1, 1, 2, 1],
            "quantity": [2, 3, 7, 5],
        }
    )

    monthly = build_monthly_demand.aggregate_monthly_demand(raw)
    demand = monthly.set_index(["product_id", "month"])["demand_units"]

    assert demand.loc[(1, pd.Timestamp("2017-01-01"))] == 5
    assert demand.loc[(2, pd.Timestamp("2017-01-01"))] == 7
    assert demand.loc[(1, pd.Timestamp("2017-02-01"))] == 5
