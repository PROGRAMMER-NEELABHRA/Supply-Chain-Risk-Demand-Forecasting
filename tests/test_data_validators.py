import pandas as pd
import pytest

from src.data.validators import (
    validate_forecast_output,
    validate_forecasting_data,
    validate_raw_data,
)


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


def test_validate_raw_data_rejects_duplicate_records():
    dataframe = pd.DataFrame(
        {
            "order_date": ["2017-01-01", "2017-01-01"],
            "product_id": [1, 1],
            "quantity": [10, 10],
        }
    )

    with pytest.raises(ValueError, match="duplicate records"):
        validate_raw_data(dataframe)


def test_validate_raw_data_rejects_duplicate_order_item_ids():
    dataframe = pd.DataFrame(
        {
            "order_date": ["2017-01-01", "2017-01-02"],
            "product_id": [1, 1],
            "quantity": [10, 20],
            "order_item_id": [5, 5],
        }
    )

    with pytest.raises(ValueError, match="duplicate order item IDs"):
        validate_raw_data(dataframe)


@pytest.mark.parametrize("quantity", [0, -1])
def test_validate_raw_data_rejects_nonpositive_quantity(quantity):
    dataframe = pd.DataFrame(
        {
            "order_date": ["2017-01-01"],
            "product_id": [1],
            "quantity": [quantity],
        }
    )

    with pytest.raises(ValueError, match="quantities must be positive"):
        validate_raw_data(dataframe)


def test_validate_forecast_output_accepts_valid_rows():
    forecasts = pd.DataFrame(
        {
            "product_id": [1, 1, 1],
            "forecast_month": pd.to_datetime(
                ["2020-02-01", "2020-03-01", "2020-04-01"]
            ),
            "forecast_demand": [12.5, 14.0, 15.0],
            "model_name": ["Random Forest"] * 3,
            "forecast_method": ["random_forest_recursive"] * 3,
            "forecast_confidence": ["HIGH"] * 3,
            "data_quality_status": ["OBSERVED_HISTORY"] * 3,
            "first_recorded_month": pd.to_datetime(["2019-01-01"] * 3),
            "last_recorded_month": pd.to_datetime(["2020-01-01"] * 3),
            "observed_history_months": [13] * 3,
            "months_since_last_record": [0] * 3,
            "internal_missing_months": [0] * 3,
            "forecast_origin_month": pd.to_datetime(["2020-01-01"] * 3),
            "forecast_horizon_month": [1, 2, 3],
        }
    )

    validate_forecast_output(forecasts)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("forecast_demand", -1, "cannot be negative"),
        ("forecast_demand", float("inf"), "non-finite"),
        ("forecast_month", "2020-02-02", "month starts"),
    ],
)
def test_validate_forecast_output_rejects_invalid_rows(
    column, value, message
):
    forecasts = pd.DataFrame(
        {
            "product_id": [1, 1, 1],
            "forecast_month": pd.to_datetime(
                ["2020-02-01", "2020-03-01", "2020-04-01"]
            ),
            "forecast_demand": [12.5, 14.0, 15.0],
            "model_name": ["Random Forest"] * 3,
            "forecast_method": ["random_forest_recursive"] * 3,
            "forecast_confidence": ["HIGH"] * 3,
            "data_quality_status": ["OBSERVED_HISTORY"] * 3,
            "first_recorded_month": pd.to_datetime(["2019-01-01"] * 3),
            "last_recorded_month": pd.to_datetime(["2020-01-01"] * 3),
            "observed_history_months": [13] * 3,
            "months_since_last_record": [0] * 3,
            "internal_missing_months": [0] * 3,
            "forecast_origin_month": pd.to_datetime(["2020-01-01"] * 3),
            "forecast_horizon_month": [1, 2, 3],
        }
    )
    if column == "forecast_month":
        forecasts.loc[0, column] = pd.to_datetime(value)
    else:
        forecasts.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_forecast_output(forecasts)