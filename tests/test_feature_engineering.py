
"""Tests for demand feature engineering."""

import pandas as pd
import pytest

from src.preprocessing.feature_engineering import create_demand_features


def sample_data() -> pd.DataFrame:
    """Build predictable monthly demand for testing."""
    return pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=15, freq="MS"),
            "product_id": [101] * 15,
            "demand_units": list(range(10, 160, 10)),
        }
    )


def test_lag_features_use_previous_months():
    result = create_demand_features(sample_data())
    row = result.loc[result["month"] == pd.Timestamp("2021-03-01")].iloc[0]

    assert row["lag_1"] == 140
    assert row["lag_2"] == 130
    assert row["lag_3"] == 120


def test_rolling_mean_excludes_current_month():
    result = create_demand_features(sample_data())
    row = result.loc[result["month"] == pd.Timestamp("2021-03-01")].iloc[0]

    assert row["rolling_mean_3"] == pytest.approx((120 + 130 + 140) / 3)



def test_features_are_created_separately_for_each_product():
    data = pd.concat(
        [
            sample_data(),
            sample_data().assign(
                product_id=202,
                demand_units=lambda df: df["demand_units"] * 2,
            ),
        ],
        ignore_index=True,
    )

    result = create_demand_features(data)

    first_product = result.loc[
        result["product_id"] == 101
    ].iloc[0]

    second_product = result.loc[
        result["product_id"] == 202
    ].iloc[0]

    assert first_product["lag_1"] == 120
    assert second_product["lag_1"] == 240



def test_duplicate_product_month_is_rejected():
    data = pd.concat([sample_data(), sample_data().iloc[[0]]])

    with pytest.raises(ValueError, match="Duplicate"):
        create_demand_features(data)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        create_demand_features(pd.DataFrame({"month": ["2020-01-01"]}))
