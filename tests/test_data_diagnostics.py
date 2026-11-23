import pandas as pd
import pytest

from src.analysis.check_product_lifecycle import observed_month_range
from src.analysis.investigate_product_anomaly import (
    latest_observed_product_month,
)


def test_observed_month_range_is_bounded_by_available_source_dates():
    dates = pd.Series(
        pd.to_datetime(["2015-01-02", "2017-08-31", None])
    )

    months = observed_month_range(dates)

    assert months[0] == pd.Period("2015-01", freq="M")
    assert months[-1] == pd.Period("2017-08", freq="M")
    assert pd.Period("2018-01", freq="M") not in months


def test_observed_month_range_rejects_all_invalid_dates():
    with pytest.raises(ValueError, match="No valid order dates"):
        observed_month_range(pd.Series(["invalid", None]))


def test_anomaly_month_uses_latest_observed_month_not_fixed_future_month():
    months = pd.Series(
        pd.PeriodIndex(["2017-07", "2017-08", None], freq="M")
    )

    assert latest_observed_product_month(months) == pd.Period(
        "2017-08",
        freq="M",
    )


def test_anomaly_month_rejects_no_observations():
    with pytest.raises(ValueError, match="No valid product observations"):
        latest_observed_product_month(pd.Series([], dtype="period[M]"))
