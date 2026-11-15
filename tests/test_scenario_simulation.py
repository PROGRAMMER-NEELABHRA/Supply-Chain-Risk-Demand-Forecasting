import numpy as np
import pandas as pd
import pytest

from src.scenarios.simulation import (
    aggregate_portfolio_scenarios,
    build_demand_scenarios,
)


def test_scenario_calculations_match_hand_computed_values():
    forecasts = pd.DataFrame(
        {
            "product_id": [1, 1],
            "forecast_month": ["2017-09-01", "2017-10-01"],
            "forecast_demand": [100, 0],
        }
    )

    result = build_demand_scenarios(
        forecasts,
        optimistic_multiplier=1.2,
        pessimistic_multiplier=0.8,
        user_multiplier=1.1,
    )

    assert result["optimistic_demand"].tolist() == [120, 0]
    assert result["pessimistic_delta_units"].tolist() == [-20, 0]
    assert result.loc[0, "user_scenario_delta_pct"] == pytest.approx(10)
    assert np.isnan(result.loc[1, "user_scenario_delta_pct"])


def test_portfolio_aggregates_products_before_percentage_deltas():
    forecasts = pd.DataFrame(
        {
            "product_id": [1, 2],
            "forecast_month": ["2017-09-01", "2017-09-01"],
            "forecast_demand": [100, 300],
        }
    )
    products = build_demand_scenarios(
        forecasts,
        optimistic_multiplier=1.2,
        pessimistic_multiplier=0.8,
        user_multiplier=1.1,
    )

    portfolio = aggregate_portfolio_scenarios(products).iloc[0]

    assert portfolio["baseline_demand"] == 400
    assert portfolio["optimistic_demand"] == 480
    assert portfolio["optimistic_delta_units"] == pytest.approx(80)
    assert portfolio["optimistic_delta_pct"] == pytest.approx(20)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"optimistic_multiplier": 0}, "greater than zero"),
        ({"optimistic_multiplier": 0.9}, "at least 1"),
        ({"pessimistic_multiplier": 1.1}, "at most 1"),
        ({"user_multiplier": float("nan")}, "finite"),
    ],
)
def test_invalid_scenario_multipliers_are_rejected(kwargs, message):
    forecasts = pd.DataFrame(
        {
            "product_id": [1],
            "forecast_month": ["2017-09-01"],
            "forecast_demand": [10],
        }
    )

    with pytest.raises(ValueError, match=message):
        build_demand_scenarios(forecasts, **kwargs)


def test_scenario_rejects_negative_demand_duplicate_rows_and_unknown_product():
    forecasts = pd.DataFrame(
        {
            "product_id": [1, 1],
            "forecast_month": ["2017-09-01", "2017-09-01"],
            "forecast_demand": [10, 11],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_demand_scenarios(forecasts)

    forecasts.loc[1, "forecast_month"] = "2017-10-01"
    forecasts.loc[1, "forecast_demand"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        build_demand_scenarios(forecasts)

    with pytest.raises(ValueError, match="No forecast rows"):
        build_demand_scenarios(forecasts.assign(forecast_demand=1), product_id=99)
