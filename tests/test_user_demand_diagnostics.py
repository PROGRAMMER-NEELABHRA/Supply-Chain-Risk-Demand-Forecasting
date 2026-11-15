import numpy as np
import pandas as pd
import pytest

from src.data.user_demand import prepare_user_demand_data
from src.forecasting.user_demand_diagnostics import (
    DemandPatternThresholds,
    diagnose_demand_patterns,
)
from src.preprocessing.user_demand_preparation import (
    ForecastReadyDemandData,
    prepare_forecast_demand,
)


def diagnose(values, *, frequency="monthly", product_column=None, dates=None):
    if dates is None:
        dates = pd.date_range(
            "2020-01-01",
            periods=len(values),
            freq={"daily": "D", "weekly": "W-MON", "monthly": "MS"}[frequency],
        )
    source = pd.DataFrame({"date": dates, "demand": values})
    prepared = prepare_user_demand_data(
        source,
        date_column="date",
        demand_column="demand",
        product_column=product_column,
        frequency=frequency,
    )
    return diagnose_demand_patterns(prepare_forecast_demand(prepared))


def only_series(result):
    assert len(result.series) == 1
    return result.series[0]


def test_regular_demand_reports_metrics_and_regular_classification():
    result = only_series(diagnose([10, 11, 9, 10, 11, 9, 10, 10]))

    assert result.primary_classification == "regular"
    assert "regular" in result.classifications
    assert result.mean_demand == 10
    assert result.standard_deviation < 1
    assert result.coefficient_of_variation < 0.1
    assert result.coverage_ratio == 1
    assert result.zero_demand_percentage == 0


def test_intermittent_demand_is_identified_from_eligible_zero_periods():
    result = only_series(diagnose([0, 0, 5, 0, 0, 8, 0, 4]))

    assert result.primary_classification == "intermittent"
    assert result.intermittent_behavior
    assert result.zero_demand_percentage == 62.5
    assert result.observed_zero_periods == 5


def test_erratic_demand_is_identified_when_zero_fraction_is_low():
    result = only_series(diagnose([1, 10, 2, 12, 1, 9, 2, 11]))

    assert result.primary_classification == "erratic"
    assert result.coefficient_of_variation > 0.5
    assert not result.intermittent_behavior


def test_lumpy_demand_requires_zeros_and_variable_positive_sizes():
    result = only_series(
        diagnose([0, 0, 1, 0, 0, 100, 0, 0, 2, 0, 0, 80])
    )

    assert result.primary_classification == "lumpy"
    assert result.intermittent_behavior
    assert "lumpy" in result.classifications


def test_seasonality_requires_sufficient_complete_history():
    seasonal_pattern = [10, 20, 15, 25, 30, 18, 12, 22, 28, 16, 24, 35]
    values = seasonal_pattern * 4
    result = only_series(diagnose(values))

    assert result.seasonal_period == 12
    assert result.seasonality_evidence
    assert "seasonal" in result.classifications
    assert "regular" in result.classifications
    assert result.seasonal_strength >= 0.6


def test_insufficient_history_is_reported_without_forcing_a_pattern():
    result = only_series(diagnose([3, 5, 4]))

    assert result.primary_classification == "insufficient_history"
    assert "insufficient_history" in result.classifications
    assert result.mean_demand == 4
    assert result.standard_deviation is not None
    assert not result.seasonality_evidence


def test_unknown_periods_excluded_but_confirmed_zero_periods_counted():
    source = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "demand": [10, 0, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        }
    )
    source = source.loc[source["date"] != pd.Timestamp("2020-03-01")].reset_index(
        drop=True
    )

    def run(confirm):
        prepared = prepare_user_demand_data(
            source,
            date_column="date",
            demand_column="demand",
            frequency="monthly",
            complete_source_coverage=confirm,
            confirm_missing_periods_are_zero=confirm,
        )
        return only_series(
            diagnose_demand_patterns(prepare_forecast_demand(prepared))
        )

    unknown_result = run(False)
    confirmed_result = run(True)

    assert unknown_result.unknown_periods == 1
    assert unknown_result.confirmed_zero_periods == 0
    assert unknown_result.eligible_periods == 11
    assert unknown_result.mean_demand == pytest.approx(730 / 11)
    assert not unknown_result.seasonality_evidence
    assert confirmed_result.unknown_periods == 0
    assert confirmed_result.confirmed_zero_periods == 1
    assert confirmed_result.eligible_periods == 12
    assert confirmed_result.mean_demand == pytest.approx(730 / 12)


def test_multiple_products_are_classified_independently():
    values = [10 + (index % 2) for index in range(12)]
    source = pd.DataFrame(
        {
            "date": list(pd.date_range("2020-01-01", periods=12, freq="MS"))
            + list(pd.date_range("2020-01-01", periods=3, freq="MS")),
            "demand": values + [0, 0, 5],
            "sku": ["regular"] * 12 + ["short"] * 3,
        }
    )
    prepared = prepare_user_demand_data(
        source,
        date_column="date",
        demand_column="demand",
        product_column="sku",
        frequency="monthly",
    )
    result = diagnose_demand_patterns(prepare_forecast_demand(prepared))
    by_product = {series.product_id: series for series in result.series}

    assert by_product["regular"].primary_classification == "regular"
    assert "insufficient_history" in by_product["short"].classifications
    assert by_product["short"].unknown_periods == 0
    assert by_product["short"].history_length == 3
    assert by_product["short"].primary_classification == "insufficient_history"


def test_all_zero_series_has_undefined_cv_and_intermittent_label():
    result = only_series(diagnose([0] * 12))

    assert result.mean_demand == 0
    assert result.standard_deviation == 0
    assert result.coefficient_of_variation is None
    assert result.zero_demand_percentage == 100
    assert result.intermittent_behavior
    assert result.primary_classification == "intermittent"
    assert any("All eligible demand is zero" in message for message in result.explanations)


def test_constant_positive_series_is_regular_with_no_cv_variability():
    result = only_series(diagnose([5] * 12))

    assert result.primary_classification == "regular"
    assert result.coefficient_of_variation == 0
    assert not result.trend_evidence
    assert not result.seasonality_evidence


def test_trend_evidence_reports_increasing_direction():
    result = only_series(diagnose(list(range(1, 13))))

    assert result.trend_evidence
    assert result.trend_direction == "increasing"
    assert result.trend_slope_per_period is not None
    assert result.trend_r_squared == pytest.approx(1)


def test_diagnostics_do_not_mutate_forecast_ready_data():
    ready = prepare_forecast_demand(
        prepare_user_demand_data(
            pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                    "demand": list(range(1, 13)),
                }
            ),
            date_column="date",
            demand_column="demand",
            frequency="monthly",
        )
    )
    original = ready.data.copy(deep=True)

    diagnose_demand_patterns(ready)

    pd.testing.assert_frame_equal(ready.data, original)


def test_expected_coverage_missingness_is_exposed_in_series_diagnostics():
    source = pd.DataFrame(
        {
            "date": ["2024-02-01", "2024-04-01"],
            "demand": [2, 4],
        }
    )
    prepared = prepare_user_demand_data(
        source,
        date_column="date",
        demand_column="demand",
        frequency="monthly",
        expected_start="2024-01-01",
        expected_end="2024-05-01",
    )

    result = only_series(
        diagnose_demand_patterns(prepare_forecast_demand(prepared))
    )

    assert result.expected_start_period == pd.Timestamp("2024-01-01")
    assert result.expected_end_period == pd.Timestamp("2024-05-01")
    assert result.missing_before_first_observation == 1
    assert result.missing_internal_periods == 1
    assert result.missing_after_last_observation == 1


def test_missing_grid_rows_must_be_materialized_as_unknown():
    ready = prepare_forecast_demand(
        prepare_user_demand_data(
            pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                    "demand": [1, 2, 3, 4],
                }
            ),
            date_column="date",
            demand_column="demand",
            frequency="monthly",
        )
    )
    incomplete = ForecastReadyDemandData(
        data=ready.data.drop(index=1).reset_index(drop=True),
        frequency=ready.frequency,
        coverage_by_product=ready.coverage_by_product,
        quality_report=ready.quality_report,
    )

    with pytest.raises(ValueError, match="period grid.*incomplete"):
        diagnose_demand_patterns(incomplete)


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.inf],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1],
    ],
)
def test_invalid_or_non_finite_eligible_values_are_rejected(values):
    ready = prepare_forecast_demand(
        prepare_user_demand_data(
            pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                    "demand": [1] * 12,
                }
            ),
            date_column="date",
            demand_column="demand",
            frequency="monthly",
        )
    )
    frame = ready.data.copy(deep=True)
    frame.loc[0, "demand_quantity"] = values[-1]
    malformed = ForecastReadyDemandData(
        data=frame,
        frequency=ready.frequency,
        coverage_by_product=ready.coverage_by_product,
        quality_report=ready.quality_report,
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        diagnose_demand_patterns(malformed)


def test_custom_thresholds_change_classification_boundary():
    prepared = prepare_forecast_demand(
        prepare_user_demand_data(
            pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=8, freq="MS"),
                    "demand": [0, 0, 5, 5, 5, 5, 5, 5],
                }
            ),
            date_column="date",
            demand_column="demand",
            frequency="monthly",
        )
    )
    result = diagnose_demand_patterns(
        prepared,
        thresholds=DemandPatternThresholds(
            min_history_periods=8,
            intermittent_zero_fraction=0.25,
        ),
    )

    assert "intermittent" in result.series[0].classifications
