import numpy as np
import pandas as pd
import pytest

from src.data.user_demand import prepare_user_demand_data
from src.forecasting.user_demand_diagnostics import (
    diagnose_demand_patterns,
)
from src.forecasting.user_demand_models import (
    CrostonDemandModel,
    ForecastModel,
    InvalidForecastError,
    ModelEligibilityError,
    MovingAverageDemandModel,
    NaiveDemandModel,
    SeasonalNaiveDemandModel,
    SimpleExponentialSmoothingDemandModel,
    evaluate_forecasts,
    evaluate_forecasting_models,
    forecast_demand_models,
    select_best_model,
)
from src.preprocessing.user_demand_preparation import (
    ForecastReadyDemandData,
    prepare_forecast_demand,
)


def build_ready(
    values_by_product,
    *,
    frequency="monthly",
    start="2018-01-01",
    dates_by_product=None,
    zero_confirmation=False,
):
    rows = []
    for product, values in values_by_product.items():
        dates = (
            dates_by_product[product]
            if dates_by_product is not None
            else pd.date_range(
                start,
                periods=len(values),
                freq={"daily": "D", "weekly": "W-MON", "monthly": "MS"}[
                    frequency
                ],
            )
        )
        rows.extend(
            {"period": date, "demand": value, "product": product}
            for date, value in zip(dates, values, strict=True)
            if value is not None
        )
    frame = pd.DataFrame(rows)
    if len(values_by_product) == 1:
        frame = frame.drop(columns=["product"])
        product_column = None
    else:
        product_column = "product"
    prepared = prepare_user_demand_data(
        frame,
        date_column="period",
        demand_column="demand",
        product_column=product_column,
        frequency=frequency,
        complete_source_coverage=zero_confirmation,
        confirm_missing_periods_are_zero=zero_confirmation,
    )
    return prepare_forecast_demand(prepared)


def diagnose(ready):
    return diagnose_demand_patterns(ready)


class NonFiniteForecastModel(ForecastModel):
    @property
    def name(self):
        return "Non-finite test model"

    @property
    def configuration(self):
        return {}

    def minimum_history(self):
        return 1

    def _fit(self, history):
        return None

    def _forecast(self, horizon):
        return np.full(horizon, np.inf)


class CoverageCaptureModel(ForecastModel):
    observed_expected_ends = []

    @property
    def name(self):
        return "Coverage capture"

    @property
    def configuration(self):
        return {}

    def minimum_history(self):
        return 2

    def fit(self, values, *, diagnostics=None):
        self.observed_expected_ends.append(
            diagnostics.expected_end_period if diagnostics is not None else None
        )
        return super().fit(values, diagnostics=diagnostics)

    def _fit(self, history):
        self.level = float(history[-1])

    def _forecast(self, horizon):
        return np.full(horizon, self.level)


def test_naive_moving_average_and_ses_expected_forecasts():
    history = [10, 20, 30, 40]

    assert NaiveDemandModel().fit(history).forecast(3).tolist() == [40, 40, 40]
    assert (
        MovingAverageDemandModel(window=3)
        .fit(history)
        .forecast(2)
        .tolist()
        == [30, 30]
    )
    assert (
        SimpleExponentialSmoothingDemandModel(alpha=0.5)
        .fit([10, 20, 30])
        .forecast(2)
        .tolist()
        == [22.5, 22.5]
    )


def test_seasonal_naive_repeats_last_cycle_with_diagnostics():
    pattern = [10, 20, 15, 25, 30, 18, 12, 22, 28, 16, 24, 35]
    ready = build_ready({"one": pattern * 3})
    series_diagnostics = diagnose(ready).series[0]
    model = SeasonalNaiveDemandModel(seasonal_period=12, minimum_cycles=2)

    forecast = model.fit(
        pattern * 3,
        diagnostics=series_diagnostics,
    ).forecast(15)

    assert forecast[:12].tolist() == pattern
    assert forecast[12:].tolist() == pattern[:3]


def test_seasonal_naive_rejects_insufficient_history_and_missing_evidence():
    ready = build_ready({"one": [10, 20, 15, 25, 30, 18, 12, 22]})
    diagnostics = diagnose(ready).series[0]
    model = SeasonalNaiveDemandModel(seasonal_period=4, minimum_cycles=2)

    with pytest.raises(ModelEligibilityError, match="seasonality evidence"):
        model.fit([1] * 8, diagnostics=diagnostics)
    with pytest.raises(ModelEligibilityError, match="at least 8"):
        model.fit([1] * 7, diagnostics=diagnostics)


def test_croston_requires_intermittent_diagnostics_and_two_nonzero_values():
    intermittent = [0, 0, 5, 0, 0, 10, 0, 0, 0, 8, 0, 0]
    ready = build_ready({"one": intermittent})
    series_diagnostics = diagnose(ready).series[0]
    model = CrostonDemandModel(alpha=0.1)

    forecast = model.fit(
        intermittent,
        diagnostics=series_diagnostics,
    ).forecast(4)

    assert (forecast > 0).all()
    assert len(set(forecast.tolist())) == 1

    regular_diagnostics = diagnose(build_ready({"one": [5] * 12})).series[0]
    with pytest.raises(ModelEligibilityError, match="intermittent demand"):
        CrostonDemandModel().fit([0, 5, 0, 5], diagnostics=regular_diagnostics)
    with pytest.raises(ModelEligibilityError, match="at least two positive"):
        CrostonDemandModel().fit(
            [0, 0, 0, 5, 0, 0, 0, 0],
            diagnostics=series_diagnostics,
        )


@pytest.mark.parametrize(
    ("model", "history", "minimum"),
    [
        (NaiveDemandModel(), [], "non-empty"),
        (MovingAverageDemandModel(window=3), [1, 2], "at least 3"),
        (SimpleExponentialSmoothingDemandModel(), [1], "at least 2"),
        (CrostonDemandModel(), [1, 2], "at least 3"),
    ],
)
def test_models_reject_insufficient_or_empty_history(model, history, minimum):
    with pytest.raises((ModelEligibilityError, ValueError), match=minimum):
        model.fit(history)


def test_unknown_periods_are_excluded_and_tail_unknown_withholds_forecast():
    values = list(range(1, 13))
    values[-1] = None
    ready = build_ready({"one": values})
    frame = ready.data.copy(deep=True)
    next_period = frame["period"].max() + pd.offsets.MonthBegin(1)
    frame.loc[len(frame)] = [
        next_period,
        np.nan,
        "unknown_unobserved_period",
        False,
    ]
    ready = ForecastReadyDemandData(
        data=frame,
        frequency=ready.frequency,
        coverage_by_product=ready.coverage_by_product,
        quality_report=ready.quality_report,
    )
    diagnostics = diagnose(ready)

    output = forecast_demand_models(
        ready,
        diagnostics,
        [NaiveDemandModel()],
        horizon=2,
    )

    assert len(output) == 1
    assert not output[0].model_eligible
    assert output[0].predicted_demand is None
    assert any(
        "Latest period is unknown" in warning for warning in output[0].warnings
    )


def test_confirmed_zero_is_eligible_for_training():
    values = list(range(1, 12)) + [0]
    ready = build_ready({"one": values}, zero_confirmation=True)
    diagnostics = diagnose(ready)
    forecasts = forecast_demand_models(
        ready,
        diagnostics,
        [NaiveDemandModel()],
        horizon=2,
    )

    assert all(result.model_eligible for result in forecasts)
    assert [result.predicted_demand for result in forecasts] == [0, 0]


def test_confirmed_absence_zero_is_used_but_unknown_period_is_excluded():
    values = list(range(1, 12)) + [None, 13]
    unknown_ready = build_ready({"one": values})
    confirmed_ready = build_ready(
        {"one": values},
        zero_confirmation=True,
    )
    unknown_forecast = forecast_demand_models(
        unknown_ready,
        diagnose(unknown_ready),
        [MovingAverageDemandModel(window=2)],
        horizon=1,
    )[0]
    confirmed_forecast = forecast_demand_models(
        confirmed_ready,
        diagnose(confirmed_ready),
        [MovingAverageDemandModel(window=2)],
        horizon=1,
    )[0]

    assert not unknown_forecast.model_eligible
    assert "at least 2 training observations" in unknown_forecast.warnings[0]
    assert confirmed_forecast.model_eligible
    assert confirmed_forecast.predicted_demand == 6.5


def test_comparison_uses_common_chronological_window_and_naive_benchmark():
    values = list(range(10, 25))
    ready = build_ready({"one": values})
    original = ready.data.copy(deep=True)
    diagnostics = diagnose(ready)
    models = [
        NaiveDemandModel(),
        MovingAverageDemandModel(window=3),
        SimpleExponentialSmoothingDemandModel(),
        SeasonalNaiveDemandModel(seasonal_period=12, minimum_cycles=2),
    ]

    comparison = evaluate_forecasting_models(
        ready,
        diagnostics,
        models,
        horizon=3,
    )
    results = comparison.evaluations

    assert {row.evaluation_start for row in results} == {pd.Timestamp("2019-01-01")}
    assert {row.evaluation_end for row in results} == {pd.Timestamp("2019-03-01")}
    assert all(row.eligible for row in results if row.model_name != "Seasonal Naive")
    seasonal = next(row for row in results if row.model_name == "Seasonal Naive")
    assert not seasonal.eligible
    assert "at least 24" in seasonal.warning
    assert select_best_model(comparison).model_name == "Naive"
    pd.testing.assert_frame_equal(ready.data, original)


def test_croston_eligibility_uses_training_only_diagnostics():
    # Intermittency appears only in the held-out tail. It must not enable
    # Croston on a regular training window.
    values = [10] * 9 + [0] * 6
    ready = build_ready({"one": values})
    full_diagnostics = diagnose(ready)
    assert full_diagnostics.series[0].intermittent_behavior

    comparison = evaluate_forecasting_models(
        ready,
        full_diagnostics,
        [NaiveDemandModel(), CrostonDemandModel()],
        horizon=6,
    )

    croston = next(
        row for row in comparison.evaluations if row.model_name == "Croston"
    )
    assert not croston.eligible
    assert "intermittent demand" in croston.warning


def test_expected_future_coverage_does_not_leak_into_holdout_diagnostics():
    source = pd.DataFrame(
        {
            "period": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "demand": list(range(1, 13)),
        }
    )
    prepared = prepare_user_demand_data(
        source,
        date_column="period",
        demand_column="demand",
        frequency="monthly",
        expected_start="2020-01-01",
        expected_end="2020-12-01",
    )
    ready = prepare_forecast_demand(prepared)
    CoverageCaptureModel.observed_expected_ends = []

    evaluate_forecasting_models(
        ready,
        diagnose(ready),
        [NaiveDemandModel(), CoverageCaptureModel()],
        horizon=2,
    )

    assert CoverageCaptureModel.observed_expected_ends
    assert all(value is None for value in CoverageCaptureModel.observed_expected_ends)


def test_model_comparison_requires_shared_actual_window_and_naive():
    ready = build_ready({"one": [1, 2, None, 4, 5]})
    diagnostics = diagnose(ready)

    result = evaluate_forecasting_models(
        ready,
        diagnostics,
        [NaiveDemandModel(), MovingAverageDemandModel(3)],
        horizon=3,
    )
    assert not any(row.eligible for row in result.evaluations)
    assert all("No contiguous run" in row.warning for row in result.evaluations)

    with pytest.raises(ValueError, match="Naive benchmark"):
        evaluate_forecasting_models(
            ready,
            diagnostics,
            [MovingAverageDemandModel(3)],
            horizon=1,
        )


def test_metrics_handle_zero_actual_total_and_report_totals():
    metrics = evaluate_forecasts([0, 0, 0], [1, 2, 3])

    assert metrics.mae == 2
    assert metrics.rmse == pytest.approx(np.sqrt(14 / 3))
    assert metrics.wape is None
    assert metrics.actual_total == 0
    assert metrics.forecast_total == 6
    assert metrics.evaluated_observations == 3


def test_metrics_reject_non_finite_predictions_and_bad_shapes():
    with pytest.raises(InvalidForecastError, match="non-finite"):
        evaluate_forecasts([1, 2], [1, np.inf])
    with pytest.raises(ValueError, match="match actual length"):
        evaluate_forecasts([1, 2], [1])


def test_non_finite_model_prediction_is_rejected_not_zeroed():
    with pytest.raises(InvalidForecastError, match="non-finite"):
        NonFiniteForecastModel().fit([1, 2]).forecast(2)


def test_multiple_products_are_evaluated_independently():
    ready = build_ready(
        {"regular": list(range(1, 13)), "zero_tail": list(range(1, 11)) + [0, 0]}
    )
    diagnostics = diagnose(ready)

    comparison = evaluate_forecasting_models(
        ready,
        diagnostics,
        [NaiveDemandModel()],
        horizon=2,
    )

    assert {row.product_id for row in comparison.evaluations} == {
        "regular",
        "zero_tail",
    }
    assert all(row.eligible for row in comparison.evaluations)
    forecasts = forecast_demand_models(
        ready,
        diagnostics,
        [NaiveDemandModel()],
        horizon=2,
    )
    assert len(forecasts) == 4
    assert {item.product_id for item in forecasts} == {"regular", "zero_tail"}


def test_one_product_can_be_ineligible_while_another_is_forecast():
    ready = build_ready(
        {"long": list(range(1, 13)), "short": [1, 2]},
    )
    diagnostics = diagnose(ready)
    comparison = evaluate_forecasting_models(
        ready,
        diagnostics,
        [NaiveDemandModel(), MovingAverageDemandModel(window=3)],
        horizon=1,
    )
    long_results = [
        row for row in comparison.evaluations if row.product_id == "long"
    ]
    short_results = [
        row for row in comparison.evaluations if row.product_id == "short"
    ]

    assert all(row.eligible for row in long_results)
    short_by_model = {row.model_name: row for row in short_results}
    assert short_by_model["Naive"].eligible
    assert not short_by_model["Moving Average"].eligible
    assert short_by_model["Moving Average"].warning


def test_future_forecast_periods_and_horizons_match_frequency():
    ready = build_ready(
        {"one": [1, 2, 3]},
        frequency="weekly",
        start="2024-01-01",
    )
    forecasts = forecast_demand_models(
        ready,
        diagnose(ready),
        [NaiveDemandModel()],
        horizon=2,
    )

    assert [item.forecast_period for item in forecasts] == list(
        pd.to_datetime(["2024-01-22", "2024-01-29"])
    )
    assert [item.forecast_horizon for item in forecasts] == [1, 2]


def test_model_input_and_model_predictions_are_deterministic():
    source = [3, 8, 2, 7]
    original = source.copy()
    model = MovingAverageDemandModel(window=2)

    first = model.fit(source).forecast(2)
    second = MovingAverageDemandModel(window=2).fit(source).forecast(2)

    assert first.tolist() == second.tolist() == [4.5, 4.5]
    assert source == original


def test_invalid_forecast_horizons_and_training_values_are_rejected():
    model = NaiveDemandModel().fit([1, 2])
    for invalid_horizon in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            model.forecast(invalid_horizon)
    with pytest.raises(ValueError, match="finite"):
        NaiveDemandModel().fit([1, np.nan])


def test_daily_and_monthly_forecasts_use_correct_next_period():
    daily = build_ready(
        {"one": [1, 2]},
        frequency="daily",
        start="2024-02-28",
    )
    monthly = build_ready(
        {"one": [1, 2]},
        frequency="monthly",
        start="2024-01-01",
    )

    daily_forecast = forecast_demand_models(
        daily, diagnose(daily), [NaiveDemandModel()], horizon=1
    )[0]
    monthly_forecast = forecast_demand_models(
        monthly, diagnose(monthly), [NaiveDemandModel()], horizon=1
    )[0]
    assert daily_forecast.forecast_period == pd.Timestamp("2024-03-01")
    assert monthly_forecast.forecast_period == pd.Timestamp("2024-03-01")
