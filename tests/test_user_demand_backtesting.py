import numpy as np
import pandas as pd
import pytest

from src.data.user_demand import prepare_user_demand_data
from src.forecasting.user_demand_diagnostics import (
    diagnose_demand_patterns,
)
from src.forecasting.user_demand_backtesting import (
    RollingOriginConfig,
    rolling_origin_backtest,
    select_rolling_origin_model,
)
from src.forecasting.user_demand_models import (
    ForecastModel,
    ModelEligibilityError,
    MovingAverageDemandModel,
    NaiveDemandModel,
    evaluate_forecasting_models,
)
from src.preprocessing.user_demand_preparation import (
    prepare_forecast_demand,
)


def build_ready(
    values,
    *,
    frequency="monthly",
    start="2018-01-01",
    complete_coverage=False,
):
    dates = pd.date_range(
        start,
        periods=len(values),
        freq={"daily": "D", "weekly": "W-MON", "monthly": "MS"}[frequency],
    )
    source = pd.DataFrame({"date": dates, "demand": values})
    source = source.loc[source["demand"].notna()].reset_index(drop=True)
    prepared = prepare_user_demand_data(
        source,
        date_column="date",
        demand_column="demand",
        frequency=frequency,
        complete_source_coverage=complete_coverage,
        confirm_missing_periods_are_zero=complete_coverage,
    )
    return prepare_forecast_demand(prepared)


def run(ready, models, **config):
    diagnostics = diagnose_demand_patterns(ready)
    settings = RollingOriginConfig(**config)
    return rolling_origin_backtest(
        ready,
        diagnostics,
        models,
        config=settings,
    )


class RecordingModel(ForecastModel):
    seen_training_values = []
    seen_diagnostics_history = []
    seen_diagnostics_expected_end = []

    @property
    def name(self):
        return "Recording"

    @property
    def configuration(self):
        return {}

    def minimum_history(self):
        return 2

    def fit(self, values, *, diagnostics=None):
        self.seen_training_values.append(list(values))
        self.seen_diagnostics_history.append(
            diagnostics.history_length if diagnostics is not None else None
        )
        self.seen_diagnostics_expected_end.append(
            diagnostics.expected_end_period if diagnostics is not None else None
        )
        return super().fit(values, diagnostics=diagnostics)

    def _fit(self, history):
        self.level = float(history[-1])

    def _forecast(self, horizon):
        return np.full(horizon, self.level)


class FailAtFiveModel(ForecastModel):
    @property
    def name(self):
        return "Fail at five"

    @property
    def configuration(self):
        return {}

    def minimum_history(self):
        return 2

    def _fit(self, history):
        if len(history) == 5:
            raise RuntimeError("intentional origin failure")
        self.level = float(history[-1])

    def _forecast(self, horizon):
        return np.full(horizon, self.level)


class OriginSpecificFailureModel(ForecastModel):
    @property
    def name(self):
        return "Origin specific failure"

    @property
    def configuration(self):
        return {}

    def minimum_history(self):
        return 2

    def _fit(self, history):
        if len(history) == 5:
            raise ModelEligibilityError("intentional eligibility failure")
        self.level = float(history[-1])

    def _forecast(self, horizon):
        return np.full(horizon, self.level)


def test_generates_multiple_expanding_window_origins():
    ready = build_ready(list(range(1, 13)))
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=2,
        min_train_periods=4,
        step=2,
        min_successful_origins=2,
        min_comparable_origins=2,
    )

    summary = result.model_summaries[0]
    assert summary.successful_origins == 4
    assert summary.failed_origins == 0
    assert summary.successful_origin_periods == tuple(
        pd.to_datetime(
            ["2018-04-01", "2018-06-01", "2018-08-01", "2018-10-01"]
        )
    )
    assert summary.aggregate_metrics.evaluated_observations == 8
    assert result.selections[0].selected_model == "Naive"


def test_training_values_and_diagnostics_exclude_future_observations():
    RecordingModel.seen_training_values = []
    RecordingModel.seen_diagnostics_history = []
    RecordingModel.seen_diagnostics_expected_end = []
    ready = build_ready(list(range(1, 10)))

    run(
        ready,
        [NaiveDemandModel(), RecordingModel()],
        horizon=1,
        min_train_periods=4,
        step=2,
        min_successful_origins=2,
        min_comparable_origins=2,
    )

    assert RecordingModel.seen_training_values == [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ]
    assert RecordingModel.seen_diagnostics_history == [4, 6, 8]


def test_expected_future_bounds_do_not_leak_into_training_diagnostics():
    source = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "demand": list(range(1, 13)),
        }
    )
    prepared = prepare_user_demand_data(
        source,
        date_column="date",
        demand_column="demand",
        frequency="monthly",
        expected_start="2020-01-01",
        expected_end="2021-12-01",
    )
    ready = prepare_forecast_demand(prepared)
    diagnostics = diagnose_demand_patterns(ready)
    RecordingModel.seen_diagnostics_expected_end = []

    rolling_origin_backtest(
        ready,
        diagnostics,
        [NaiveDemandModel(), RecordingModel()],
        config=RollingOriginConfig(
            horizon=1,
            min_train_periods=3,
            min_successful_origins=1,
            min_comparable_origins=1,
        ),
    )

    assert diagnostics.series[0].expected_end_period == pd.Timestamp("2021-12-01")
    assert RecordingModel.seen_diagnostics_expected_end
    assert all(value is None for value in RecordingModel.seen_diagnostics_expected_end)


def test_future_extremes_cannot_change_earlier_origin_training_or_diagnostics():
    RecordingModel.seen_training_values = []
    RecordingModel.seen_diagnostics_history = []
    first = build_ready([2, 4, 6, 8, 10, 12, 14, 16])
    altered = build_ready([2, 4, 6, 8, 10, 12, 1400, 1600])

    config = {
        "horizon": 3,
        "min_train_periods": 4,
        "step": 2,
        "max_origins": 2,
        "min_successful_origins": 1,
        "min_comparable_origins": 1,
    }
    first_result = run(first, [NaiveDemandModel(), RecordingModel()], **config)
    first_values = RecordingModel.seen_training_values[0]
    first_history_size = RecordingModel.seen_diagnostics_history[0]
    changed_result = run(
        altered,
        [NaiveDemandModel(), RecordingModel()],
        **config,
    )

    assert first_values == [2.0, 4.0, 6.0, 8.0]
    assert RecordingModel.seen_training_values[-2] == first_values
    assert RecordingModel.seen_diagnostics_history[-2] == first_history_size
    assert (
        first_result.origin_results[0].origin_period
        == changed_result.origin_results[0].origin_period
    )


def test_model_failures_at_individual_origins_are_reported_and_counted():
    ready = build_ready(list(range(1, 12)))
    result = run(
        ready,
        [NaiveDemandModel(), FailAtFiveModel(), OriginSpecificFailureModel()],
        horizon=1,
        min_train_periods=3,
        step=1,
        min_successful_origins=2,
        min_comparable_origins=2,
    )

    failed = [
        row
        for row in result.origin_results
        if row.model_name in {"Fail at five", "Origin specific failure"}
        and len(row.message or "") > 0
        and "intentional" in row.message
    ]
    assert {row.status for row in failed} == {"failed", "ineligible"}
    for model_name in ("Fail at five", "Origin specific failure"):
        summary = next(
            item
            for item in result.model_summaries
            if item.model_name == model_name
        )
        assert summary.failed_origins == 1
        assert summary.successful_origins > 0
        assert any("intentional" in message for message in summary.failure_reasons)
    assert result.selections[0].selected_model == "Naive"
    assert "shared by all models" in result.selections[0].rationale


def test_unknown_targets_are_skipped_but_confirmed_zero_targets_are_evaluated():
    values = [1, 2, None, 4, 5, 6, 7, 8, 9]
    unknown_ready = build_ready(values)
    confirmed_ready = build_ready(values, complete_coverage=True)

    unknown_result = run(
        unknown_ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=2,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
    )
    confirmed_result = run(
        confirmed_ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=2,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    assert unknown_result.skipped_origins
    assert all(
        "unknown periods" in origin.reason
        for origin in unknown_result.skipped_origins
    )
    assert any(
        item.status == "success"
        and item.metrics.actual_total == 0
        for item in confirmed_result.origin_results
    )


def test_unknown_gap_is_not_crossed_during_training():
    ready = build_ready([10, 20, None, 100, 200, 300, 400, 500])
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=4,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    valid_origins = [
        item
        for item in result.origin_results
        if item.status == "success"
    ]
    assert valid_origins
    assert valid_origins[0].origin_period == pd.Timestamp("2018-07-01")
    assert valid_origins[0].metrics.forecast_total == 400
    assert all(
        item.origin_period >= pd.Timestamp("2018-07-01")
        for item in result.origin_results
    )
    assert not result.skipped_origins


def test_origins_after_unknown_gap_wait_for_configured_contiguous_history():
    ready = build_ready([10, 20, 30, None, 100, 200, 300, 400, 500, 600])
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=3,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    assert [item.origin_period for item in result.origin_results] == list(
        pd.to_datetime(["2018-07-01", "2018-08-01", "2018-09-01"])
    )
    assert all(item.status == "success" for item in result.origin_results)
    assert [
        item.metrics.forecast_total for item in result.origin_results
    ] == [300, 400, 500]
    assert len(result.skipped_origins) == 1
    assert "unknown periods" in result.skipped_origins[0].reason


def test_naive_cannot_bypass_configured_minimum_training_history():
    ready = build_ready([10, None, 20, 30, 40, 50, 60])
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=3,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    assert [item.origin_period for item in result.origin_results] == list(
        pd.to_datetime(["2018-05-01", "2018-06-01"])
    )
    assert all(item.status == "success" for item in result.origin_results)
    assert [
        item.metrics.forecast_total for item in result.origin_results
    ] == [40, 50]
    assert not result.skipped_origins


def test_complete_history_still_generates_configured_expanding_origins():
    ready = build_ready(list(range(1, 10)))
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=3,
        step=2,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    assert [item.origin_period for item in result.origin_results] == list(
        pd.to_datetime(["2018-03-01", "2018-05-01", "2018-07-01"])
    )
    assert all(item.status == "success" for item in result.origin_results)


def test_aggregate_metrics_use_all_valid_origin_period_pairs():
    ready = build_ready([1, 2, 3, 4, 5, 6, 7, 8])
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=2,
        min_train_periods=3,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    summary = result.model_summaries[0]
    successful = [
        outcome
        for outcome in result.origin_results
        if outcome.status == "success"
    ]
    actual_total = sum(outcome.metrics.actual_total for outcome in successful)
    assert summary.aggregate_metrics.actual_total == actual_total
    assert summary.aggregate_metrics.evaluated_observations == len(successful) * 2


def test_zero_total_wape_is_unavailable_in_rolling_metrics():
    ready = build_ready([0] * 10)
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=2,
        min_train_periods=3,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
        selection_metric="wape",
    )

    summary = result.model_summaries[0]
    assert summary.aggregate_metrics.actual_total == 0
    assert summary.aggregate_metrics.wape is None
    assert result.selections[0].selected_model is None
    assert "No candidate has a finite WAPE" in result.selections[0].rationale


def test_selection_metric_can_be_changed_and_rationale_is_returned():
    ready = build_ready([1, 10, 1, 10, 1, 10, 1, 10, 1, 10])
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=3,
        step=1,
        min_successful_origins=2,
        min_comparable_origins=2,
    )

    selected = select_rolling_origin_model(result, metric="rmse")
    assert selected.selected_model == "Naive"
    assert selected.metric == "rmse"
    assert "lowest RMSE" in selected.rationale
    assert selected.candidate_scores


def test_minimum_successful_origins_and_comparability_block_selection():
    ready = build_ready(list(range(1, 9)))
    result = run(
        ready,
        [NaiveDemandModel()],
        horizon=1,
        min_train_periods=6,
        step=1,
        min_successful_origins=3,
        min_comparable_origins=3,
    )

    assert result.model_summaries[0].successful_origins == 2
    assert result.selections[0].selected_model is None
    assert "No model reached" in result.selections[0].rationale


def test_insufficient_training_history_is_reported_per_origin():
    ready = build_ready(list(range(1, 8)))
    result = run(
        ready,
        [NaiveDemandModel(), MovingAverageDemandModel(window=3)],
        horizon=2,
        min_train_periods=2,
        step=1,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    assert result.origin_results
    moving_average = [
        row
        for row in result.origin_results
        if row.model_name == "Moving Average"
    ]
    assert moving_average[0].status == "ineligible"
    assert "at least 3 training observations" in moving_average[0].message
    assert moving_average[-1].status == "success"
    assert moving_average[0].origin_period == pd.Timestamp("2018-02-01")


def test_common_origins_and_selection_use_identical_coverage():
    ready = build_ready(list(range(1, 12)))
    result = run(
        ready,
        [NaiveDemandModel(), FailAtFiveModel()],
        horizon=1,
        min_train_periods=3,
        step=1,
        min_successful_origins=2,
        min_comparable_origins=8,
    )

    naive = next(
        summary for summary in result.model_summaries if summary.model_name == "Naive"
    )
    failing = next(
        summary
        for summary in result.model_summaries
        if summary.model_name == "Fail at five"
    )
    selection = result.selections[0]
    assert naive.comparable_origins == failing.comparable_origins == 7
    assert selection.selected_model is None
    assert "at least 8 are required" in selection.rationale


def test_deterministic_repeat_and_existing_single_holdout_api_compatibility():
    ready = build_ready(list(range(1, 10)))
    diagnostics = diagnose_demand_patterns(ready)
    models = [NaiveDemandModel()]
    config = RollingOriginConfig(
        horizon=2,
        min_train_periods=3,
        step=2,
        min_successful_origins=1,
        min_comparable_origins=1,
    )
    first = rolling_origin_backtest(ready, diagnostics, models, config=config)
    second = rolling_origin_backtest(ready, diagnostics, models, config=config)
    assert first == second

    holdout = evaluate_forecasting_models(
        ready,
        diagnostics,
        [NaiveDemandModel()],
        horizon=2,
    )
    assert len(holdout.evaluations) == 1
    assert holdout.evaluations[0].eligible


@pytest.mark.parametrize(
    "config",
    [
        {"horizon": 0},
        {"horizon": 1, "step": 0},
        {"horizon": 1, "min_success_fraction": 0},
        {"horizon": 1, "selection_metric": "bad"},
    ],
)
def test_invalid_rolling_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        RollingOriginConfig(**config)


def test_naive_benchmark_is_required_for_comparisons():
    ready = build_ready(list(range(1, 9)))
    with pytest.raises(ValueError, match="Naive benchmark"):
        run(
            ready,
            [MovingAverageDemandModel(window=2)],
            horizon=1,
            min_train_periods=3,
        )
