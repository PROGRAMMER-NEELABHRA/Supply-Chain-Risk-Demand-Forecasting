"""Expanding-window rolling-origin evaluation for generic demand models."""

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.forecasting.user_demand_diagnostics import (
    DemandPatternDiagnostics,
    DemandPatternThresholds,
    SeriesDemandDiagnostics,
    diagnose_demand_patterns,
)
from src.forecasting.user_demand_models import (
    EvaluationMetrics,
    ForecastModel,
    ModelEligibilityError,
    NaiveDemandModel,
    _fresh_model,
    _trailing_eligible_prefix,
    evaluate_forecasts,
)
from src.preprocessing.user_demand_preparation import (
    ForecastReadyDemandData,
)


SelectionMetric = Literal["mae", "rmse", "wape"]
OriginStatus = Literal["success", "ineligible", "failed"]


@dataclass(frozen=True)
class RollingOriginConfig:
    """Controls expanding-window origins and selection requirements."""

    horizon: int
    min_train_periods: int = 8
    step: int = 1
    max_origins: int | None = None
    min_successful_origins: int = 2
    min_comparable_origins: int = 2
    selection_metric: SelectionMetric = "mae"
    min_success_fraction: float = 0.5

    def __post_init__(self) -> None:
        for field_name in (
            "horizon",
            "min_train_periods",
            "step",
            "min_successful_origins",
            "min_comparable_origins",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.max_origins is not None and (
            isinstance(self.max_origins, bool)
            or not isinstance(self.max_origins, int)
            or self.max_origins < 1
        ):
            raise ValueError("max_origins must be a positive integer or None.")
        if self.selection_metric not in {"mae", "rmse", "wape"}:
            raise ValueError("selection_metric must be mae, rmse, or wape.")
        if (
            not np.isfinite(self.min_success_fraction)
            or not 0 < self.min_success_fraction <= 1
        ):
            raise ValueError("min_success_fraction must be in (0, 1].")


@dataclass(frozen=True)
class RollingOriginModelResult:
    """One model attempt at one shared origin and evaluation window."""

    product_id: object | None
    origin_period: pd.Timestamp
    evaluation_periods: tuple[pd.Timestamp, ...]
    model_name: str
    status: OriginStatus
    metrics: EvaluationMetrics | None
    message: str | None


@dataclass(frozen=True)
class SkippedRollingOrigin:
    """An origin skipped because actuals were not fully eligible."""

    product_id: object | None
    origin_period: pd.Timestamp
    evaluation_periods: tuple[pd.Timestamp, ...]
    reason: str


@dataclass(frozen=True)
class RollingModelSummary:
    """Aggregate performance and origin counts for one product/model."""

    product_id: object | None
    model_name: str
    successful_origins: int
    failed_origins: int
    eligible_origins: int
    success_fraction: float
    successful_origin_periods: tuple[pd.Timestamp, ...]
    failure_reasons: tuple[str, ...]
    aggregate_metrics: EvaluationMetrics | None
    comparable_metrics: EvaluationMetrics | None
    comparable_origins: int


@dataclass(frozen=True)
class RollingModelSelection:
    """Selection result with common-coverage scores and explicit rationale."""

    product_id: object | None
    selected_model: str | None
    metric: SelectionMetric
    comparable_origins: tuple[pd.Timestamp, ...]
    candidate_scores: tuple[tuple[str, float], ...]
    rationale: str


@dataclass(frozen=True)
class RollingOriginBacktest:
    """Full rolling-origin outcomes, skips, summaries, and selection."""

    frequency: str
    horizon: int
    config: RollingOriginConfig
    origin_results: tuple[RollingOriginModelResult, ...]
    skipped_origins: tuple[SkippedRollingOrigin, ...]
    model_summaries: tuple[RollingModelSummary, ...]
    selections: tuple[RollingModelSelection, ...]


def rolling_origin_backtest(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    models: Sequence[ForecastModel],
    *,
    config: RollingOriginConfig,
) -> RollingOriginBacktest:
    """Evaluate models on common expanding-window origins per series.

    An origin index denotes the first period outside training; the origin
    period reported is the last period available to the model. At every
    origin, only the training prefix is used to calculate diagnostics, and
    model training uses only its latest contiguous eligible suffix. An
    evaluation window is shared by all candidates only if every target period
    is eligible.
    """
    _validate_inputs(demand, diagnostics, models, config)
    series_inputs = _series_inputs(demand, diagnostics)
    origin_results: list[RollingOriginModelResult] = []
    skipped: list[SkippedRollingOrigin] = []
    summaries: list[RollingModelSummary] = []
    selections: list[RollingModelSelection] = []

    for product_id, frame, pattern in series_inputs:
        actual_origins = _candidate_origins(frame, config)
        successful_origins_by_model: dict[str, set[pd.Timestamp]] = {
            model.name: set() for model in models
        }
        successful_pairs_by_model: dict[
            str, tuple[list[float], list[float]]
        ] = {model.name: ([], []) for model in models}
        origin_pairs_by_model: dict[
            str, dict[pd.Timestamp, tuple[list[float], list[float]]]
        ] = {model.name: {} for model in models}
        failures_by_model: dict[str, list[str]] = {
            model.name: [] for model in models
        }
        attempted_by_model: dict[str, int] = {
            model.name: 0 for model in models
        }

        for origin_index in actual_origins:
            target_indices = np.arange(
                origin_index,
                origin_index + config.horizon,
                dtype=int,
            )
            target_periods = tuple(
                pd.Timestamp(value)
                for value in frame.loc[target_indices, "period"]
            )
            origin_period = pd.Timestamp(frame.loc[origin_index - 1, "period"])
            target_eligibility = frame.loc[
                target_indices, "training_eligible"
            ].to_numpy(dtype=bool)
            if not target_eligibility.all():
                missing_periods = [
                    str(target_periods[offset])
                    for offset, is_eligible in enumerate(target_eligibility)
                    if not is_eligible
                ]
                skipped.append(
                    SkippedRollingOrigin(
                        product_id=product_id,
                        origin_period=origin_period,
                        evaluation_periods=target_periods,
                        reason=(
                            "Evaluation window includes unknown periods: "
                            + ", ".join(missing_periods)
                        ),
                    )
                )
                continue

            training = _trailing_eligible_prefix(frame, origin_index)
            training_diagnostics = _training_diagnostics(
                demand=demand,
                frame=frame,
                origin_index=origin_index,
                thresholds=diagnostics.thresholds,
            )
            actual = pd.to_numeric(
                frame.loc[target_indices, "demand_quantity"]
            ).to_numpy(dtype=float)

            for model in models:
                attempted_by_model[model.name] += 1
                candidate = _fresh_model(model)
                try:
                    candidate.fit(training, diagnostics=training_diagnostics)
                    forecast = candidate.forecast(config.horizon)
                    metrics = evaluate_forecasts(actual, forecast)
                except ModelEligibilityError as exc:
                    reason = str(exc)
                    failures_by_model[model.name].append(
                        f"Origin {origin_period:%Y-%m-%d}: {reason}"
                    )
                    origin_results.append(
                        RollingOriginModelResult(
                            product_id=product_id,
                            origin_period=origin_period,
                            evaluation_periods=target_periods,
                            model_name=model.name,
                            status="ineligible",
                            metrics=None,
                            message=reason,
                        )
                    )
                    continue
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failures_by_model[model.name].append(
                        f"Origin {origin_period:%Y-%m-%d}: {reason}"
                    )
                    origin_results.append(
                        RollingOriginModelResult(
                            product_id=product_id,
                            origin_period=origin_period,
                            evaluation_periods=target_periods,
                            model_name=model.name,
                            status="failed",
                            metrics=None,
                            message=reason,
                        )
                    )
                    continue

                successful_origins_by_model[model.name].add(origin_period)
                actual_values, forecast_values = successful_pairs_by_model[
                    model.name
                ]
                actual_values.extend(actual.tolist())
                forecast_values.extend(forecast.tolist())
                origin_pairs_by_model[model.name][origin_period] = (
                    actual.tolist(),
                    forecast.tolist(),
                )
                origin_results.append(
                    RollingOriginModelResult(
                        product_id=product_id,
                        origin_period=origin_period,
                        evaluation_periods=target_periods,
                        model_name=model.name,
                        status="success",
                        metrics=metrics,
                        message=None,
                    )
                )

        product_summaries = _summarize_models(
            product_id=product_id,
            models=models,
            successful_origins=successful_origins_by_model,
            successful_pairs=successful_pairs_by_model,
            origin_pairs=origin_pairs_by_model,
            failure_reasons=failures_by_model,
            attempted=attempted_by_model,
            minimum_successful=config.min_successful_origins,
            minimum_fraction=config.min_success_fraction,
        )
        summaries.extend(product_summaries)
        selections.append(
            _select_from_summaries(
                product_id=product_id,
                summaries=product_summaries,
                metric=config.selection_metric,
                minimum_successful=config.min_successful_origins,
                minimum_comparable=config.min_comparable_origins,
                minimum_fraction=config.min_success_fraction,
            )
        )

    return RollingOriginBacktest(
        frequency=demand.frequency,
        horizon=config.horizon,
        config=config,
        origin_results=tuple(origin_results),
        skipped_origins=tuple(skipped),
        model_summaries=tuple(summaries),
        selections=tuple(selections),
    )


def select_rolling_origin_model(
    backtest: RollingOriginBacktest,
    *,
    product_id: object | None = None,
    metric: SelectionMetric | None = None,
    min_successful_origins: int | None = None,
    min_comparable_origins: int | None = None,
    min_success_fraction: float | None = None,
) -> RollingModelSelection:
    """Select by a configured metric only on common successful origins."""
    if not isinstance(backtest, RollingOriginBacktest):
        raise TypeError("backtest must be a RollingOriginBacktest.")
    selected_metric = metric or backtest.config.selection_metric
    if selected_metric not in {"mae", "rmse", "wape"}:
        raise ValueError("metric must be mae, rmse, or wape.")
    minimum_successful = (
        min_successful_origins
        if min_successful_origins is not None
        else backtest.config.min_successful_origins
    )
    minimum_comparable = (
        min_comparable_origins
        if min_comparable_origins is not None
        else backtest.config.min_comparable_origins
    )
    _positive_integer(minimum_successful, "min_successful_origins")
    _positive_integer(minimum_comparable, "min_comparable_origins")
    minimum_fraction = (
        min_success_fraction
        if min_success_fraction is not None
        else backtest.config.min_success_fraction
    )
    if not np.isfinite(minimum_fraction) or not 0 < minimum_fraction <= 1:
        raise ValueError("min_success_fraction must be in (0, 1].")
    summaries = [
        summary
        for summary in backtest.model_summaries
        if summary.product_id == product_id
    ]
    if not summaries:
        return RollingModelSelection(
            product_id=product_id,
            selected_model=None,
            metric=selected_metric,
            comparable_origins=(),
            candidate_scores=(),
            rationale="No model summaries are available for this product.",
        )
    return _select_from_summaries(
        product_id=product_id,
        summaries=summaries,
        metric=selected_metric,
        minimum_successful=minimum_successful,
        minimum_comparable=minimum_comparable,
        minimum_fraction=minimum_fraction,
    )


def _validate_inputs(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    models: Sequence[ForecastModel],
    config: RollingOriginConfig,
) -> None:
    if not isinstance(demand, ForecastReadyDemandData):
        raise TypeError("demand must be ForecastReadyDemandData.")
    if not isinstance(diagnostics, DemandPatternDiagnostics):
        raise TypeError("diagnostics must be DemandPatternDiagnostics.")
    if not isinstance(config, RollingOriginConfig):
        raise TypeError("config must be RollingOriginConfig.")
    if diagnostics.frequency != demand.frequency:
        raise ValueError("Demand and diagnostics frequencies do not match.")
    if not models:
        raise ValueError("At least one model is required.")
    if any(not isinstance(model, ForecastModel) for model in models):
        raise TypeError("models must contain ForecastModel instances.")
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique.")
    if not any(isinstance(model, NaiveDemandModel) for model in models):
        raise ValueError("Rolling model comparison requires a Naive benchmark.")


def _series_inputs(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
) -> tuple[tuple[object | None, pd.DataFrame, SeriesDemandDiagnostics], ...]:
    frame = demand.data.copy(deep=True)
    required = {
        "period",
        "demand_quantity",
        "observation_status",
        "training_eligible",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Forecast-ready data is missing columns: "
            f"{sorted(required - set(frame.columns))}"
        )
    if frame.empty:
        raise ValueError("Forecast-ready demand cannot be empty.")
    if not pd.api.types.is_datetime64_any_dtype(frame["period"].dtype):
        raise ValueError("Forecast-ready periods must be datetime values.")
    if frame["period"].isna().any() or frame["period"].dt.tz is not None:
        raise ValueError("Forecast-ready periods must be timezone-naive.")
    status_eligibility = frame["observation_status"].isin(
        {
            "observed_demand",
            "observed_zero_demand",
            "confirmed_zero_demand",
        }
    )
    if not frame["training_eligible"].eq(status_eligibility).all():
        raise ValueError(
            "training_eligible conflicts with observation_status."
        )
    known = frame["training_eligible"]
    values = pd.to_numeric(frame["demand_quantity"], errors="coerce")
    known_values = values.loc[known].to_numpy(dtype=float)
    if not np.isfinite(known_values).all() or (known_values < 0).any():
        raise ValueError("Eligible demand must be finite and non-negative.")
    if values.loc[~known].notna().any():
        raise ValueError("Unknown periods must not contain demand values.")

    has_product = "product_id" in frame.columns
    key_columns = ["period"] + (["product_id"] if has_product else [])
    if frame.duplicated(key_columns).any():
        raise ValueError("Demand contains duplicate period/product keys.")
    if has_product:
        grouped = list(frame.groupby("product_id", sort=False, dropna=False))
    else:
        grouped = [(None, frame)]

    diagnostic_map = {
        diagnostic.product_id: diagnostic for diagnostic in diagnostics.series
    }
    if len(grouped) != len(diagnostics.series):
        raise ValueError("Demand and diagnostics series counts do not match.")
    offset = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}[
        demand.frequency
    ]
    result: list[
        tuple[object | None, pd.DataFrame, SeriesDemandDiagnostics]
    ] = []
    for product, group in grouped:
        group = group.sort_values("period", kind="stable").reset_index(drop=True)
        expected = pd.date_range(
            group["period"].min(),
            group["period"].max(),
            freq=offset,
        )
        if not pd.DatetimeIndex(group["period"]).equals(expected):
            raise ValueError(
                f"Product {product!r} has an incomplete period grid; "
                "unknown periods must be explicit."
            )
        diagnostic = (
            diagnostic_map.get(product)
            if has_product
            else diagnostics.series[0] if diagnostics.series else None
        )
        if diagnostic is None:
            raise ValueError(f"Diagnostics are missing product {product!r}.")
        result.append((product if has_product else None, group, diagnostic))
    return tuple(result)


def _candidate_origins(
    frame: pd.DataFrame,
    config: RollingOriginConfig,
) -> list[int]:
    final_origin = len(frame) - config.horizon
    if final_origin < config.min_train_periods:
        return []
    eligible = frame["training_eligible"].to_numpy(dtype=bool)
    origins = []
    for origin_index in range(
        config.min_train_periods,
        final_origin + 1,
        config.step,
    ):
        suffix_start = origin_index
        while suffix_start > 0 and eligible[suffix_start - 1]:
            suffix_start -= 1
        if origin_index - suffix_start >= config.min_train_periods:
            origins.append(origin_index)
    if config.max_origins is not None:
        origins = origins[-config.max_origins :]
    return origins


def _training_diagnostics(
    *,
    demand: ForecastReadyDemandData,
    frame: pd.DataFrame,
    origin_index: int,
    thresholds: DemandPatternThresholds,
) -> SeriesDemandDiagnostics | None:
    training_end = origin_index
    start = training_end
    eligible = frame["training_eligible"].to_numpy(dtype=bool)
    while start > 0 and eligible[start - 1]:
        start -= 1
    if start == training_end:
        return None
    training = frame.iloc[start:training_end].copy().reset_index(drop=True)
    training_ready = ForecastReadyDemandData(
        data=training,
        frequency=demand.frequency,
        coverage_by_product=pd.DataFrame(),
        quality_report=demand.quality_report,
    )
    result = diagnose_demand_patterns(training_ready, thresholds=thresholds)
    return result.series[0] if result.series else None


def _summarize_models(
    *,
    product_id: object | None,
    models: Sequence[ForecastModel],
    successful_origins: dict[str, set[pd.Timestamp]],
    successful_pairs: dict[str, tuple[list[float], list[float]]],
    origin_pairs: dict[
        str, dict[pd.Timestamp, tuple[list[float], list[float]]]
    ],
    failure_reasons: dict[str, list[str]],
    attempted: dict[str, int],
    minimum_successful: int,
    minimum_fraction: float,
) -> tuple[RollingModelSummary, ...]:
    provisional: list[RollingModelSummary] = []
    qualified_sets: list[set[pd.Timestamp]] = []
    for model in models:
        origins = successful_origins[model.name]
        success_count = len(origins)
        failed_count = len(failure_reasons[model.name])
        eligible_count = attempted[model.name]
        fraction = success_count / eligible_count if eligible_count else 0.0
        enough = (
            success_count >= minimum_successful
            and fraction >= minimum_fraction
        )
        if enough:
            qualified_sets.append(origins)
        actual_values, forecast_values = successful_pairs[model.name]
        aggregate = (
            evaluate_forecasts(actual_values, forecast_values)
            if actual_values
            else None
        )
        provisional.append(
            RollingModelSummary(
                product_id=product_id,
                model_name=model.name,
                successful_origins=success_count,
                failed_origins=failed_count,
                eligible_origins=eligible_count,
                success_fraction=fraction,
                successful_origin_periods=tuple(sorted(origins)),
                failure_reasons=tuple(failure_reasons[model.name]),
                aggregate_metrics=aggregate,
                comparable_metrics=None,
                comparable_origins=0,
            )
        )

    common_origins = set.intersection(*qualified_sets) if qualified_sets else set()
    result: list[RollingModelSummary] = []
    for summary in provisional:
        actual_common: list[float] = []
        forecast_common: list[float] = []
        for origin in sorted(common_origins):
            pair = origin_pairs[summary.model_name].get(origin)
            if pair is None:
                actual_common = []
                forecast_common = []
                break
            actual_common.extend(pair[0])
            forecast_common.extend(pair[1])
        comparable_metrics = (
            evaluate_forecasts(actual_common, forecast_common)
            if actual_common
            else None
        )
        result.append(
            RollingModelSummary(
                product_id=summary.product_id,
                model_name=summary.model_name,
                successful_origins=summary.successful_origins,
                failed_origins=summary.failed_origins,
                eligible_origins=summary.eligible_origins,
                success_fraction=summary.success_fraction,
                successful_origin_periods=summary.successful_origin_periods,
                failure_reasons=summary.failure_reasons,
                aggregate_metrics=summary.aggregate_metrics,
                comparable_metrics=comparable_metrics,
                comparable_origins=(
                    len(common_origins)
                    if summary.model_name in successful_origins
                    and common_origins.issubset(
                        successful_origins[summary.model_name]
                    )
                    else 0
                ),
            )
        )
    return tuple(result)


def _select_from_summaries(
    *,
    product_id: object | None,
    summaries: Sequence[RollingModelSummary],
    metric: SelectionMetric,
    minimum_successful: int,
    minimum_comparable: int,
    minimum_fraction: float,
) -> RollingModelSelection:
    qualified = [
        summary
        for summary in summaries
        if summary.successful_origins >= minimum_successful
        and summary.success_fraction >= minimum_fraction
    ]
    if not qualified:
        return RollingModelSelection(
            product_id=product_id,
            selected_model=None,
            metric=metric,
            comparable_origins=(),
            candidate_scores=(),
            rationale=(
                f"No model reached both the required {minimum_successful} "
                f"successful origins and {minimum_fraction:.0%} success rate."
            ),
        )
    common = set.intersection(
        *(set(summary.successful_origin_periods) for summary in qualified)
    )
    if len(common) < minimum_comparable:
        return RollingModelSelection(
            product_id=product_id,
            selected_model=None,
            metric=metric,
            comparable_origins=tuple(sorted(common)),
            candidate_scores=(),
            rationale=(
                f"Only {len(common)} origins are shared by all models meeting "
                f"the successful-origin threshold; at least "
                f"{minimum_comparable} are required for comparable selection."
            ),
        )
    scores: list[tuple[str, float]] = []
    for summary in qualified:
        metrics = summary.comparable_metrics
        if metrics is None:
            continue
        score = getattr(metrics, metric)
        if score is not None and np.isfinite(score):
            scores.append((summary.model_name, float(score)))
    if not scores:
        return RollingModelSelection(
            product_id=product_id,
            selected_model=None,
            metric=metric,
            comparable_origins=tuple(sorted(common)),
            candidate_scores=(),
            rationale=(
                f"No candidate has a finite {metric.upper()} on the common "
                "evaluation origins."
            ),
        )
    selected = min(scores, key=lambda score: score[1])
    return RollingModelSelection(
        product_id=product_id,
        selected_model=selected[0],
        metric=metric,
        comparable_origins=tuple(sorted(common)),
        candidate_scores=tuple(scores),
        rationale=(
            f"Selected {selected[0]} because it had the lowest {metric.upper()} "
            f"({selected[1]:.6g}) across {len(common)} origins shared by all "
            "models meeting the origin-count threshold."
        ),
    )


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
