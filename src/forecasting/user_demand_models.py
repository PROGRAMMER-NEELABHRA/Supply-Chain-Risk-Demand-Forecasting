"""Reusable forecasting models and evaluation for generic demand series."""

from abc import ABC, abstractmethod
from copy import deepcopy
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
from src.preprocessing.user_demand_preparation import (
    ForecastReadyDemandData,
)


class ModelEligibilityError(ValueError):
    """Raised when a model cannot be fit to the supplied history."""


class InvalidForecastError(ValueError):
    """Raised when a model produces invalid forecasts."""


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics for a common chronological evaluation window."""

    mae: float
    rmse: float
    wape: float | None
    actual_total: float
    forecast_total: float
    evaluated_observations: int


@dataclass(frozen=True)
class ModelEvaluation:
    """One model's eligibility and scores on the shared evaluation window."""

    product_id: object | None
    model_name: str
    eligible: bool
    evaluation_start: pd.Timestamp | None
    evaluation_end: pd.Timestamp | None
    metrics: EvaluationMetrics | None
    warning: str | None


@dataclass(frozen=True)
class ForecastComparison:
    """Chronological model comparison, preserving ineligible candidates."""

    frequency: str
    horizon: int
    evaluations: tuple[ModelEvaluation, ...]


@dataclass(frozen=True)
class DemandForecast:
    """A single future forecast with model and pattern context."""

    product_id: object | None
    forecast_period: pd.Timestamp | None
    predicted_demand: float | None
    model_name: str
    forecast_horizon: int
    model_eligible: bool
    warnings: tuple[str, ...]
    demand_pattern: tuple[str, ...]


class ForecastModel(ABC):
    """Small common interface for generic univariate demand models."""

    def __init__(self) -> None:
        self._history: np.ndarray | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def configuration(self) -> dict[str, object]:
        """Model parameters suitable for reports."""

    @abstractmethod
    def minimum_history(self) -> int:
        """Minimum number of finite training observations."""

    def fit(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        diagnostics: SeriesDemandDiagnostics | None = None,
    ) -> "ForecastModel":
        """Fit on known chronological values, rejecting invalid data."""
        self._history = None
        history = _validated_demand(values, "training history")
        if len(history) < self.minimum_history():
            raise ModelEligibilityError(
                f"{self.name} requires at least {self.minimum_history()} "
                f"training observations; received {len(history)}."
            )
        self._validate_diagnostics(diagnostics)
        self._fit(history)
        self._history = history.copy()
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        """Forecast a positive number of future periods."""
        if self._history is None:
            raise ModelEligibilityError(f"{self.name} must be fit before forecasting.")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
            raise ValueError("horizon must be a positive integer.")
        if horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        predictions = np.asarray(self._forecast(int(horizon)), dtype=float)
        if predictions.ndim != 1 or len(predictions) != horizon:
            raise InvalidForecastError(
                f"{self.name} returned {predictions.size} values for horizon "
                f"{horizon}; expected a one-dimensional array of that length."
            )
        if not np.isfinite(predictions).all():
            raise InvalidForecastError(
                f"{self.name} produced non-finite forecast values."
            )
        if (predictions < 0).any():
            raise InvalidForecastError(
                f"{self.name} produced negative demand forecasts."
            )
        return predictions.copy()

    def _validate_diagnostics(
        self,
        diagnostics: SeriesDemandDiagnostics | None,
    ) -> None:
        del diagnostics

    @abstractmethod
    def _fit(self, history: np.ndarray) -> None:
        """Fit model-specific state."""

    @abstractmethod
    def _forecast(self, horizon: int) -> np.ndarray:
        """Return model-specific predictions."""


class NaiveDemandModel(ForecastModel):
    """Repeat the most recent known demand observation."""

    @property
    def name(self) -> str:
        return "Naive"

    @property
    def configuration(self) -> dict[str, object]:
        return {}

    def minimum_history(self) -> int:
        return 1

    def _fit(self, history: np.ndarray) -> None:
        self._level = float(history[-1])

    def _forecast(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._level, dtype=float)


class MovingAverageDemandModel(ForecastModel):
    """Repeat the mean of the configured trailing training window."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer.")
        self.window = window

    @property
    def name(self) -> str:
        return "Moving Average"

    @property
    def configuration(self) -> dict[str, object]:
        return {"window": self.window}

    def minimum_history(self) -> int:
        return self.window

    def _fit(self, history: np.ndarray) -> None:
        self._level = float(np.mean(history[-self.window :]))

    def _forecast(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._level, dtype=float)


class SeasonalNaiveDemandModel(ForecastModel):
    """Repeat the latest complete seasonal cycle."""

    def __init__(self, seasonal_period: int, minimum_cycles: int = 3) -> None:
        super().__init__()
        if (
            isinstance(seasonal_period, bool)
            or not isinstance(seasonal_period, int)
            or seasonal_period < 2
        ):
            raise ValueError("seasonal_period must be an integer of at least 2.")
        if (
            isinstance(minimum_cycles, bool)
            or not isinstance(minimum_cycles, int)
            or minimum_cycles < 2
        ):
            raise ValueError("minimum_cycles must be an integer of at least 2.")
        self.seasonal_period = seasonal_period
        self.minimum_cycles = minimum_cycles

    @property
    def name(self) -> str:
        return "Seasonal Naive"

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "seasonal_period": self.seasonal_period,
            "minimum_cycles": self.minimum_cycles,
        }

    def minimum_history(self) -> int:
        return self.seasonal_period * self.minimum_cycles

    def _validate_diagnostics(
        self,
        diagnostics: SeriesDemandDiagnostics | None,
    ) -> None:
        if diagnostics is None or not diagnostics.seasonality_evidence:
            raise ModelEligibilityError(
                "Seasonal Naive requires Phase 3 diagnostics with "
                "sufficient-history seasonality evidence."
            )
        if diagnostics.seasonal_period != self.seasonal_period:
            raise ModelEligibilityError(
                "Seasonal Naive period does not match the detected frequency."
            )

    def _fit(self, history: np.ndarray) -> None:
        self._season = history[-self.seasonal_period :].copy()

    def _forecast(self, horizon: int) -> np.ndarray:
        return np.resize(self._season, horizon).astype(float)


class SimpleExponentialSmoothingDemandModel(ForecastModel):
    """Simple exponential smoothing with a fixed smoothing coefficient."""

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be finite and in (0, 1].")
        self.alpha = float(alpha)

    @property
    def name(self) -> str:
        return "Simple Exponential Smoothing"

    @property
    def configuration(self) -> dict[str, object]:
        return {"alpha": self.alpha}

    def minimum_history(self) -> int:
        return 2

    def _fit(self, history: np.ndarray) -> None:
        level = float(history[0])
        for observation in history[1:]:
            level = self.alpha * float(observation) + (1 - self.alpha) * level
        self._level = level

    def _forecast(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._level, dtype=float)


class CrostonDemandModel(ForecastModel):
    """Classic Croston intermittent-demand forecast with fixed alpha."""

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be finite and in (0, 1].")
        self.alpha = float(alpha)

    @property
    def name(self) -> str:
        return "Croston"

    @property
    def configuration(self) -> dict[str, object]:
        return {"alpha": self.alpha}

    def minimum_history(self) -> int:
        return 3

    def _validate_diagnostics(
        self,
        diagnostics: SeriesDemandDiagnostics | None,
    ) -> None:
        if diagnostics is None or not diagnostics.intermittent_behavior:
            raise ModelEligibilityError(
                "Croston requires Phase 3 diagnostics identifying "
                "intermittent demand."
            )

    def _fit(self, history: np.ndarray) -> None:
        nonzero_indices = np.flatnonzero(history > 0)
        if len(nonzero_indices) < 2:
            raise ModelEligibilityError(
                "Croston requires at least two positive-demand observations."
            )
        first = int(nonzero_indices[0])
        demand_estimate = float(history[first])
        interval_estimate = float(first + 1)
        previous = first
        for index in nonzero_indices[1:]:
            interval = int(index) - previous
            demand_estimate = (
                self.alpha * float(history[index])
                + (1 - self.alpha) * demand_estimate
            )
            interval_estimate = (
                self.alpha * interval + (1 - self.alpha) * interval_estimate
            )
            previous = int(index)
        if interval_estimate <= 0 or not np.isfinite(interval_estimate):
            raise ModelEligibilityError(
                "Croston could not estimate a valid inter-demand interval."
            )
        self._rate = demand_estimate / interval_estimate

    def _forecast(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._rate, dtype=float)


@dataclass(frozen=True)
class _SeriesInput:
    product_id: object | None
    frame: pd.DataFrame
    diagnostics: SeriesDemandDiagnostics | None


def evaluate_forecasting_models(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    models: Sequence[ForecastModel],
    *,
    horizon: int,
) -> ForecastComparison:
    """Compare candidates on the same latest contiguous chronological window.

    The test window is the most recent contiguous block of eligible actuals
    of the requested length. Training uses the latest contiguous eligible
    prefix immediately before that window. Unknown periods are neither zero-
    filled nor crossed by model history.
    """
    _validate_framework_inputs(demand, diagnostics, models, horizon)
    evaluations: list[ModelEvaluation] = []
    for series in _series_inputs(demand, diagnostics):
        test = _last_eligible_run(series.frame, horizon)
        if test is None:
            reason = (
                f"No contiguous run of {horizon} eligible actual periods "
                "exists for a common evaluation window."
            )
            evaluations.extend(
                _ineligible_evaluation(series.product_id, model, reason)
                for model in models
            )
            continue
        test_start = int(test[0])
        test_end = int(test[-1])
        training = _trailing_eligible_prefix(
            series.frame, test_start
        )
        training_diagnostics = _diagnose_training_series(
            demand,
            series,
            test_start,
            diagnostics.thresholds,
        )
        actual = series.frame.loc[test, "demand_quantity"].to_numpy(dtype=float)
        start_period = pd.Timestamp(series.frame.loc[test_start, "period"])
        end_period = pd.Timestamp(series.frame.loc[test_end, "period"])

        for model in models:
            candidate = _fresh_model(model)
            try:
                candidate.fit(training, diagnostics=training_diagnostics)
                prediction = candidate.forecast(horizon)
            except ModelEligibilityError as exc:
                evaluations.append(
                    ModelEvaluation(
                        product_id=series.product_id,
                        model_name=model.name,
                        eligible=False,
                        evaluation_start=start_period,
                        evaluation_end=end_period,
                        metrics=None,
                        warning=str(exc),
                    )
                )
                continue
            metrics = evaluate_forecasts(actual, prediction)
            evaluations.append(
                ModelEvaluation(
                    product_id=series.product_id,
                    model_name=model.name,
                    eligible=True,
                    evaluation_start=start_period,
                    evaluation_end=end_period,
                    metrics=metrics,
                    warning=None,
                )
            )
    return ForecastComparison(
        frequency=demand.frequency,
        horizon=horizon,
        evaluations=tuple(evaluations),
    )


def evaluate_forecasts(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> EvaluationMetrics:
    """Calculate MAE, RMSE, WAPE, and totals for a single evaluation window."""
    actual_values = _validated_demand(actual, "actual values")
    forecast_values = _validated_predictions(predicted, len(actual_values))
    errors = actual_values - forecast_values
    actual_total = float(np.sum(actual_values))
    forecast_total = float(np.sum(forecast_values))
    wape = (
        float(np.sum(np.abs(errors)) / actual_total * 100)
        if actual_total > 0
        else None
    )
    return EvaluationMetrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        wape=wape,
        actual_total=actual_total,
        forecast_total=forecast_total,
        evaluated_observations=len(actual_values),
    )


def select_best_model(
    comparison: ForecastComparison,
    *,
    product_id: object | None = None,
) -> ModelEvaluation:
    """Select the lowest-MAE eligible model from common-window scores."""
    candidates = [
        result
        for result in comparison.evaluations
        if result.product_id == product_id
        and result.eligible
        and result.metrics is not None
    ]
    if not candidates:
        raise ModelEligibilityError(
            f"No eligible evaluated models for product {product_id!r}."
        )
    return min(candidates, key=lambda result: result.metrics.mae)


def forecast_demand_models(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    models: Sequence[ForecastModel],
    *,
    horizon: int,
) -> tuple[DemandForecast, ...]:
    """Fit eligible candidates and return validated future forecast rows.

    Future forecasts are withheld for a series that ends in an unknown period,
    because doing otherwise would silently bridge an unobserved interval.
    """
    _validate_framework_inputs(
        demand,
        diagnostics,
        models,
        horizon,
        require_naive_benchmark=False,
    )
    forecasts: list[DemandForecast] = []
    for series in _series_inputs(demand, diagnostics):
        frame = series.frame
        warning: str | None = None
        if not bool(frame.iloc[-1]["training_eligible"]):
            warning = (
                "Latest period is unknown; future forecasting is withheld "
                "rather than bridging an unobserved period."
            )
        else:
            training = _trailing_eligible_prefix(frame, len(frame))
            for model in models:
                candidate = _fresh_model(model)
                try:
                    candidate.fit(training, diagnostics=series.diagnostics)
                    predictions = candidate.forecast(horizon)
                except ModelEligibilityError as exc:
                    forecasts.append(
                        DemandForecast(
                            product_id=series.product_id,
                            forecast_period=None,
                            predicted_demand=None,
                            model_name=model.name,
                            forecast_horizon=horizon,
                            model_eligible=False,
                            warnings=(str(exc),),
                            demand_pattern=(
                                series.diagnostics.classifications
                                if series.diagnostics
                                else ()
                            ),
                        )
                    )
                    continue
                future_periods = _future_periods(
                    pd.Timestamp(frame["period"].iloc[-1]),
                    demand.frequency,
                    horizon,
                )
                pattern = (
                    series.diagnostics.classifications
                    if series.diagnostics
                    else ()
                )
                for step, (period, prediction) in enumerate(
                    zip(future_periods, predictions, strict=True),
                    start=1,
                ):
                    forecasts.append(
                        DemandForecast(
                            product_id=series.product_id,
                            forecast_period=period,
                            predicted_demand=float(prediction),
                            model_name=model.name,
                            forecast_horizon=step,
                            model_eligible=True,
                            warnings=(),
                            demand_pattern=pattern,
                        )
                    )
            continue

        pattern = (
            series.diagnostics.classifications
            if series.diagnostics
            else ()
        )
        for model in models:
            forecasts.append(
                DemandForecast(
                    product_id=series.product_id,
                    forecast_period=None,
                    predicted_demand=None,
                    model_name=model.name,
                    forecast_horizon=horizon,
                    model_eligible=False,
                    warnings=(warning,) if warning else (),
                    demand_pattern=pattern,
                )
            )
    return tuple(forecasts)


def _validate_framework_inputs(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    models: Sequence[ForecastModel],
    horizon: int,
    *,
    require_naive_benchmark: bool = True,
) -> None:
    if not isinstance(demand, ForecastReadyDemandData):
        raise TypeError("demand must be ForecastReadyDemandData.")
    if not isinstance(diagnostics, DemandPatternDiagnostics):
        raise TypeError("diagnostics must be DemandPatternDiagnostics.")
    if diagnostics.frequency != demand.frequency:
        raise ValueError("Demand and diagnostics frequencies do not match.")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise ValueError("horizon must be a positive integer.")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if not models:
        raise ValueError("At least one forecasting model is required.")
    if require_naive_benchmark and not any(
        isinstance(model, NaiveDemandModel) for model in models
    ):
        raise ValueError("Model comparisons must include a Naive benchmark.")
    if any(not isinstance(model, ForecastModel) for model in models):
        raise TypeError("models must contain ForecastModel instances.")
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique within a comparison.")


def _series_inputs(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
) -> tuple[_SeriesInput, ...]:
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
    product_available = "product_id" in frame.columns
    keys = ["period"] + (["product_id"] if product_available else [])
    if frame.duplicated(keys).any():
        raise ValueError("Demand contains duplicate period/product keys.")
    if frame["period"].isna().any() or not pd.api.types.is_datetime64_any_dtype(
        frame["period"].dtype
    ):
        raise ValueError("Demand periods must be valid datetime values.")
    if frame["period"].dt.tz is not None:
        raise ValueError("Demand periods must be timezone-naive.")
    valid_statuses = {
        "observed_demand",
        "observed_zero_demand",
        "confirmed_zero_demand",
        "unknown_unobserved_period",
    }
    if frame["observation_status"].isna().any() or not frame[
        "observation_status"
    ].isin(valid_statuses).all():
        raise ValueError("Demand contains unsupported observation statuses.")
    if not pd.api.types.is_bool_dtype(frame["training_eligible"].dtype):
        raise ValueError("training_eligible must contain boolean values.")
    expected_eligible = frame["observation_status"].isin(
        {
            "observed_demand",
            "observed_zero_demand",
            "confirmed_zero_demand",
        }
    )
    if not frame["training_eligible"].eq(expected_eligible).all():
        raise ValueError(
            "training_eligible conflicts with observation_status."
        )
    known = frame["training_eligible"].astype(bool)
    numeric = pd.to_numeric(frame["demand_quantity"], errors="coerce")
    if not np.isfinite(numeric.loc[known].to_numpy(dtype=float)).all():
        raise ValueError("Eligible demand values must be finite.")
    if (numeric.loc[known] < 0).any():
        raise ValueError("Eligible demand values cannot be negative.")
    if numeric.loc[~known].notna().any():
        raise ValueError("Unknown periods must not contain demand values.")
    zero_status = frame["observation_status"].isin(
        {"observed_zero_demand", "confirmed_zero_demand"}
    )
    if numeric.loc[zero_status].ne(0).any():
        raise ValueError("Zero-demand statuses must have zero values.")
    if numeric.loc[
        frame["observation_status"].eq("observed_demand")
    ].eq(0).any():
        raise ValueError(
            "Observed zero values must use observed_zero_demand status."
        )
    if product_available and frame["product_id"].isna().any():
        raise ValueError("Product identifiers cannot be missing.")

    if product_available:
        groups = list(frame.groupby("product_id", sort=False, dropna=False))
    else:
        groups = [(None, frame)]
    diagnostic_map = {
        item.product_id: item
        for item in diagnostics.series
    }
    if len(diagnostic_map) != len(diagnostics.series):
        raise ValueError("Diagnostics contain duplicate product identifiers.")
    if len(groups) != len(diagnostics.series):
        raise ValueError("Demand and diagnostics series counts do not match.")

    output: list[_SeriesInput] = []
    frequency_offset = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}[
        demand.frequency
    ]
    for product, group in groups:
        group = group.sort_values("period", kind="stable").reset_index(drop=True)
        if demand.frequency == "daily":
            aligned = group["period"].dt.normalize()
        elif demand.frequency == "weekly":
            aligned = group["period"].dt.to_period("W-SUN").dt.start_time
        else:
            aligned = group["period"].dt.to_period("M").dt.start_time
        if group["period"].ne(aligned).any():
            raise ValueError(
                f"Product {product!r} has periods misaligned to "
                f"{demand.frequency}."
            )
        grid = pd.date_range(
            group["period"].min(),
            group["period"].max(),
            freq=frequency_offset,
        )
        if not pd.DatetimeIndex(group["period"]).equals(grid):
            raise ValueError(
                f"Product {product!r} has an incomplete period grid; "
                "unknown periods must be explicit."
            )
        pattern = diagnostic_map.get(product) if product_available else (
            diagnostics.series[0] if diagnostics.series else None
        )
        if product_available and pattern is None:
            raise ValueError(
                f"Diagnostics are missing product {product!r}."
            )
        output.append(_SeriesInput(product, group, pattern))
    return tuple(output)


def _last_eligible_run(
    frame: pd.DataFrame,
    horizon: int,
) -> np.ndarray | None:
    eligible = frame["training_eligible"].to_numpy(dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(eligible):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(eligible)))
    for start_index, end_index in reversed(runs):
        if end_index - start_index >= horizon:
            return np.arange(end_index - horizon, end_index, dtype=int)
    return None


def _trailing_eligible_prefix(
    frame: pd.DataFrame,
    end_exclusive: int,
) -> np.ndarray:
    start = end_exclusive
    eligible = frame["training_eligible"].to_numpy(dtype=bool)
    while start > 0 and eligible[start - 1]:
        start -= 1
    if start == end_exclusive:
        return np.array([], dtype=float)
    return pd.to_numeric(
        frame.iloc[start:end_exclusive]["demand_quantity"]
    ).to_numpy(dtype=float)


def _ineligible_evaluation(
    product_id: object | None,
    model: ForecastModel,
    reason: str,
) -> ModelEvaluation:
    return ModelEvaluation(
        product_id=product_id,
        model_name=model.name,
        eligible=False,
        evaluation_start=None,
        evaluation_end=None,
        metrics=None,
        warning=reason,
    )


def _fresh_model(model: ForecastModel) -> ForecastModel:
    return deepcopy(model)


def _diagnose_training_series(
    demand: ForecastReadyDemandData,
    series: _SeriesInput,
    test_start: int,
    thresholds: DemandPatternThresholds,
) -> SeriesDemandDiagnostics | None:
    """Recompute eligibility evidence on training-only data to avoid leakage."""
    end = test_start
    start = end
    eligible = series.frame["training_eligible"].to_numpy(dtype=bool)
    while start > 0 and eligible[start - 1]:
        start -= 1
    if start == end:
        return None
    training_frame = series.frame.iloc[start:end].copy().reset_index(drop=True)
    training_ready = ForecastReadyDemandData(
        data=training_frame,
        frequency=demand.frequency,
        coverage_by_product=pd.DataFrame(),
        quality_report=demand.quality_report,
    )
    train_diagnostics = diagnose_demand_patterns(
        training_ready,
        thresholds=thresholds,
    )
    return train_diagnostics.series[0] if train_diagnostics.series else None


def _validated_demand(
    values: Sequence[float] | np.ndarray,
    label: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values.") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array.")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite values.")
    if (result < 0).any():
        raise ValueError(f"{label} cannot contain negative demand.")
    return result.copy()


def _validated_predictions(
    values: Sequence[float] | np.ndarray,
    expected_length: int,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions must be numeric.") from exc
    if result.ndim != 1 or len(result) != expected_length:
        raise ValueError(
            "Predictions must be one-dimensional and match actual length."
        )
    if not np.isfinite(result).all():
        raise InvalidForecastError("Predictions contain non-finite values.")
    if (result < 0).any():
        raise InvalidForecastError("Predictions cannot be negative.")
    return result.copy()


def _future_periods(
    last_period: pd.Timestamp,
    frequency: str,
    horizon: int,
) -> pd.DatetimeIndex:
    offset = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}[frequency]
    return pd.date_range(
        last_period,
        periods=horizon + 1,
        freq=offset,
    )[1:]
