"""Heuristic demand-pattern diagnostics for the generic user workflow."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.preprocessing.user_demand_preparation import (
    ForecastReadyDemandData,
)


Classification = Literal[
    "regular",
    "intermittent",
    "erratic",
    "lumpy",
    "seasonal",
    "insufficient_history",
    "data_quality_concern",
]

_OBSERVED_STATUS = "observed_demand"
_OBSERVED_ZERO_STATUS = "observed_zero_demand"
_CONFIRMED_ZERO_STATUS = "confirmed_zero_demand"
_UNKNOWN_STATUS = "unknown_unobserved_period"
_ELIGIBLE_STATUSES = {
    _OBSERVED_STATUS,
    _OBSERVED_ZERO_STATUS,
    _CONFIRMED_ZERO_STATUS,
}
_ALL_STATUSES = _ELIGIBLE_STATUSES | {_UNKNOWN_STATUS}


@dataclass(frozen=True)
class DemandPatternThresholds:
    """Configurable thresholds for descriptive, non-statistical labels.

    Defaults are screening heuristics, not significance tests: at least 8
    eligible periods for basic classification; 30% zeros for intermittency;
    CV 0.5 separates lower from higher variability; three complete seasonal
    cycles and strength 0.6 for seasonal evidence; and 90% coverage for a
    data-quality concern.
    """

    min_history_periods: int = 8
    intermittent_zero_fraction: float = 0.30
    regular_cv_max: float = 0.50
    lumpy_positive_cv_min: float = 0.50
    minimum_seasonal_cycles: int = 3
    seasonal_strength_min: float = 0.60
    minimum_coverage: float = 0.90
    trend_r_squared_min: float = 0.60
    trend_relative_change_min: float = 0.20

    def __post_init__(self) -> None:
        if self.min_history_periods < 2:
            raise ValueError("min_history_periods must be at least 2.")
        if self.minimum_seasonal_cycles < 2:
            raise ValueError("minimum_seasonal_cycles must be at least 2.")
        for field_name in (
            "intermittent_zero_fraction",
            "regular_cv_max",
            "lumpy_positive_cv_min",
            "seasonal_strength_min",
            "minimum_coverage",
            "trend_r_squared_min",
        ):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value < 0 or value > 1:
                raise ValueError(f"{field_name} must be between 0 and 1.")
        if (
            not np.isfinite(self.trend_relative_change_min)
            or self.trend_relative_change_min < 0
        ):
            raise ValueError("trend_relative_change_min must be non-negative.")


@dataclass(frozen=True)
class SeriesDemandDiagnostics:
    """Descriptive metrics and heuristic labels for one demand series."""

    product_id: object | None
    history_length: int
    eligible_periods: int
    coverage_ratio: float
    unknown_periods: int
    confirmed_zero_periods: int
    observed_zero_periods: int
    zero_demand_percentage: float | None
    mean_demand: float | None
    standard_deviation: float | None
    coefficient_of_variation: float | None
    trend_direction: Literal["increasing", "decreasing", "none", "undetermined"]
    trend_slope_per_period: float | None
    trend_r_squared: float | None
    trend_evidence: bool
    seasonal_period: int | None
    seasonal_strength: float | None
    seasonality_evidence: bool
    intermittent_behavior: bool
    classifications: tuple[Classification, ...]
    primary_classification: Classification
    explanations: tuple[str, ...]
    warnings: tuple[str, ...]
    expected_start_period: pd.Timestamp | None = None
    expected_end_period: pd.Timestamp | None = None
    missing_before_first_observation: int = 0
    missing_internal_periods: int = 0
    missing_after_last_observation: int = 0
    excluded_before_first_observation_periods: int = 0
    excluded_after_last_observation_periods: int = 0


@dataclass(frozen=True)
class DemandPatternDiagnostics:
    """Diagnostics for all series represented by a prepared demand object."""

    frequency: str
    thresholds: DemandPatternThresholds
    series: tuple[SeriesDemandDiagnostics, ...]
    warnings: tuple[str, ...]


def diagnose_demand_patterns(
    demand: ForecastReadyDemandData,
    *,
    thresholds: DemandPatternThresholds | None = None,
) -> DemandPatternDiagnostics:
    """Describe each series using eligible values without filling unknowns.

    Trend and seasonality labels are evidence-based heuristics only. Seasonal
    evidence is withheld unless the full period grid has enough complete
    cycles; therefore unknown periods are never silently skipped or treated
    as zeros during seasonal analysis.
    """
    if not isinstance(demand, ForecastReadyDemandData):
        raise TypeError("demand must be a ForecastReadyDemandData instance.")
    thresholds = thresholds or DemandPatternThresholds()
    frame = demand.data.copy(deep=True)
    required = {
        "period",
        "demand_quantity",
        "observation_status",
        "training_eligible",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Forecast-ready data is missing columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("Forecast-ready demand data cannot be empty.")
    if not pd.api.types.is_datetime64_any_dtype(frame["period"].dtype):
        raise ValueError("Periods must use normalized datetime values.")
    if frame["period"].isna().any() or frame["period"].dt.tz is not None:
        raise ValueError("Periods must be valid timezone-naive datetimes.")
    if demand.frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("Unsupported demand frequency.")
    if demand.frequency == "daily":
        aligned_periods = frame["period"].dt.normalize()
        grid_frequency = "D"
    elif demand.frequency == "weekly":
        aligned_periods = (
            frame["period"].dt.to_period("W-SUN").dt.start_time
        )
        grid_frequency = "W-MON"
    else:
        aligned_periods = frame["period"].dt.to_period("M").dt.start_time
        grid_frequency = "MS"
    if frame["period"].ne(aligned_periods).any():
        raise ValueError(
            f"Periods are not aligned to the {demand.frequency} frequency."
        )
    if frame["observation_status"].isna().any() or not frame[
        "observation_status"
    ].isin(_ALL_STATUSES).all():
        raise ValueError("Demand contains unsupported observation statuses.")
    expected_eligibility = frame["observation_status"].isin(_ELIGIBLE_STATUSES)
    if not frame["training_eligible"].eq(expected_eligibility).all():
        raise ValueError(
            "training_eligible conflicts with observation_status."
        )
    if frame["period"].duplicated().any() and "product_id" not in frame.columns:
        raise ValueError("Single-series data contains duplicate periods.")
    if "product_id" in frame.columns and frame.duplicated(
        ["product_id", "period"]
    ).any():
        raise ValueError("Demand contains duplicate product-period keys.")

    known = frame["training_eligible"]
    values = pd.to_numeric(frame["demand_quantity"], errors="coerce")
    known_values = values.loc[known].to_numpy(dtype=float)
    if not np.isfinite(known_values).all() or (known_values < 0).any():
        raise ValueError("Eligible demand must be finite and non-negative.")
    if values.loc[~known].notna().any():
        raise ValueError("Unknown periods must not contain demand values.")
    zero_status = frame["observation_status"].isin(
        [_OBSERVED_ZERO_STATUS, _CONFIRMED_ZERO_STATUS]
    )
    if values.loc[zero_status].ne(0).any():
        raise ValueError("Zero-demand status rows must have zero demand.")
    if values.loc[
        frame["observation_status"].eq(_OBSERVED_STATUS)
    ].eq(0).any():
        raise ValueError(
            "Observed zero rows must use observed_zero_demand status."
        )

    product_available = "product_id" in frame.columns
    if product_available and frame["product_id"].isna().any():
        raise ValueError("Product identifiers cannot be missing.")
    if product_available:
        groups = list(frame.groupby("product_id", sort=False, dropna=False))
    else:
        groups = [(None, frame)]

    for product, group in groups:
        expected_grid = pd.date_range(
            group["period"].min(),
            group["period"].max(),
            freq=grid_frequency,
        )
        actual_grid = pd.DatetimeIndex(group["period"].sort_values())
        if not actual_grid.equals(expected_grid):
            product_text = (
                f" for product {product!r}" if product_available else ""
            )
            raise ValueError(
                f"Prepared period grid{product_text} is incomplete; represent "
                "absent periods explicitly as unknown before diagnostics."
            )

    diagnostics = tuple(
        _diagnose_series(
            product_id=product if product_available else None,
            group=group,
            frequency=demand.frequency,
            thresholds=thresholds,
            coverage_metadata=_coverage_for_product(
                demand.coverage_by_product,
                product if product_available else None,
                product_available,
            ),
        )
        for product, group in groups
    )
    return DemandPatternDiagnostics(
        frequency=demand.frequency,
        thresholds=thresholds,
        series=diagnostics,
        warnings=(
            "Classifications are configurable screening heuristics, not "
            "statistical tests or proven descriptions of future behavior.",
        ),
    )


def _diagnose_series(
    *,
    product_id: object | None,
    group: pd.DataFrame,
    frequency: str,
    thresholds: DemandPatternThresholds,
    coverage_metadata: pd.Series | None = None,
) -> SeriesDemandDiagnostics:
    group = group.sort_values("period", kind="stable")
    eligible = group.loc[group["training_eligible"]]
    values = pd.to_numeric(eligible["demand_quantity"]).to_numpy(dtype=float)
    history_length = len(group)
    eligible_count = len(values)
    unknown_count = int(
        group["observation_status"].eq(_UNKNOWN_STATUS).sum()
    )
    confirmed_zeros = int(
        group["observation_status"].eq(_CONFIRMED_ZERO_STATUS).sum()
    )
    observed_zeros = int(
        group["observation_status"].eq(_OBSERVED_ZERO_STATUS).sum()
    )
    coverage = eligible_count / history_length if history_length else 0.0
    zero_count = int(np.count_nonzero(values == 0))
    zero_percentage = (
        100.0 * zero_count / eligible_count if eligible_count else None
    )
    mean = float(np.mean(values)) if eligible_count else None
    standard_deviation = (
        float(np.std(values, ddof=0)) if eligible_count else None
    )
    coefficient = (
        standard_deviation / mean
        if mean is not None and mean > 0
        else None
    )
    trend = _trend_evidence(eligible, values, frequency, thresholds)
    seasonal_period = _seasonal_period(frequency)
    seasonal_strength, seasonal_evidence, seasonal_reason = _seasonal_evidence(
        group,
        values,
        frequency,
        thresholds,
    )
    intermittent = bool(
        zero_percentage is not None
        and zero_percentage / 100 >= thresholds.intermittent_zero_fraction
    )

    classifications: list[Classification] = []
    explanations: list[str] = []
    warnings: list[str] = []
    if eligible_count < thresholds.min_history_periods:
        classifications.append("insufficient_history")
        explanations.append(
            f"{eligible_count} eligible periods are below the configured "
            f"minimum of {thresholds.min_history_periods}."
        )
    else:
        positive_values = values[values > 0]
        positive_mean = (
            float(np.mean(positive_values)) if positive_values.size else 0.0
        )
        positive_cv = (
            float(np.std(positive_values, ddof=0) / positive_mean)
            if positive_mean > 0
            else 0.0
        )
        if intermittent:
            if (
                positive_values.size > 0
                and positive_cv >= thresholds.lumpy_positive_cv_min
            ):
                classifications.append("lumpy")
                explanations.append(
                    f"{zero_percentage:.1f}% of eligible periods are zero and "
                    f"positive-demand CV is {positive_cv:.3f}, meeting the "
                    "configured lumpy thresholds."
                )
            else:
                classifications.append("intermittent")
                explanations.append(
                    f"{zero_percentage:.1f}% of eligible periods are zero, "
                    "meeting the configured intermittency threshold."
                )
            if positive_values.size == 0:
                explanations.append(
                    "All eligible demand is zero; no positive-demand "
                    "variability can be estimated."
                )
        elif coefficient is not None and coefficient <= thresholds.regular_cv_max:
            classifications.append("regular")
            explanations.append(
                f"Demand CV {coefficient:.3f} is at or below the configured "
                f"regularity threshold {thresholds.regular_cv_max:.3f}."
            )
        elif coefficient is not None:
            classifications.append("erratic")
            explanations.append(
                f"Demand CV {coefficient:.3f} exceeds the configured "
                f"regularity threshold {thresholds.regular_cv_max:.3f}."
            )
        else:
            classifications.append("intermittent")
            explanations.append(
                "Mean demand is zero, so coefficient of variation is "
                "undefined; the series contains no positive demand."
            )

    if coverage < thresholds.minimum_coverage:
        classifications.append("data_quality_concern")
        explanations.append(
            f"Coverage {coverage:.1%} is below the configured minimum "
            f"{thresholds.minimum_coverage:.1%}."
        )
    if unknown_count:
        warnings.append(
            f"{unknown_count} unknown periods were excluded from demand "
            "statistics and trend calculations."
        )
    if seasonal_reason:
        warnings.append(seasonal_reason)
    if seasonal_evidence:
        classifications.append("seasonal")
        explanations.append(
            f"Seasonal strength {seasonal_strength:.3f} meets the configured "
            f"threshold {thresholds.seasonal_strength_min:.3f} across at "
            f"least {thresholds.minimum_seasonal_cycles} complete cycles."
        )
    if trend["evidence"]:
        explanations.append(
            f"Trend evidence is {trend['direction']} with R-squared "
            f"{trend['r_squared']:.3f} and relative fitted change "
            f"{trend['relative_change']:.1%}."
        )
    elif eligible_count >= 4:
        explanations.append(
            "The fitted trend did not meet both configured fit and "
            "relative-change thresholds."
        )

    primary = _primary_classification(classifications)
    return SeriesDemandDiagnostics(
        product_id=product_id,
        history_length=history_length,
        eligible_periods=eligible_count,
        coverage_ratio=coverage,
        unknown_periods=unknown_count,
        confirmed_zero_periods=confirmed_zeros,
        observed_zero_periods=observed_zeros,
        zero_demand_percentage=zero_percentage,
        mean_demand=mean,
        standard_deviation=standard_deviation,
        coefficient_of_variation=coefficient,
        trend_direction=trend["direction"],
        trend_slope_per_period=trend["slope"],
        trend_r_squared=trend["r_squared"],
        trend_evidence=bool(trend["evidence"]),
        seasonal_period=seasonal_period,
        seasonal_strength=seasonal_strength,
        seasonality_evidence=seasonal_evidence,
        intermittent_behavior=intermittent,
        classifications=tuple(classifications),
        primary_classification=primary,
        explanations=tuple(explanations),
        warnings=tuple(warnings),
        expected_start_period=(
            coverage_metadata.get("expected_start_period")
            if coverage_metadata is not None
            else None
        ),
        expected_end_period=(
            coverage_metadata.get("expected_end_period")
            if coverage_metadata is not None
            else None
        ),
        missing_before_first_observation=int(
            coverage_metadata.get("missing_before_first_observation", 0)
            if coverage_metadata is not None
            else 0
        ),
        missing_internal_periods=int(
            coverage_metadata.get("missing_internal_periods", 0)
            if coverage_metadata is not None
            else 0
        ),
        missing_after_last_observation=int(
            coverage_metadata.get("missing_after_last_observation", 0)
            if coverage_metadata is not None
            else 0
        ),
        excluded_before_first_observation_periods=int(
            coverage_metadata.get("excluded_before_first_observation_periods", 0)
            if coverage_metadata is not None
            else 0
        ),
        excluded_after_last_observation_periods=int(
            coverage_metadata.get("excluded_after_last_observation_periods", 0)
            if coverage_metadata is not None
            else 0
        ),
    )


def _coverage_for_product(
    coverage: pd.DataFrame,
    product_id: object | None,
    product_available: bool,
) -> pd.Series | None:
    if coverage.empty:
        return None
    if not product_available:
        return coverage.iloc[0]
    matching = coverage.loc[coverage["product_id"].eq(product_id)]
    return matching.iloc[0] if not matching.empty else None


def _primary_classification(
    classifications: list[Classification],
) -> Classification:
    for label in (
        "data_quality_concern",
        "insufficient_history",
        "seasonal",
        "lumpy",
        "intermittent",
        "erratic",
        "regular",
    ):
        if label in classifications:
            return label
    return "insufficient_history"


def _trend_evidence(
    eligible: pd.DataFrame,
    values: np.ndarray,
    frequency: str,
    thresholds: DemandPatternThresholds,
) -> dict[str, object]:
    result: dict[str, object] = {
        "direction": "undetermined",
        "slope": None,
        "r_squared": None,
        "relative_change": None,
        "evidence": False,
    }
    if len(values) < 4:
        return result
    x = _period_ordinals(eligible["period"], frequency)
    x = x - x[0]
    if np.unique(x).size < 2:
        return result
    slope, intercept = np.polyfit(x.astype(float), values, 1)
    fitted = intercept + slope * x
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    residual_sum_squares = float(np.sum((values - fitted) ** 2))
    r_squared = (
        max(0.0, 1.0 - residual_sum_squares / total_sum_squares)
        if total_sum_squares > 0
        else 0.0
    )
    mean = float(np.mean(values))
    relative_change = (
        abs(float(slope) * float(x[-1] - x[0]) / mean)
        if mean > 0
        else 0.0
    )
    direction = (
        "increasing"
        if slope > 0
        else "decreasing" if slope < 0 else "none"
    )
    evidence = (
        direction != "none"
        and r_squared >= thresholds.trend_r_squared_min
        and relative_change >= thresholds.trend_relative_change_min
    )
    result.update(
        direction=direction,
        slope=float(slope),
        r_squared=r_squared,
        relative_change=relative_change,
        evidence=evidence,
    )
    return result


def _seasonal_period(frequency: str) -> int:
    return {"daily": 7, "weekly": 52, "monthly": 12}[frequency]


def _seasonal_evidence(
    group: pd.DataFrame,
    eligible_values: np.ndarray,
    frequency: str,
    thresholds: DemandPatternThresholds,
) -> tuple[float | None, bool, str | None]:
    period = _seasonal_period(frequency)
    required_length = period * thresholds.minimum_seasonal_cycles
    if len(group) < required_length:
        return (
            None,
            False,
            f"Seasonality not assessed: {len(group)} periods do not provide "
            f"{thresholds.minimum_seasonal_cycles} complete {period}-period "
            "cycles.",
        )
    if group["observation_status"].eq(_UNKNOWN_STATUS).any():
        return (
            None,
            False,
            "Seasonality not assessed because unknown periods interrupt the "
            "history; unknown values were not filled.",
        )
    if len(eligible_values) != len(group):
        return None, False, "Seasonality not assessed due to incomplete values."
    if not np.isfinite(eligible_values).all():
        return None, False, "Seasonality not assessed due to non-finite values."
    if float(np.var(eligible_values)) == 0:
        return 0.0, False, "Seasonality not assessed for a constant series."

    ordinals = _period_ordinals(group["period"], frequency)
    seasonal_indices = ordinals % period
    values = pd.DataFrame(
        {"season": seasonal_indices, "value": eligible_values}
    )
    seasonal_means = values.groupby("season")["value"].mean()
    fitted = values["season"].map(seasonal_means).to_numpy(dtype=float)
    residual_variance = float(np.var(eligible_values - fitted))
    strength = float(
        np.clip(1.0 - residual_variance / float(np.var(eligible_values)), 0, 1)
    )
    return (
        strength,
        strength >= thresholds.seasonal_strength_min,
        None,
    )


def _period_ordinals(periods: pd.Series, frequency: str) -> np.ndarray:
    timestamps = pd.to_datetime(periods)
    if timestamps.dt.tz is not None:
        raise ValueError("Periods must be timezone-naive.")
    day_ordinals = (
        timestamps.astype("int64").to_numpy() // 86_400_000_000_000
    )
    if frequency == "daily":
        return day_ordinals.astype(np.int64)
    if frequency == "weekly":
        return (day_ordinals // 7).astype(np.int64)
    if frequency == "monthly":
        return (
            timestamps.dt.year.to_numpy() * 12
            + timestamps.dt.month.to_numpy()
        ).astype(np.int64)
    raise ValueError("Unsupported demand frequency.")
