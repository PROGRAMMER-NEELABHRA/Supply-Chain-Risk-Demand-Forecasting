"""Preparation utilities for validated user-provided demand observations."""

from dataclasses import dataclass, replace
from numbers import Number
from typing import Literal

import numpy as np
import pandas as pd

from src.data.user_demand import (
    DemandDataQualityReport,
    Frequency,
    PreparedDemandData,
    prepare_user_demand_data,
)


_OBSERVED_STATUS = "observed_demand"
_OBSERVED_ZERO_STATUS = "observed_zero_demand"
_CONFIRMED_ZERO_STATUS = "confirmed_zero_demand"
_UNKNOWN_STATUS = "unknown_unobserved_period"
_KNOWN_STATUSES = {
    _OBSERVED_STATUS,
    _OBSERVED_ZERO_STATUS,
    _CONFIRMED_ZERO_STATUS,
}
_ALL_STATUSES = _KNOWN_STATUSES | {_UNKNOWN_STATUS}


@dataclass(frozen=True)
class TransactionAggregationSummary:
    """Counts and method used to aggregate transaction-level input."""

    input_rows: int
    aggregated_period_rows: int
    method: Literal["sum_transactions"]
    exact_duplicate_rows: int = 0


@dataclass(frozen=True)
class ForecastReadyDemandData:
    """Normalized demand series plus coverage and training eligibility.

    Unknown periods have null demand and ``training_eligible=False``.
    Observed demand, observed zeros, and explicitly confirmed zeros are
    eligible. No values are interpolated or synthesized.
    """

    data: pd.DataFrame
    frequency: Frequency
    coverage_by_product: pd.DataFrame
    quality_report: DemandDataQualityReport
    aggregation_summary: TransactionAggregationSummary | None = None
    expected_coverage_start: pd.Timestamp | None = None
    expected_coverage_end: pd.Timestamp | None = None
    products_expected_throughout_coverage: bool = False


def prepare_forecast_demand(
    prepared: PreparedDemandData,
) -> ForecastReadyDemandData:
    """Add training eligibility and preserve expected/product coverage metadata."""
    if not isinstance(prepared, PreparedDemandData):
        raise TypeError("prepared must be a PreparedDemandData instance.")
    if not isinstance(prepared.quality_report, DemandDataQualityReport):
        raise TypeError("prepared.quality_report has an unsupported type.")

    source = prepared.data.copy(deep=True)
    required = {"period", "demand_quantity", "observation_status"}
    missing_columns = required - set(source.columns)
    if missing_columns:
        raise ValueError(
            f"Prepared demand is missing columns: {sorted(missing_columns)}"
        )
    product_available = "product_id" in source.columns
    key_columns = ["period"] + (["product_id"] if product_available else [])
    if source.empty:
        raise ValueError("Prepared demand cannot be empty.")
    if source["period"].isna().any():
        raise ValueError("Prepared demand contains missing periods.")
    if not pd.api.types.is_datetime64_any_dtype(source["period"].dtype):
        raise ValueError("Prepared periods must use normalized datetime values.")
    if source["period"].dt.tz is not None:
        raise ValueError("Prepared periods must be timezone-naive.")
    frequency = prepared.quality_report.frequency
    if frequency == "daily":
        aligned_periods = source["period"].dt.normalize()
    elif frequency == "weekly":
        aligned_periods = source["period"].dt.to_period("W-SUN").dt.start_time
    elif frequency == "monthly":
        aligned_periods = source["period"].dt.to_period("M").dt.start_time
    else:
        raise ValueError("Prepared data has an unsupported frequency.")
    if source["period"].ne(aligned_periods).any():
        raise ValueError(
            f"Prepared periods are not aligned to the {frequency} frequency."
        )
    if product_available and source["product_id"].isna().any():
        raise ValueError("Prepared demand contains missing product identifiers.")
    if source["observation_status"].isna().any() or not source[
        "observation_status"
    ].isin(_ALL_STATUSES).all():
        raise ValueError("Prepared demand contains unsupported observation statuses.")
    if source.duplicated(key_columns).any():
        raise ValueError("Prepared demand contains duplicate period/product keys.")

    numeric = pd.to_numeric(source["demand_quantity"], errors="coerce")
    known = source["observation_status"].isin(_KNOWN_STATUSES)
    unknown = source["observation_status"].eq(_UNKNOWN_STATUS)
    if numeric.loc[known].isna().any() or not np.isfinite(
        numeric.loc[known].to_numpy(dtype=float)
    ).all():
        raise ValueError("Known demand observations must contain finite values.")
    if (numeric.loc[known] < 0).any():
        raise ValueError("Known demand observations cannot be negative.")
    if numeric.loc[unknown].notna().any():
        raise ValueError("Unknown periods must not contain demand values.")

    zero_status = source["observation_status"].isin(
        [_OBSERVED_ZERO_STATUS, _CONFIRMED_ZERO_STATUS]
    )
    if numeric.loc[zero_status].ne(0).any():
        raise ValueError("Zero-demand statuses must have a demand value of zero.")
    if numeric.loc[source["observation_status"].eq(_OBSERVED_STATUS)].eq(0).any():
        raise ValueError(
            "Observed zero values must use the observed_zero_demand status."
        )

    source["demand_quantity"] = numeric
    source["training_eligible"] = known
    source = source.sort_values("period", kind="stable", ignore_index=True)

    groups = (
        source.groupby("product_id", sort=False, dropna=False)
        if product_available
        else [(None, source)]
    )
    coverage_rows: list[dict[str, object]] = []
    for product, group in groups:
        statuses = group["observation_status"]
        expected = len(group)
        observed_count = int(
            statuses.isin([_OBSERVED_STATUS, _OBSERVED_ZERO_STATUS]).sum()
        )
        confirmed_zero_count = int(statuses.eq(_CONFIRMED_ZERO_STATUS).sum())
        unknown_count = int(statuses.eq(_UNKNOWN_STATUS).sum())
        eligible_count = observed_count + confirmed_zero_count
        coverage_row: dict[str, object] = {
            "periods_expected": expected,
            "observed_periods": observed_count,
            "confirmed_zero_periods": confirmed_zero_count,
            "unknown_periods": unknown_count,
            "training_eligible_periods": eligible_count,
            "coverage_ratio": eligible_count / expected if expected else 0.0,
            "first_period": group["period"].min(),
            "last_period": group["period"].max(),
        }
        source_coverage = prepared.coverage_by_product
        if not source_coverage.empty:
            if product_available:
                matching_coverage = source_coverage.loc[
                    source_coverage["product_id"].eq(product)
                ]
            else:
                matching_coverage = source_coverage
            if not matching_coverage.empty:
                for column in (
                    "expected_start_period",
                    "expected_end_period",
                    "first_supplied_period",
                    "last_supplied_period",
                    "missing_before_first_observation",
                    "missing_internal_periods",
                    "missing_after_last_observation",
                    "excluded_before_first_observation_periods",
                    "excluded_after_last_observation_periods",
                ):
                    if column in matching_coverage:
                        coverage_row[column] = matching_coverage.iloc[0][column]
        if product_available:
            coverage_row = {"product_id": product, **coverage_row}
        coverage_rows.append(coverage_row)

    ordered_columns = (
        ["product_id"] if product_available else []
    ) + [
        "periods_expected",
        "observed_periods",
        "confirmed_zero_periods",
        "unknown_periods",
        "training_eligible_periods",
        "coverage_ratio",
        "first_period",
        "last_period",
        "expected_start_period",
        "expected_end_period",
        "first_supplied_period",
        "last_supplied_period",
        "missing_before_first_observation",
        "missing_internal_periods",
        "missing_after_last_observation",
        "excluded_before_first_observation_periods",
        "excluded_after_last_observation_periods",
    ]
    coverage = pd.DataFrame(coverage_rows, columns=ordered_columns)
    return ForecastReadyDemandData(
        data=source,
        frequency=frequency,
        coverage_by_product=coverage,
        quality_report=prepared.quality_report,
        expected_coverage_start=prepared.quality_report.expected_coverage_start,
        expected_coverage_end=prepared.quality_report.expected_coverage_end,
        products_expected_throughout_coverage=(
            prepared.products_expected_throughout_coverage
        ),
    )


def aggregate_transaction_demand(
    dataframe: pd.DataFrame,
    *,
    date_column: str,
    demand_column: str,
    frequency: Frequency,
    aggregation: Literal["sum_transactions"],
    product_column: str | None = None,
    complete_source_coverage: bool = False,
    confirm_missing_periods_are_zero: bool = False,
    expected_start: object | None = None,
    expected_end: object | None = None,
    products_expected_throughout_coverage: bool = False,
) -> ForecastReadyDemandData:
    """Sum explicitly declared additive transactions into chosen periods.

    Each input row is treated as an additive transaction only because callers
    must explicitly set ``aggregation="sum_transactions"``. Dates may be any
    timezone-naive instant; they are mapped to calendar days, Monday-start
    weeks, or calendar months before aggregation. Product identifiers retain
    their original scalar type and value. Optional expected bounds and the
    explicit product-wide expectation declaration follow the same rules as
    :func:`prepare_user_demand_data`.
    """
    if aggregation != "sum_transactions":
        raise ValueError("aggregation must be 'sum_transactions'.")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("Transaction dataset cannot be empty.")
    if not dataframe.columns.is_unique:
        raise ValueError("Transaction dataset contains duplicate column names.")
    if frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("frequency must be daily, weekly, or monthly.")
    if confirm_missing_periods_are_zero and not complete_source_coverage:
        raise ValueError(
            "Confirm complete source coverage before treating missing "
            "periods as zero demand."
        )
    mapped_columns = [date_column, demand_column]
    if product_column is not None:
        mapped_columns.append(product_column)
    if len(set(mapped_columns)) != len(mapped_columns):
        raise ValueError("Date, demand, and product columns must be distinct.")
    missing_columns = [column for column in mapped_columns if column not in dataframe]
    if missing_columns:
        raise ValueError(
            f"Missing mapped transaction columns: {sorted(missing_columns)}"
        )

    source = dataframe.copy(deep=True)
    exact_duplicate_rows = int(source.duplicated(keep="first").sum())
    parsed_dates: list[pd.Timestamp] = []
    invalid_rows: list[tuple[int, str]] = []
    for position, value in enumerate(source[date_column].tolist()):
        if pd.isna(value) or isinstance(value, (bool, np.bool_, Number)):
            invalid_rows.append((position, "date is missing or invalid"))
            parsed_dates.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            invalid_rows.append((position, "date is missing or invalid"))
            parsed_dates.append(pd.NaT)
            continue
        if pd.isna(timestamp) or timestamp.tzinfo is not None:
            invalid_rows.append((position, "date is invalid or timezone-aware"))
            parsed_dates.append(pd.NaT)
            continue
        parsed_dates.append(timestamp)

    demand_values = pd.to_numeric(source[demand_column], errors="coerce")
    numeric_values = demand_values.to_numpy(dtype=float, na_value=np.nan)
    for position, original in enumerate(source[demand_column].tolist()):
        if isinstance(original, (bool, np.bool_)):
            invalid_rows.append((position, "demand must be numeric, not boolean"))
        elif pd.isna(original) or pd.isna(numeric_values[position]):
            invalid_rows.append((position, "demand is missing or non-numeric"))
        elif not np.isfinite(numeric_values[position]):
            invalid_rows.append((position, "demand must be finite"))
        elif numeric_values[position] < 0:
            invalid_rows.append((position, "demand cannot be negative"))

    if product_column is not None:
        for position, value in enumerate(source[product_column].tolist()):
            if pd.isna(value) or (
                isinstance(value, str) and not value.strip()
            ):
                invalid_rows.append((position, "product identifier is missing"))
            elif isinstance(value, (bool, np.bool_)):
                invalid_rows.append(
                    (position, "product identifier cannot be boolean")
                )
            elif not isinstance(value, (str, Number, pd.Timestamp)):
                invalid_rows.append(
                    (position, "product identifier must be a scalar value")
                )

    if invalid_rows:
        detail = "; ".join(
            f"row {position + 1}: {reason}"
            for position, reason in invalid_rows
        )
        raise ValueError(f"Invalid transaction input: {detail}")

    dates = pd.Series(parsed_dates, dtype="datetime64[ns]")
    if frequency == "daily":
        periods = dates.dt.normalize()
    elif frequency == "weekly":
        periods = dates.dt.to_period("W-SUN").dt.start_time
    else:
        periods = dates.dt.to_period("M").dt.start_time

    aggregate_map: dict[tuple[object, ...], float] = {}
    product_values: dict[tuple[object, ...], object] = {}
    for position, (period, demand) in enumerate(
        zip(periods.tolist(), numeric_values, strict=True)
    ):
        if product_column is None:
            key: tuple[object, ...] = (period,)
        else:
            product = source.iloc[position][product_column]
            try:
                hash(product)
            except TypeError as exc:
                raise ValueError(
                    f"Product identifier on row {position + 1} is not hashable."
                ) from exc
            key = (period, type(product), product)
            product_values[key] = product
        aggregate_map[key] = aggregate_map.get(key, 0.0) + float(demand)

    records: list[dict[str, object]] = []
    for key, demand in aggregate_map.items():
        record: dict[str, object] = {
            "period": key[0],
            "demand_quantity": demand,
        }
        if product_column is not None:
            record["product_id"] = product_values[key]
        records.append(record)
    aggregated = pd.DataFrame.from_records(records)
    aggregated_rows = len(aggregated)

    phase_one_result = prepare_user_demand_data(
        aggregated,
        date_column="period",
        demand_column="demand_quantity",
        product_column="product_id" if product_column is not None else None,
        frequency=frequency,
        complete_source_coverage=complete_source_coverage,
        confirm_missing_periods_are_zero=confirm_missing_periods_are_zero,
        expected_start=expected_start,
        expected_end=expected_end,
        products_expected_throughout_coverage=(
            products_expected_throughout_coverage
        ),
    )
    aggregation_warnings = [
        f"Summed {len(source)} additive transaction rows into "
        f"{aggregated_rows} distinct period/series observations. "
        "All input rows were retained; repeated period/product keys are expected "
        "in transaction mode."
    ]
    if exact_duplicate_rows:
        aggregation_warnings.append(
            f"Found {exact_duplicate_rows} exact duplicate input rows. They were "
            "retained in the sums; confirm whether they represent distinct "
            "transactions or accidental duplicates."
        )
    quality_report = replace(
        phase_one_result.quality_report,
        input_rows=len(source),
        valid_rows=len(source),
        validation_warnings=(
            *phase_one_result.quality_report.validation_warnings,
            *aggregation_warnings,
        ),
    )
    phase_one_result = PreparedDemandData(
        data=phase_one_result.data,
        quality_report=quality_report,
        coverage_by_product=phase_one_result.coverage_by_product,
        products_expected_throughout_coverage=(
            phase_one_result.products_expected_throughout_coverage
        ),
    )
    result = prepare_forecast_demand(phase_one_result)
    return ForecastReadyDemandData(
        data=result.data,
        frequency=result.frequency,
        coverage_by_product=result.coverage_by_product,
        quality_report=result.quality_report,
        aggregation_summary=TransactionAggregationSummary(
            input_rows=len(source),
            aggregated_period_rows=aggregated_rows,
            method="sum_transactions",
            exact_duplicate_rows=exact_duplicate_rows,
        ),
        expected_coverage_start=result.expected_coverage_start,
        expected_coverage_end=result.expected_coverage_end,
        products_expected_throughout_coverage=(
            result.products_expected_throughout_coverage
        ),
    )
