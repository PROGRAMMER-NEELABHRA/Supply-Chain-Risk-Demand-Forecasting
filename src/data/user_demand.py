"""Generic ingestion and validation for user-provided demand data."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd


Frequency = Literal["daily", "weekly", "monthly"]

_FREQUENCY_OFFSETS = {
    "daily": "D",
    "weekly": "W-MON",
    "monthly": "MS",
}


@dataclass(frozen=True)
class DemandDataQualityReport:
    """Validation and coverage summary for an uploaded demand dataset."""

    input_rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows: int
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None
    frequency: Frequency
    number_of_products: int
    missing_periods: int
    validation_warnings: tuple[str, ...]
    expected_coverage_start: pd.Timestamp | None = None
    expected_coverage_end: pd.Timestamp | None = None
    missing_before_first_observation: int = 0
    missing_internal_periods: int = 0
    missing_after_last_observation: int = 0


@dataclass(frozen=True)
class PreparedDemandData:
    """Normalized demand periods and their accompanying quality report."""

    data: pd.DataFrame
    quality_report: DemandDataQualityReport
    coverage_by_product: pd.DataFrame = field(default_factory=pd.DataFrame)
    products_expected_throughout_coverage: bool = False


class DemandDataValidationError(ValueError):
    """Raised when input rows violate the generic demand-data contract."""

    def __init__(
        self,
        message: str,
        *,
        quality_report: DemandDataQualityReport,
        invalid_rows: pd.DataFrame,
    ) -> None:
        super().__init__(message)
        self.quality_report = quality_report
        self.invalid_rows = invalid_rows


def prepare_user_demand_data(
    dataframe: pd.DataFrame,
    *,
    date_column: str,
    demand_column: str,
    product_column: str | None = None,
    frequency: Frequency,
    complete_source_coverage: bool = False,
    confirm_missing_periods_are_zero: bool = False,
    expected_start: str | date | datetime | pd.Timestamp | None = None,
    expected_end: str | date | datetime | pd.Timestamp | None = None,
    products_expected_throughout_coverage: bool = False,
) -> PreparedDemandData:
    """Validate and normalize user data without mutating its input.

    Dates must already align to the selected period: calendar day at midnight,
    Monday at midnight for weekly data, or the first day of a month at midnight.
    Missing periods are retained with null demand unless complete source
    coverage and the interpretation of absent periods as zero are both
    explicitly confirmed. Optional expected bounds are normalized to the
    selected frequency. For product data, each product's supplied lifecycle is
    used unless the caller explicitly declares every product expected
    throughout both configured bounds; excluded boundary periods are reported
    separately and are never materialized as history.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("Demand dataset cannot be empty.")
    if not dataframe.columns.is_unique:
        raise ValueError("Demand dataset contains duplicate column names.")
    if frequency not in _FREQUENCY_OFFSETS:
        raise ValueError(
            "frequency must be one of: daily, weekly, monthly."
        )
    if confirm_missing_periods_are_zero and not complete_source_coverage:
        raise ValueError(
            "Confirm complete source coverage before treating missing "
            "periods as zero demand."
        )
    if products_expected_throughout_coverage and product_column is None:
        raise ValueError(
            "Product-wide coverage can only be declared when a product "
            "identifier column is selected."
        )
    if products_expected_throughout_coverage and (
        expected_start is None or expected_end is None
    ):
        raise ValueError(
            "Declare both expected coverage bounds before marking every "
            "product as expected throughout that window."
        )

    mapped_columns = [date_column, demand_column]
    if product_column is not None:
        mapped_columns.append(product_column)
    if len(set(mapped_columns)) != len(mapped_columns):
        raise ValueError("Date, demand, and product columns must be distinct.")
    missing_columns = [
        column for column in mapped_columns if column not in dataframe
    ]
    if missing_columns:
        raise ValueError(
            f"Missing mapped demand columns: {sorted(missing_columns)}"
        )

    source = dataframe.copy(deep=True)
    row_count = len(source)
    row_errors: list[list[str]] = [[] for _ in range(row_count)]
    parsed_dates: list[pd.Timestamp | None] = [None] * row_count
    numeric_demand = pd.to_numeric(source[demand_column], errors="coerce")

    for position, value in enumerate(source[date_column].tolist()):
        if pd.isna(value) or isinstance(value, (int, float, np.number)):
            row_errors[position].append("date is missing or invalid")
            continue
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            row_errors[position].append("date is missing or invalid")
            continue
        if pd.isna(timestamp):
            row_errors[position].append("date is missing or invalid")
            continue
        if timestamp.tzinfo is not None:
            row_errors[position].append("timezone-aware dates are not supported")
            continue
        parsed_dates[position] = timestamp

    parsed_date_series = pd.Series(parsed_dates, dtype="datetime64[ns]")
    aligned_dates = parsed_date_series.copy()
    valid_date_mask = parsed_date_series.notna()

    if frequency == "daily":
        aligned_dates = parsed_date_series.dt.normalize()
        misaligned = valid_date_mask & parsed_date_series.ne(aligned_dates)
    elif frequency == "weekly":
        aligned_dates = (
            parsed_date_series.dt.to_period("W-SUN").dt.start_time
        )
        misaligned = valid_date_mask & parsed_date_series.ne(aligned_dates)
    else:
        aligned_dates = parsed_date_series.dt.to_period("M").dt.start_time
        misaligned = valid_date_mask & parsed_date_series.ne(aligned_dates)

    for position in np.flatnonzero(misaligned.to_numpy()):
        row_errors[position].append(
            f"date is not aligned to the {frequency} period start"
        )

    demand_array = numeric_demand.to_numpy(dtype=float, na_value=np.nan)
    for position, value in enumerate(source[demand_column].tolist()):
        if isinstance(value, (bool, np.bool_)):
            row_errors[position].append("demand must be numeric, not boolean")
            continue
        numeric_value = demand_array[position]
        if pd.isna(value) or pd.isna(numeric_value):
            row_errors[position].append("demand is missing or non-numeric")
        elif not np.isfinite(numeric_value):
            row_errors[position].append("demand must be finite")
        elif numeric_value < 0:
            row_errors[position].append("demand cannot be negative")

    if product_column is not None:
        for position, value in enumerate(source[product_column].tolist()):
            if pd.isna(value) or (
                isinstance(value, str) and not value.strip()
            ):
                row_errors[position].append("product identifier is missing")
            elif isinstance(value, (bool, np.bool_)):
                row_errors[position].append(
                    "product identifier cannot be boolean"
                )

    duplicate_rows = 0
    key_columns = ["period"]
    keys = pd.DataFrame({"period": aligned_dates})
    if product_column is not None:
        keys["product_id"] = source[product_column].to_numpy(copy=True)
        key_columns.append("product_id")
    candidate_keys = keys.loc[
        valid_date_mask.to_numpy()
        & np.isfinite(demand_array)
        & (demand_array >= 0)
    ]
    duplicate_mask = candidate_keys.duplicated(key_columns, keep="first")
    duplicate_positions = candidate_keys.index[duplicate_mask].to_numpy()
    duplicate_rows = len(duplicate_positions)
    for position in duplicate_positions:
        row_errors[position].append("duplicate date/product period")

    invalid_mask = np.fromiter(
        (bool(errors) for errors in row_errors),
        dtype=bool,
        count=row_count,
    )
    valid_mask = ~invalid_mask
    warnings: list[str] = []
    valid_dates = parsed_date_series.loc[valid_mask]
    date_range = (
        (valid_dates.min(), valid_dates.max())
        if not valid_dates.empty
        else None
    )
    if valid_dates.size > 1 and not valid_dates.is_monotonic_increasing:
        warnings.append(
            "Input dates were unsorted; normalized output is sorted by period."
        )
    valid_products = (
        source.loc[valid_mask, product_column].nunique()
        if product_column is not None
        else 1
    )
    quality_report = DemandDataQualityReport(
        input_rows=row_count,
        valid_rows=int(valid_mask.sum()),
        invalid_rows=int(invalid_mask.sum()),
        duplicate_rows=duplicate_rows,
        date_range=date_range,
        frequency=frequency,
        number_of_products=int(valid_products),
        missing_periods=0,
        validation_warnings=tuple(warnings),
    )

    if invalid_mask.any():
        invalid_rows = source.iloc[np.flatnonzero(invalid_mask)].copy()
        invalid_rows.insert(
            0,
            "_input_row",
            np.flatnonzero(invalid_mask) + 1,
        )
        invalid_rows["_validation_errors"] = [
            "; ".join(row_errors[position])
            for position in np.flatnonzero(invalid_mask)
        ]
        raise DemandDataValidationError(
            "Demand dataset contains invalid rows; inspect invalid_rows and "
            "quality_report.",
            quality_report=quality_report,
            invalid_rows=invalid_rows,
        )

    if date_range is None:
        raise ValueError("Demand dataset contains no valid dates.")

    normalized_start = (
        _normalize_period_bound(expected_start, frequency)
        if expected_start is not None
        else date_range[0]
    )
    normalized_end = (
        _normalize_period_bound(expected_end, frequency)
        if expected_end is not None
        else date_range[1]
    )
    if normalized_start > normalized_end:
        raise ValueError("Expected coverage start must not be after expected end.")
    if date_range[0] < normalized_start or date_range[1] > normalized_end:
        raise ValueError(
            "Observed dates fall outside the normalized expected coverage bounds."
        )
    input_data = pd.DataFrame(
        {
            "period": aligned_dates,
            "demand_quantity": demand_array,
        }
    )
    if product_column is not None:
        input_data["product_id"] = source[product_column].to_numpy(copy=True)

    products = (
        input_data["product_id"].drop_duplicates().tolist()
        if product_column is not None
        else [None]
    )
    normalized_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    missing_periods = 0
    missing_before = 0
    missing_internal = 0
    missing_after = 0
    for product in products:
        if product_column is None:
            observed = input_data[["period", "demand_quantity"]]
        else:
            observed = input_data.loc[
                input_data["product_id"].eq(product),
                ["period", "demand_quantity"],
            ]
        first_observed = pd.Timestamp(observed["period"].min())
        last_observed = pd.Timestamp(observed["period"].max())
        use_full_coverage = (
            product_column is None
            or products_expected_throughout_coverage
        )
        grid_start = normalized_start if use_full_coverage else first_observed
        grid_end = normalized_end if use_full_coverage else last_observed
        period_grid = pd.date_range(
            grid_start,
            grid_end,
            freq=_FREQUENCY_OFFSETS[frequency],
        )
        periods = pd.DataFrame({"period": period_grid})
        part = periods.merge(observed, on="period", how="left", validate="one_to_one")
        if product_column is not None:
            part["product_id"] = product
        absent = part["demand_quantity"].isna()
        before = int(
            (absent & part["period"].lt(first_observed)).sum()
        )
        after = int(
            (absent & part["period"].gt(last_observed)).sum()
        )
        internal = int(
            (
                absent
                & part["period"].gt(first_observed)
                & part["period"].lt(last_observed)
            ).sum()
        )
        missing_before += before
        missing_internal += internal
        missing_after += after
        missing_periods += int(absent.sum())
        part["observation_status"] = np.where(
            absent,
            "unknown_unobserved_period",
            np.where(
                part["demand_quantity"].eq(0),
                "observed_zero_demand",
                "observed_demand",
            ),
        )
        if complete_source_coverage and confirm_missing_periods_are_zero:
            part.loc[absent, "demand_quantity"] = 0.0
            part.loc[absent, "observation_status"] = "confirmed_zero_demand"
        normalized_parts.append(part)
        excluded_before = (
            len(
                pd.date_range(
                    normalized_start,
                    first_observed,
                    freq=_FREQUENCY_OFFSETS[frequency],
                )
            )
            - 1
            if product_column is not None and not use_full_coverage
            else 0
        )
        excluded_after = (
            len(
                pd.date_range(
                    last_observed,
                    normalized_end,
                    freq=_FREQUENCY_OFFSETS[frequency],
                )
            )
            - 1
            if product_column is not None and not use_full_coverage
            else 0
        )
        coverage_row: dict[str, object] = {
            "expected_start_period": normalized_start,
            "expected_end_period": normalized_end,
            "first_supplied_period": first_observed,
            "last_supplied_period": last_observed,
            "missing_before_first_observation": before,
            "missing_internal_periods": internal,
            "missing_after_last_observation": after,
            "excluded_before_first_observation_periods": excluded_before,
            "excluded_after_last_observation_periods": excluded_after,
        }
        if product_column is not None:
            coverage_row = {"product_id": product, **coverage_row}
        coverage_rows.append(coverage_row)

    normalized = pd.concat(normalized_parts, ignore_index=True)
    output_columns = ["period"]
    if product_column is not None:
        output_columns.append("product_id")
    output_columns.extend(["demand_quantity", "observation_status"])
    normalized = normalized[output_columns].sort_values(
        ["period"],
        kind="stable",
        ignore_index=True,
    )
    if missing_periods:
        if complete_source_coverage and confirm_missing_periods_are_zero:
            warnings.append(
                f"{missing_periods} absent periods were marked as confirmed "
                "zero demand based on explicit coverage confirmation."
            )
        else:
            warnings.append(
                f"{missing_periods} absent periods remain unknown; no demand "
                "values were imputed."
            )

    quality_report = DemandDataQualityReport(
        input_rows=quality_report.input_rows,
        valid_rows=quality_report.valid_rows,
        invalid_rows=quality_report.invalid_rows,
        duplicate_rows=quality_report.duplicate_rows,
        date_range=quality_report.date_range,
        frequency=quality_report.frequency,
        number_of_products=quality_report.number_of_products,
        missing_periods=missing_periods,
        validation_warnings=tuple(warnings),
        expected_coverage_start=normalized_start,
        expected_coverage_end=normalized_end,
        missing_before_first_observation=missing_before,
        missing_internal_periods=missing_internal,
        missing_after_last_observation=missing_after,
    )
    return PreparedDemandData(
        data=normalized,
        quality_report=quality_report,
        coverage_by_product=pd.DataFrame(coverage_rows),
        products_expected_throughout_coverage=(
            products_expected_throughout_coverage
        ),
    )


def _normalize_period_bound(value: object, frequency: Frequency) -> pd.Timestamp:
    if isinstance(value, (bool, int, float, np.number)):
        raise ValueError("Expected coverage bounds must be valid dates.")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected coverage bounds must be valid dates.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        raise ValueError(
            "Expected coverage bounds must be valid timezone-naive dates."
        )
    if frequency == "daily":
        return timestamp.normalize()
    if frequency == "weekly":
        return timestamp.to_period("W-SUN").start_time
    return timestamp.to_period("M").start_time
