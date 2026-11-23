"""Integration helpers for the isolated user-upload forecasting page."""

import csv
import hashlib
import io
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.data.user_demand import (
    DemandDataValidationError,
    Frequency,
    prepare_user_demand_data,
)
from src.forecasting.user_demand_backtesting import (
    RollingOriginBacktest,
    RollingOriginConfig,
    rolling_origin_backtest,
)
from src.forecasting.user_demand_diagnostics import (
    DemandPatternDiagnostics,
    diagnose_demand_patterns,
)
from src.forecasting.user_demand_models import (
    CrostonDemandModel,
    DemandForecast,
    ForecastModel,
    MovingAverageDemandModel,
    NaiveDemandModel,
    SeasonalNaiveDemandModel,
    SimpleExponentialSmoothingDemandModel,
    forecast_demand_models,
)
from src.preprocessing.user_demand_preparation import (
    ForecastReadyDemandData,
    aggregate_transaction_demand,
    prepare_forecast_demand,
)


MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
_SESSION_PREFIX = "user_demand_forecasting_"
_SESSION_UPLOAD_HASH = f"{_SESSION_PREFIX}upload_hash"
_SESSION_FILENAME = f"{_SESSION_PREFIX}filename"
_SESSION_MODEL_SELECTIONS = f"{_SESSION_PREFIX}model_selections"
_SESSION_MODEL_SELECTION_CONTEXT = f"{_SESSION_PREFIX}model_selection_context"
USE_RECOMMENDATION = "__use_recommendation__"
_SESSION_SOURCE = f"{_SESSION_PREFIX}source"
_SESSION_PREPARED = f"{_SESSION_PREFIX}prepared"
_SESSION_DIAGNOSTICS = f"{_SESSION_PREFIX}diagnostics"
_SESSION_PREPARATION_SIGNATURE = f"{_SESSION_PREFIX}preparation_signature"
_SESSION_BACKTEST = f"{_SESSION_PREFIX}backtest"
_SESSION_BACKTEST_SIGNATURE = f"{_SESSION_PREFIX}backtest_signature"
_SESSION_FORECASTS = f"{_SESSION_PREFIX}forecasts"
_SESSION_FORECAST_WARNINGS = f"{_SESSION_PREFIX}forecast_warnings"

_PREPARATION_KEYS = (
    _SESSION_PREPARED,
    _SESSION_DIAGNOSTICS,
    _SESSION_PREPARATION_SIGNATURE,
    f"{_SESSION_PREFIX}current_preparation_signature",
    _SESSION_BACKTEST,
    _SESSION_BACKTEST_SIGNATURE,
    f"{_SESSION_PREFIX}current_backtest_signature",
    _SESSION_FORECASTS,
    _SESSION_FORECAST_WARNINGS,
    f"{_SESSION_PREFIX}forecast_signature",
    f"{_SESSION_PREFIX}current_forecast_signature",
    _SESSION_MODEL_SELECTIONS,
    _SESSION_MODEL_SELECTION_CONTEXT,
)


class UserDemandUploadError(ValueError):
    """Raised when an uploaded file cannot be safely read as a demand CSV."""


@dataclass(frozen=True)
class PreparedUserUpload:
    """Prepared demand and diagnostics from one uploaded dataset."""

    demand: ForecastReadyDemandData
    diagnostics: DemandPatternDiagnostics


@dataclass(frozen=True)
class UserOperationalData:
    """Current upload bundle exposed read-only to operational dashboard pages."""

    filename: str
    upload_hash: str
    prepared: PreparedUserUpload
    backtest: RollingOriginBacktest
    forecasts: tuple[DemandForecast, ...]
    forecast_warnings: pd.DataFrame
    preparation_signature: tuple[object, ...]
    backtest_signature: tuple[object, ...]


class UserDemandForecastingError(ValueError):
    """Raised when the upload workflow configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        invalid_rows: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(message)
        self.invalid_rows = invalid_rows


@dataclass(frozen=True)
class UserDemandColumnSuggestions:
    """Value-checked header suggestions for common demand CSV columns."""

    date_column: str | None
    demand_column: str | None
    product_column: str | None
    category_column: str | None


@dataclass(frozen=True)
class UserDemandFrequencySuggestion:
    """A date-based frequency suggestion and its rationale."""

    frequency: Frequency | None
    explanation: str


def suggest_user_demand_columns(
    dataframe: pd.DataFrame,
) -> UserDemandColumnSuggestions:
    """Suggest common column roles only when all values pass basic checks."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if not dataframe.columns.is_unique:
        raise ValueError("Demand dataset contains duplicate column names.")

    date_column = _suggest_column(
        dataframe,
        (
            "transaction_date",
            "order_date",
            "sales_date",
            "date_of_sale",
            "invoice_date",
            "week_start",
            "date",
            "datetime",
            "timestamp",
            "period",
            "week",
            "month",
        ),
        _valid_date_values,
    )
    demand_column = _suggest_column(
        dataframe,
        (
            "sales_quantity",
            "sales_qty",
            "quantity_sold",
            "qty_sold",
            "units_sold",
            "demand_quantity",
            "demand_qty",
            "quantity",
            "qty",
            "units",
            "demand",
            "volume_sold",
        ),
        _valid_demand_values,
    )
    product_column = _suggest_column(
        dataframe,
        (
            "product_id",
            "sku",
            "item_code",
            "stock_code",
            "material_code",
            "product_name",
            "item_name",
            "product",
            "item",
        ),
        _valid_identifier_values,
    )
    category_column = _suggest_column(
        dataframe,
        (
            "product_category",
            "category_name",
            "business_unit",
            "product_group",
            "department",
            "category",
            "group",
            "brand",
            "family",
            "segment",
        ),
        _valid_identifier_values,
    )
    return UserDemandColumnSuggestions(
        date_column=date_column,
        demand_column=demand_column,
        product_column=product_column,
        category_column=category_column,
    )


def suggest_user_demand_frequency(
    dataframe: pd.DataFrame,
    date_column: str | None,
) -> UserDemandFrequencySuggestion:
    """Suggest a period size from a validated date column's cadence."""
    if date_column is None or date_column not in dataframe.columns:
        return UserDemandFrequencySuggestion(
            None,
            "Select a date column with valid values to suggest a frequency.",
        )
    dates = _parsed_suggestion_dates(dataframe[date_column])
    if dates is None:
        return UserDemandFrequencySuggestion(
            None,
            "The selected date column has missing or invalid values.",
        )

    unique_dates = pd.DatetimeIndex(dates.dt.normalize().drop_duplicates()).sort_values()
    normalized_name = _normalize_column_name(date_column)
    one_per_month = (
        len(unique_dates.to_period("M").unique()) == len(unique_dates)
    )
    monthly_boundaries = (
        len(unique_dates) >= 2
        and one_per_month
        and (
            (unique_dates.day == unique_dates.day[0]).all()
            or (unique_dates + pd.offsets.MonthEnd(0) == unique_dates).all()
        )
    )
    if "month" in normalized_name.split("_") or monthly_boundaries:
        return UserDemandFrequencySuggestion(
            "monthly",
            "Dates show one consistent monthly observation day or month-end, "
            "or the column is named as a month.",
        )
    if "week" in normalized_name.split("_"):
        return UserDemandFrequencySuggestion(
            "weekly",
            "The selected date column is named as a week or week start.",
        )
    if len(unique_dates) >= 2:
        day_gaps = np.diff(unique_dates.values).astype("timedelta64[D]").astype(int)
        same_weekday = len(set(unique_dates.dayofweek)) == 1
        if same_weekday and np.all(day_gaps % 7 == 0):
            return UserDemandFrequencySuggestion(
                "weekly",
                "Observed dates share a weekday and their gaps are whole weeks.",
            )
    return UserDemandFrequencySuggestion(
        "daily",
        "Dates vary within calendar periods or do not show a regular weekly/monthly cadence.",
    )


def suggest_transaction_aggregation(
    dataframe: pd.DataFrame,
    *,
    date_column: str | None,
    series_column: str | None,
    frequency: Frequency | None,
) -> bool:
    """Suggest summing rows when dates are unaligned or repeat within a series."""
    if (
        date_column is None
        or date_column not in dataframe.columns
        or frequency is None
    ):
        return False
    dates = _parsed_suggestion_dates(dataframe[date_column])
    if dates is None:
        return False
    if frequency == "daily":
        periods = dates.dt.normalize()
    elif frequency == "weekly":
        periods = dates.dt.to_period("W-SUN").dt.start_time
    else:
        periods = dates.dt.to_period("M").dt.start_time

    keys = pd.DataFrame({"period": periods})
    if series_column is not None and series_column in dataframe.columns:
        keys["series"] = dataframe[series_column].to_numpy(copy=True)
    unaligned = dates.ne(periods).any()
    repeated_periods = keys.duplicated(keep=False).any()
    return bool(unaligned or repeated_periods)


def _normalize_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _suggest_column(
    dataframe: pd.DataFrame,
    aliases: tuple[str, ...],
    value_check,
) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    for position, column in enumerate(dataframe.columns):
        normalized = _normalize_column_name(column)
        score = _column_name_score(normalized, aliases)
        if score is None or not value_check(dataframe[column]):
            continue
        candidates.append((score, -position, column))
    return max(candidates)[2] if candidates else None


def _column_name_score(
    normalized: str,
    aliases: tuple[str, ...],
) -> int | None:
    tokens = set(normalized.split("_"))
    for alias_position, alias in enumerate(aliases):
        if alias in {"product", "item"} and {
            "category",
            "group",
            "type",
        }.intersection(tokens):
            continue
        alias_tokens = set(alias.split("_"))
        if normalized == alias:
            return 1000 - alias_position
        if alias_tokens.issubset(tokens):
            return 100 - alias_position
        if len(alias_tokens) == 1 and alias in tokens:
            return 50 - alias_position
    return None


def _valid_date_values(values: pd.Series) -> bool:
    return _parsed_suggestion_dates(values) is not None


def _parsed_suggestion_dates(
    values: pd.Series,
) -> pd.Series | None:
    if values.empty or values.astype(str).str.strip().eq("").any():
        return None
    if values.map(
        lambda value: isinstance(value, (bool, np.bool_, int, float, np.number))
    ).any():
        return None
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.isna().any() or not pd.api.types.is_datetime64_any_dtype(
        parsed.dtype
    ):
        return None
    if parsed.dt.tz is not None:
        return None
    return parsed


def _valid_demand_values(values: pd.Series) -> bool:
    if values.empty or values.astype(str).str.strip().eq("").any():
        return False
    if values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        return False
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    return bool(np.isfinite(numeric).all() and (numeric >= 0).all())


def _valid_identifier_values(values: pd.Series) -> bool:
    return bool(
        not values.empty
        and values.astype(str).str.strip().ne("").all()
    )


def validate_user_demand_upload_size(size: int) -> None:
    """Reject uploads above the configured per-file size limit."""
    if size > MAX_UPLOAD_BYTES:
        raise UserDemandUploadError(
            f"The CSV exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit."
        )


def parse_user_demand_csv(filename: str, content: bytes) -> pd.DataFrame:
    """Read a UTF-8 CSV without changing or dropping its contents."""
    if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
        raise UserDemandUploadError("Upload a file with the .csv extension.")
    if not isinstance(content, bytes):
        raise TypeError("CSV content must be bytes.")
    if not content:
        raise UserDemandUploadError("The uploaded CSV file is empty.")
    validate_user_demand_upload_size(len(content))

    try:
        with io.TextIOWrapper(
            io.BytesIO(content),
            encoding="utf-8-sig",
            newline="",
        ) as text:
            header = next(csv.reader(text, strict=True), None)
    except UnicodeDecodeError as exc:
        raise UserDemandUploadError(
            "The CSV could not be decoded as UTF-8. Save or export it as "
            "UTF-8 CSV and upload it again."
        ) from exc
    except csv.Error as exc:
        raise UserDemandUploadError(
            f"The CSV header is malformed: {exc}"
        ) from exc
    if not header or not any(name.strip() for name in header):
        raise UserDemandUploadError(
            "The CSV must include a header row and at least one named column."
        )
    if any(not name.strip() for name in header):
        raise UserDemandUploadError(
            "The CSV header contains a blank column name. Rename every column."
        )
    if len(set(header)) != len(header):
        raise UserDemandUploadError(
            "The CSV header contains duplicate column names. Rename them before upload."
        )

    try:
        dataframe = pd.read_csv(
            io.BytesIO(content),
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            on_bad_lines="error",
        )
    except UnicodeDecodeError as exc:
        raise UserDemandUploadError(
            "The CSV could not be decoded as UTF-8. Save or export it as "
            "UTF-8 CSV and upload it again."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise UserDemandUploadError(
            "The CSV has a header but contains no data rows."
        ) from exc
    except (pd.errors.ParserError, csv.Error, ValueError) as exc:
        raise UserDemandUploadError(
            f"The CSV rows could not be parsed: {exc}"
        ) from exc
    if dataframe.empty:
        raise UserDemandUploadError(
            "The CSV has a header but contains no data rows."
        )
    if dataframe.shape[1] != len(header):
        raise UserDemandUploadError(
            "CSV rows have a different number of fields than the header. "
            "Check the delimiter and row formatting."
        )
    if not dataframe.columns.is_unique:
        raise UserDemandUploadError(
            "The CSV contains duplicate column names. Rename them before upload."
        )
    return dataframe


def store_uploaded_csv(
    session: MutableMapping[str, object],
    filename: str,
    content: bytes,
) -> pd.DataFrame:
    """Store an upload only in the caller's Streamlit session state."""
    if not isinstance(filename, str):
        raise UserDemandUploadError("Upload a file with the .csv extension.")
    if not isinstance(content, bytes):
        raise TypeError("CSV content must be bytes.")
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    upload_hash = digest.hexdigest()
    existing = session.get(_SESSION_SOURCE)
    if (
        session.get(_SESSION_UPLOAD_HASH) == upload_hash
        and isinstance(existing, pd.DataFrame)
    ):
        return existing.copy(deep=True)

    _clear_keys(session, (_SESSION_SOURCE, *_PREPARATION_KEYS))
    session.pop(_SESSION_UPLOAD_HASH, None)
    session.pop(_SESSION_FILENAME, None)
    dataframe = parse_user_demand_csv(filename, content)
    session[_SESSION_SOURCE] = dataframe
    session[_SESSION_UPLOAD_HASH] = upload_hash
    session[_SESSION_FILENAME] = filename
    return dataframe.copy(deep=True)


def clear_user_demand_session(session: MutableMapping[str, object]) -> None:
    """Clear this feature's data and derived results from a session mapping."""
    _clear_keys(
        session,
        tuple(key for key in session if key.startswith(_SESSION_PREFIX)),
    )


def user_upload_hash(session: Mapping[str, object]) -> str | None:
    """Return the digest for this session's currently stored upload."""
    value = session.get(_SESSION_UPLOAD_HASH)
    return value if isinstance(value, str) else None


def user_upload_name(session: Mapping[str, object]) -> str | None:
    """Return the filename associated with the current upload."""
    value = session.get(_SESSION_FILENAME)
    return value if isinstance(value, str) else None


def user_operational_data(
    session: MutableMapping[str, object],
) -> tuple[UserOperationalData | None, str | None]:
    """Return only the prepared and forecasted bundle matching current signatures."""
    filename = user_upload_name(session)
    upload_hash = user_upload_hash(session)
    prepared = session.get(_SESSION_PREPARED)
    preparation_signature = session.get(_SESSION_PREPARATION_SIGNATURE)
    current_preparation_signature = session.get(
        f"{_SESSION_PREFIX}current_preparation_signature"
    )
    if filename is None or upload_hash is None:
        return None, "No user CSV is available in this browser session."
    if not isinstance(prepared, PreparedUserUpload) or not isinstance(
        preparation_signature, tuple
    ):
        return None, "Upload and prepare a dataset on User Demand Forecasting first."
    if (
        not preparation_signature
        or preparation_signature[0] != upload_hash
        or current_preparation_signature != preparation_signature
    ):
        _clear_keys(
            session,
            (
                _SESSION_PREPARED,
                _SESSION_DIAGNOSTICS,
                _SESSION_PREPARATION_SIGNATURE,
                _SESSION_BACKTEST,
                _SESSION_BACKTEST_SIGNATURE,
                f"{_SESSION_PREFIX}current_backtest_signature",
                _SESSION_FORECASTS,
                _SESSION_FORECAST_WARNINGS,
                f"{_SESSION_PREFIX}forecast_signature",
                f"{_SESSION_PREFIX}current_forecast_signature",
            ),
        )
        return None, (
            "The upload mapping or preparation options changed. Re-prepare, "
            "compare models, and generate forecasts before using this page."
        )

    backtest = session.get(_SESSION_BACKTEST)
    backtest_signature = session.get(_SESSION_BACKTEST_SIGNATURE)
    if not isinstance(backtest, RollingOriginBacktest) or not isinstance(
        backtest_signature, tuple
    ):
        return None, (
            "Run model comparison and generate forecasts on User Demand "
            "Forecasting before using this page."
        )
    if (
        not backtest_signature
        or backtest_signature[0] != preparation_signature
        or session.get(f"{_SESSION_PREFIX}current_backtest_signature")
        != backtest_signature
    ):
        _clear_keys(
            session,
            (
                _SESSION_BACKTEST,
                _SESSION_BACKTEST_SIGNATURE,
                _SESSION_FORECASTS,
                _SESSION_FORECAST_WARNINGS,
                f"{_SESSION_PREFIX}forecast_signature",
                f"{_SESSION_PREFIX}current_forecast_signature",
            ),
        )
        return None, (
            "The forecast comparison settings changed. Run model comparison "
            "and generate forecasts again."
        )

    forecasts = session.get(_SESSION_FORECASTS)
    forecast_warnings = session.get(_SESSION_FORECAST_WARNINGS)
    forecast_signature = session.get(f"{_SESSION_PREFIX}forecast_signature")
    current_forecast_signature = session.get(
        f"{_SESSION_PREFIX}current_forecast_signature"
    )
    if not isinstance(forecasts, tuple) or not isinstance(
        forecast_warnings, pd.DataFrame
    ):
        return None, (
            "Generate future forecasts on User Demand Forecasting before "
            "using this page."
        )
    if (
        not isinstance(forecast_signature, tuple)
        or len(forecast_signature) != 2
        or forecast_signature[0] != backtest_signature
        or current_forecast_signature != forecast_signature
    ):
        _clear_keys(
            session,
            (
                _SESSION_FORECASTS,
                _SESSION_FORECAST_WARNINGS,
                f"{_SESSION_PREFIX}forecast_signature",
            ),
        )
        return None, (
            "The selected forecasting model changed. Generate forecasts again "
            "before using this page."
        )
    if (
        any(not isinstance(forecast, DemandForecast) for forecast in forecasts)
        or backtest.frequency != prepared.demand.frequency
        or backtest.horizon < 1
    ):
        _clear_keys(
            session,
            (
                _SESSION_FORECASTS,
                _SESSION_FORECAST_WARNINGS,
                f"{_SESSION_PREFIX}forecast_signature",
            ),
        )
        return None, "The saved forecast does not match the prepared demand frequency."

    return (
        UserOperationalData(
            filename=filename,
            upload_hash=upload_hash,
            prepared=prepared,
            backtest=backtest,
            forecasts=forecasts,
            forecast_warnings=forecast_warnings.copy(deep=True),
            preparation_signature=preparation_signature,
            backtest_signature=backtest_signature,
        ),
        None,
    )


def product_model_selections(
    session: MutableMapping[str, object],
    *,
    context: str,
) -> dict[str, str | None]:
    """Return stored model choices, clearing choices from another data context."""
    if session.get(_SESSION_MODEL_SELECTION_CONTEXT) != context:
        session[_SESSION_MODEL_SELECTION_CONTEXT] = context
        session[_SESSION_MODEL_SELECTIONS] = {}
    choices = session.get(_SESSION_MODEL_SELECTIONS)
    if not isinstance(choices, dict):
        choices = {}
        session[_SESSION_MODEL_SELECTIONS] = choices
    return choices


def product_selection_key(product_id: object | None) -> str:
    """Build a stable key that distinguishes numeric and textual product IDs."""
    return f"{type(product_id).__module__}.{type(product_id).__qualname__}:{product_id!r}"


def save_product_model_selection(
    session: MutableMapping[str, object],
    *,
    context: str,
    product_id: object | None,
    model_name: str | None,
) -> None:
    """Persist one product's validated model choice in this session context."""
    choices = product_model_selections(session, context=context)
    choices[product_selection_key(product_id)] = model_name


def prepare_user_upload(
    dataframe: pd.DataFrame,
    *,
    date_column: str,
    demand_column: str,
    product_column: str | None,
    frequency: Frequency,
    aggregate_transactions: bool = False,
    complete_source_coverage: bool = False,
    confirm_missing_periods_are_zero: bool = False,
    expected_start: object | None = None,
    expected_end: object | None = None,
    products_expected_throughout_coverage: bool = False,
) -> PreparedUserUpload:
    """Validate, normalize, and diagnose user data through the generic APIs."""
    try:
        if aggregate_transactions:
            demand = aggregate_transaction_demand(
                dataframe,
                date_column=date_column,
                demand_column=demand_column,
                product_column=product_column,
                frequency=frequency,
                aggregation="sum_transactions",
                complete_source_coverage=complete_source_coverage,
                confirm_missing_periods_are_zero=(
                    confirm_missing_periods_are_zero
                ),
                expected_start=expected_start,
                expected_end=expected_end,
                products_expected_throughout_coverage=(
                    products_expected_throughout_coverage
                ),
            )
        else:
            prepared = prepare_user_demand_data(
                dataframe,
                date_column=date_column,
                demand_column=demand_column,
                product_column=product_column,
                frequency=frequency,
                complete_source_coverage=complete_source_coverage,
                confirm_missing_periods_are_zero=(
                    confirm_missing_periods_are_zero
                ),
                expected_start=expected_start,
                expected_end=expected_end,
                products_expected_throughout_coverage=(
                    products_expected_throughout_coverage
                ),
            )
            demand = prepare_forecast_demand(prepared)
        diagnostics = diagnose_demand_patterns(demand)
    except DemandDataValidationError as exc:
        raise UserDemandForecastingError(
            str(exc),
            invalid_rows=exc.invalid_rows.copy(deep=True),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise UserDemandForecastingError(str(exc)) from exc
    return PreparedUserUpload(demand=demand, diagnostics=diagnostics)


def available_forecasting_models(frequency: Frequency) -> tuple[ForecastModel, ...]:
    """Return the supported candidate set, including conditionally eligible models."""
    seasonal_period = {"daily": 7, "weekly": 52, "monthly": 12}[frequency]
    return (
        NaiveDemandModel(),
        MovingAverageDemandModel(window=3),
        SeasonalNaiveDemandModel(
            seasonal_period=seasonal_period,
            minimum_cycles=3,
        ),
        SimpleExponentialSmoothingDemandModel(alpha=0.3),
        CrostonDemandModel(alpha=0.1),
    )


def compare_user_demand_models(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    *,
    horizon: int,
    min_train_periods: int = 8,
    min_successful_origins: int = 2,
    min_comparable_origins: int = 2,
    min_success_fraction: float = 0.5,
    selection_metric: Literal["mae", "rmse", "wape"] = "mae",
) -> RollingOriginBacktest:
    """Run rolling-origin comparison for the generic candidate set."""
    config = RollingOriginConfig(
        horizon=horizon,
        min_train_periods=min_train_periods,
        min_successful_origins=min_successful_origins,
        min_comparable_origins=min_comparable_origins,
        min_success_fraction=min_success_fraction,
        selection_metric=selection_metric,
    )
    return rolling_origin_backtest(
        demand,
        diagnostics,
        available_forecasting_models(demand.frequency),
        config=config,
    )


def generate_user_forecasts(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    backtest: RollingOriginBacktest,
    *,
    horizon: int,
    selected_models: Mapping[object | None, str | None],
) -> tuple[tuple[DemandForecast, ...], pd.DataFrame]:
    """Generate only the explicitly selected, backtest-eligible model per series."""
    if backtest.frequency != demand.frequency or backtest.horizon != horizon:
        raise UserDemandForecastingError(
            "Backtest results do not match the prepared demand frequency "
            "and forecast horizon."
        )
    candidates = {
        model.name: model for model in available_forecasting_models(demand.frequency)
    }
    valid_names = {model.name for model in candidates.values()}
    for product_id, model_name in selected_models.items():
        if model_name is not None and model_name not in valid_names:
            raise UserDemandForecastingError(
                f"Unknown forecasting model {model_name!r} for product "
                f"{product_id!r}."
            )
        matching = next(
            (
                selection
                for selection in backtest.selections
                if selection.product_id == product_id
            ),
            None,
        )
        if model_name is not None and (
            matching is None
            or model_name not in dict(matching.candidate_scores)
        ):
            raise UserDemandForecastingError(
                f"{model_name} was not validated on comparable rolling-origin "
                f"windows for product {product_id!r}."
            )

    selected_names = list(
        dict.fromkeys(name for name in selected_models.values() if name is not None)
    )
    if not selected_names:
        warnings = pd.DataFrame(
            [
                {
                    "product_id": product_id,
                    "model_name": None,
                    "warning": "No model met the configured rolling-origin "
                    "evaluation requirements.",
                }
                for product_id in selected_models
            ],
            columns=["product_id", "model_name", "warning"],
        )
        return (), warnings

    forecasts = forecast_demand_models(
        demand,
        diagnostics,
        [candidates[name] for name in selected_names],
        horizon=horizon,
    )
    chosen: list[DemandForecast] = []
    warnings: list[dict[str, object]] = []
    for product_id, model_name in selected_models.items():
        if model_name is None:
            warnings.append(
                {
                    "product_id": product_id,
                    "model_name": None,
                    "warning": "No model met the configured rolling-origin "
                    "evaluation requirements.",
                }
            )
            continue
        result = next(
            (
                forecast
                for forecast in forecasts
                if forecast.product_id == product_id
                and forecast.model_name == model_name
            ),
            None,
        )
        if result is None:
            warnings.append(
                {
                    "product_id": product_id,
                    "model_name": model_name,
                    "warning": "No forecast result was returned for this series.",
                }
            )
        elif not result.model_eligible:
            warnings.append(
                {
                    "product_id": product_id,
                    "model_name": model_name,
                    "warning": "; ".join(result.warnings),
                }
            )
        else:
            chosen.extend(
                forecast
                for forecast in forecasts
                if forecast.product_id == product_id
                and forecast.model_name == model_name
            )
    return tuple(chosen), pd.DataFrame(
        warnings,
        columns=["product_id", "model_name", "warning"],
    )


def build_forecast_export(
    forecasts: tuple[DemandForecast, ...],
    backtest: RollingOriginBacktest,
) -> pd.DataFrame:
    """Build the forecast CSV schema in memory."""
    selection_map = {
        selection.product_id: selection
        for selection in backtest.selections
    }
    rows = []
    for forecast in forecasts:
        selection = selection_map.get(forecast.product_id)
        rows.append(
            {
                "product_id": forecast.product_id,
                "period": forecast.forecast_period,
                "predicted_demand": forecast.predicted_demand,
                "model_name": forecast.model_name,
                "forecast_horizon": forecast.forecast_horizon,
                "frequency": backtest.frequency,
                "demand_pattern": "; ".join(forecast.demand_pattern),
                "selection_metric": selection.metric if selection else None,
                "selection_rationale": (
                    selection.rationale if selection else None
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "product_id",
            "period",
            "predicted_demand",
            "model_name",
            "forecast_horizon",
            "frequency",
            "demand_pattern",
            "selection_metric",
            "selection_rationale",
        ],
    )


def build_evaluation_export(backtest: RollingOriginBacktest) -> pd.DataFrame:
    """Build an evaluation CSV with model outcomes and skipped windows."""
    selection_map = {
        selection.product_id: selection
        for selection in backtest.selections
    }
    rows = []
    for summary in backtest.model_summaries:
        selection = selection_map.get(summary.product_id)
        metrics = summary.comparable_metrics
        rows.append(
            {
                "product_id": summary.product_id,
                "record_type": "model_summary",
                "model_name": summary.model_name,
                "selection_metric": selection.metric if selection else None,
                "selected_model": selection.selected_model if selection else None,
                "selection_rationale": (
                    selection.rationale if selection else None
                ),
                "successful_origins": summary.successful_origins,
                "failed_origins": summary.failed_origins,
                "eligible_origins": summary.eligible_origins,
                "comparable_origins": summary.comparable_origins,
                "evaluated_observations": (
                    metrics.evaluated_observations if metrics else 0
                ),
                "mae": metrics.mae if metrics else None,
                "rmse": metrics.rmse if metrics else None,
                "wape": metrics.wape if metrics else None,
                "actual_total": metrics.actual_total if metrics else None,
                "forecast_total": metrics.forecast_total if metrics else None,
                "failure_reasons": "; ".join(summary.failure_reasons),
                "origin_period": None,
                "evaluation_periods": None,
                "skipped_reason": None,
            }
        )
    for skipped in backtest.skipped_origins:
        rows.append(
            {
                "product_id": skipped.product_id,
                "record_type": "skipped_window",
                "model_name": None,
                "selection_metric": None,
                "selected_model": None,
                "selection_rationale": None,
                "successful_origins": 0,
                "failed_origins": 0,
                "eligible_origins": 0,
                "comparable_origins": 0,
                "evaluated_observations": 0,
                "mae": None,
                "rmse": None,
                "wape": None,
                "actual_total": None,
                "forecast_total": None,
                "failure_reasons": "",
                "origin_period": skipped.origin_period,
                "evaluation_periods": "; ".join(
                    period.isoformat() for period in skipped.evaluation_periods
                ),
                "skipped_reason": skipped.reason,
            }
        )
    return pd.DataFrame(rows)


def diagnostics_frame(diagnostics: DemandPatternDiagnostics) -> pd.DataFrame:
    """Return a display-ready one-row-per-series diagnostics table."""
    return pd.DataFrame(
        [
            {
                "product_id": item.product_id,
                "history_periods": item.history_length,
                "eligible_periods": item.eligible_periods,
                "coverage": item.coverage_ratio,
                "unknown_periods": item.unknown_periods,
                "confirmed_zero_periods": item.confirmed_zero_periods,
                "observed_zero_periods": item.observed_zero_periods,
                "zero_demand_pct": item.zero_demand_percentage,
                "mean_demand": item.mean_demand,
                "standard_deviation": item.standard_deviation,
                "coefficient_of_variation": item.coefficient_of_variation,
                "classification": "; ".join(item.classifications),
                "explanation": "; ".join(item.explanations),
                "warnings": "; ".join(item.warnings),
                "expected_start_period": item.expected_start_period,
                "expected_end_period": item.expected_end_period,
                "missing_before_first_observation": (
                    item.missing_before_first_observation
                ),
                "missing_internal_periods": item.missing_internal_periods,
                "missing_after_last_observation": (
                    item.missing_after_last_observation
                ),
                "excluded_before_first_observation_periods": (
                    item.excluded_before_first_observation_periods
                ),
                "excluded_after_last_observation_periods": (
                    item.excluded_after_last_observation_periods
                ),
            }
            for item in diagnostics.series
        ]
    )


def _clear_keys(
    mapping: MutableMapping[str, object],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        mapping.pop(key, None)
