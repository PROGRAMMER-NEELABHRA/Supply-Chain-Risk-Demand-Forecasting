"""Isolated forecasting workflow for user-uploaded demand data."""

import hashlib
from typing import cast

import pandas as pd
import streamlit as st

from dashboard.components.kpi_cards import page_header
from dashboard.operational_data import USER_UPLOAD_SOURCE, set_operational_source
from src.data.user_demand import Frequency
from dashboard.user_demand_forecasting import (
    MAX_UPLOAD_SIZE_MB,
    PreparedUserUpload,
    UserDemandForecastingError,
    UserDemandUploadError,
    build_evaluation_export,
    build_forecast_export,
    compare_user_demand_models,
    diagnostics_frame,
    generate_user_forecasts,
    prepare_user_upload,
    product_model_selections,
    product_selection_key,
    save_product_model_selection,
    suggest_transaction_aggregation,
    suggest_user_demand_columns,
    suggest_user_demand_frequency,
    store_uploaded_csv,
    USE_RECOMMENDATION,
    user_upload_hash,
    validate_user_demand_upload_size,
)


_MODEL_CHOICE_PREFIX = "user_demand_forecasting_model_choice_"
_CHOOSE_MODEL = "Choose a model"
_MAPPING_CONTEXT_KEY = "user_demand_forecasting_mapping_upload_hash"
_DATE_COLUMN_KEY = "user_demand_forecasting_date_column"
_DEMAND_COLUMN_KEY = "user_demand_forecasting_demand_column"
_SERIES_COLUMN_KEY = "user_demand_forecasting_series_column"
_FREQUENCY_KEY = "user_demand_forecasting_frequency"
_AGGREGATION_KEY = "user_demand_forecasting_aggregate_transactions"
_AGGREGATION_OVERRIDE_KEY = "user_demand_forecasting_aggregation_override"
_COMPLETE_COVERAGE_KEY = "user_demand_forecasting_complete_coverage"
_ZERO_MISSING_KEY = "user_demand_forecasting_confirm_missing_zero"
_EXPECTED_BOUNDS_KEY = "user_demand_forecasting_specify_expected_coverage"
_EXPECTED_START_KEY = "user_demand_forecasting_expected_start"
_EXPECTED_END_KEY = "user_demand_forecasting_expected_end"
_PRODUCTS_EXPECTED_KEY = "user_demand_forecasting_products_expected"
_MAPPING_WIDGET_KEYS = (
    _DATE_COLUMN_KEY,
    _DEMAND_COLUMN_KEY,
    _SERIES_COLUMN_KEY,
    _FREQUENCY_KEY,
    _AGGREGATION_KEY,
    _AGGREGATION_OVERRIDE_KEY,
    _COMPLETE_COVERAGE_KEY,
    _ZERO_MISSING_KEY,
    _EXPECTED_BOUNDS_KEY,
    _EXPECTED_START_KEY,
    _EXPECTED_END_KEY,
    _PRODUCTS_EXPECTED_KEY,
)


def _largest_unknown_run(frame: pd.DataFrame) -> int:
    groups = (
        frame.groupby("product_id", sort=False)
        if "product_id" in frame.columns
        else [(None, frame)]
    )
    longest = 0
    for _, group in groups:
        unknown = group["observation_status"].eq(
            "unknown_unobserved_period"
        )
        if unknown.any():
            run_ids = unknown.ne(unknown.shift()).cumsum()
            longest = max(
                longest,
                int(unknown.groupby(run_ids).sum().max()),
            )
    return longest


page_header(
    title="User demand forecasting",
    subtitle=(
        "Prepare and evaluate forecasts for your own demand data. "
        "Uploaded data stays in this browser session and is not saved to project files."
    ),
    category="USER DATA WORKSPACE",
    badge_text="Session isolated",
    badge_variant="blue",
)

st.info(
    "An **unknown period** means the source has no record and coverage is not "
    "confirmed. It is not treated as zero. A **confirmed zero** means you have "
    "confirmed complete source coverage and explicitly state that absent periods "
    "represent zero demand."
)

uploaded_file = st.file_uploader(
    "Upload a demand CSV",
    type=["csv"],
    max_upload_size=MAX_UPLOAD_SIZE_MB,
    key="user_demand_forecasting_uploader",
    help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB. UTF-8 CSV files are supported.",
)
if uploaded_file is not None:
    try:
        validate_user_demand_upload_size(uploaded_file.size)
        source = store_uploaded_csv(
            st.session_state,
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    except UserDemandUploadError as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()
else:
    source = st.session_state.get("user_demand_forecasting_source")

if not isinstance(source, pd.DataFrame):
    st.caption("Upload a CSV to begin. Your upload and generated tables are kept in this session only.")
    st.stop()

st.subheader("Preview and map columns")
st.dataframe(
    source.head(12),
    hide_index=True,
)
st.caption(
    f"{len(source):,} input rows · {len(source.columns):,} columns · "
    "CSV data is held in session memory only."
)

column_names = source.columns.tolist()
upload_hash = user_upload_hash(st.session_state)
if st.session_state.get(_MAPPING_CONTEXT_KEY) != upload_hash:
    for key in _MAPPING_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state[_MAPPING_CONTEXT_KEY] = upload_hash

suggestions = suggest_user_demand_columns(source)
no_series_column = "(No product or grouping column)"
series_options = [no_series_column, *column_names]
default_series_column = (
    suggestions.product_column or suggestions.category_column
)
if _DATE_COLUMN_KEY not in st.session_state:
    st.session_state[_DATE_COLUMN_KEY] = (
        suggestions.date_column or column_names[0]
    )
if _DEMAND_COLUMN_KEY not in st.session_state:
    st.session_state[_DEMAND_COLUMN_KEY] = (
        suggestions.demand_column or column_names[0]
    )
if _SERIES_COLUMN_KEY not in st.session_state:
    st.session_state[_SERIES_COLUMN_KEY] = (
        default_series_column or no_series_column
    )
initial_frequency_suggestion = suggest_user_demand_frequency(
    source,
    st.session_state[_DATE_COLUMN_KEY],
)
if _FREQUENCY_KEY not in st.session_state:
    st.session_state[_FREQUENCY_KEY] = initial_frequency_suggestion.frequency


def _refresh_aggregation_suggestion() -> None:
    if st.session_state.get(_AGGREGATION_OVERRIDE_KEY, False):
        return
    st.session_state[_AGGREGATION_KEY] = suggest_transaction_aggregation(
        source,
        date_column=st.session_state.get(_DATE_COLUMN_KEY),
        series_column=(
            None
            if st.session_state.get(_SERIES_COLUMN_KEY) == no_series_column
            else st.session_state.get(_SERIES_COLUMN_KEY)
        ),
        frequency=st.session_state.get(_FREQUENCY_KEY),
    )


def _date_column_changed() -> None:
    suggestion = suggest_user_demand_frequency(
        source,
        st.session_state.get(_DATE_COLUMN_KEY),
    )
    st.session_state[_FREQUENCY_KEY] = suggestion.frequency
    _refresh_aggregation_suggestion()


def _aggregation_choice_changed() -> None:
    st.session_state[_AGGREGATION_OVERRIDE_KEY] = True


def _aggregation_inputs_changed() -> None:
    _refresh_aggregation_suggestion()


frequency_options: list[Frequency] = ["daily", "weekly", "monthly"]
if _AGGREGATION_KEY not in st.session_state:
    _refresh_aggregation_suggestion()

st.caption(
    "Value-checked suggestions: "
    f"date {suggestions.date_column or 'not identified'}; "
    f"demand {suggestions.demand_column or 'not identified'}; "
    f"product {suggestions.product_column or 'not identified'}; "
    f"category/group {suggestions.category_column or 'not identified'}. "
    "Suggestions are based on column names and all-row value checks; verify "
    "the mappings before preparation."
)

date_column, demand_column = st.columns(2)
with date_column:
    selected_date_column = st.selectbox(
        "Date or transaction date column",
        column_names,
        key=_DATE_COLUMN_KEY,
        on_change=_date_column_changed,
    )
with demand_column:
    selected_demand_column = st.selectbox(
        "Demand or sales quantity column",
        column_names,
        key=_DEMAND_COLUMN_KEY,
    )
selected_series_option = st.selectbox(
    "Product or grouping column (optional)",
    series_options,
    key=_SERIES_COLUMN_KEY,
    on_change=_aggregation_inputs_changed,
    help=(
        "Choose one product, SKU, item, category, or grouping column to create "
        "separate series. Leaving this unset combines all rows into one series. "
        "The current forecasting data model supports one series identifier at "
        "a time; it does not combine product and category."
    ),
)
selected_series_column = (
    None
    if selected_series_option == no_series_column
    else selected_series_option
)
frequency_suggestion = suggest_user_demand_frequency(
    source,
    selected_date_column,
)
st.caption(
    f"Frequency suggestion: "
    f"{frequency_suggestion.frequency or 'not available'} — "
    f"{frequency_suggestion.explanation} Confirm or change it below."
)
frequency = st.segmented_control(
    "Forecast frequency",
    options=frequency_options,
    format_func=lambda value: value.capitalize(),
    key=_FREQUENCY_KEY,
    on_change=_aggregation_inputs_changed,
)

if _AGGREGATION_KEY not in st.session_state:
    _refresh_aggregation_suggestion()
aggregate_transactions = st.checkbox(
    "Treat rows as additive transactions and sum demand by series and period",
    key=_AGGREGATION_KEY,
    on_change=_aggregation_choice_changed,
    help=(
        "The default is suggested from repeated or unaligned date/series keys. "
        "When selected, every row is retained and its mapped demand value is "
        "summed. Exact duplicate input rows are reported, not dropped. "
        "Uncheck only when each row already represents one period."
    ),
)
mapped_columns = {
    selected_date_column,
    selected_demand_column,
    *([selected_series_column] if selected_series_column is not None else []),
}
unmapped_columns = [
    column for column in column_names if column not in mapped_columns
]
st.caption(
    "Only the selected date, demand, frequency, aggregation rule, and one "
    "series identifier feed forecasting. Other CSV columns are not automatically "
    "used for stock, lead time, costs, supplier, or logistics inputs: "
    + (", ".join(unmapped_columns) if unmapped_columns else "none")
    + "."
)
complete_source_coverage = st.checkbox(
    "I confirm the source covers every period for each series",
    key=_COMPLETE_COVERAGE_KEY,
)
confirm_missing_periods_are_zero = st.checkbox(
    "I confirm absent periods mean zero demand, not missing records",
    key=_ZERO_MISSING_KEY,
    help=(
        "This is applied only when complete source coverage is also confirmed. "
        "Otherwise absent periods remain unknown."
    ),
)
specify_expected_coverage = st.checkbox(
    "Set expected coverage start and end dates",
    key=_EXPECTED_BOUNDS_KEY,
    help=(
        "Expected bounds reveal missing periods before and after the "
        "supplied observations. They remain unknown unless both coverage "
        "and zero-demand confirmations are selected."
    ),
)
expected_start_column, expected_end_column = st.columns(2)
with expected_start_column:
    expected_start = st.date_input(
        "Expected coverage start",
        value=None,
        key=_EXPECTED_START_KEY,
        help="Used only when expected coverage bounds are enabled above.",
    )
with expected_end_column:
    expected_end = st.date_input(
        "Expected coverage end",
        value=None,
        key=_EXPECTED_END_KEY,
        help="Used only when expected coverage bounds are enabled above.",
    )
products_expected_throughout_coverage = st.checkbox(
    "I confirm every selected series was expected to exist throughout this whole window",
    key=_PRODUCTS_EXPECTED_KEY,
    help=(
        "Without this declaration, each series starts and ends at its "
        "own supplied observations. This declaration is used only when "
        "expected bounds and a series identifier are selected."
    ),
)
prepare_clicked = st.button(
    "Validate and prepare data",
    type="primary",
)

configured_expected_start = expected_start if specify_expected_coverage else None
configured_expected_end = expected_end if specify_expected_coverage else None
configured_products_expected_throughout = (
    products_expected_throughout_coverage
    and specify_expected_coverage
    and selected_series_option != no_series_column
)
preparation_signature = (
    upload_hash,
    selected_date_column,
    selected_demand_column,
    selected_series_column,
    frequency,
    aggregate_transactions,
    complete_source_coverage,
    confirm_missing_periods_are_zero,
    configured_expected_start,
    configured_expected_end,
    configured_products_expected_throughout,
)
st.session_state[
    "user_demand_forecasting_current_preparation_signature"
] = preparation_signature

if prepare_clicked:
    if (
        st.session_state.get(
            "user_demand_forecasting_preparation_signature"
        )
        != preparation_signature
    ):
        for key in (
            "user_demand_forecasting_prepared",
            "user_demand_forecasting_diagnostics",
            "user_demand_forecasting_preparation_signature",
            "user_demand_forecasting_backtest",
            "user_demand_forecasting_backtest_signature",
            "user_demand_forecasting_current_backtest_signature",
            "user_demand_forecasting_forecasts",
            "user_demand_forecasting_forecast_warnings",
            "user_demand_forecasting_forecast_signature",
            "user_demand_forecasting_current_forecast_signature",
            "user_demand_forecasting_model_selections",
            "user_demand_forecasting_model_selection_context",
        ):
            st.session_state.pop(key, None)
        for key in list(st.session_state.keys()):
            if key.startswith(_MODEL_CHOICE_PREFIX):
                st.session_state.pop(key, None)
    if specify_expected_coverage and (
        configured_expected_start is None or configured_expected_end is None
    ):
        st.error("Choose both expected coverage dates.")
    elif frequency is None:
        st.error("Choose a forecast frequency.")
    elif frequency not in frequency_options:
        st.error("Choose a supported forecast frequency.")
    else:
        selected_frequency = cast(Frequency, frequency)
        try:
            prepared_upload = prepare_user_upload(
                source,
                date_column=selected_date_column,
                demand_column=selected_demand_column,
                product_column=selected_series_column,
                frequency=selected_frequency,
                aggregate_transactions=aggregate_transactions,
                complete_source_coverage=complete_source_coverage,
                confirm_missing_periods_are_zero=(
                    confirm_missing_periods_are_zero
                ),
                expected_start=configured_expected_start,
                expected_end=configured_expected_end,
                products_expected_throughout_coverage=(
                    configured_products_expected_throughout
                ),
            )
        except UserDemandForecastingError as exc:
            st.error(str(exc), icon=":material/error:")
            if exc.invalid_rows is not None and not exc.invalid_rows.empty:
                st.dataframe(
                    exc.invalid_rows.head(100),
                    hide_index=True,
                )
                if len(exc.invalid_rows) > 100:
                    st.caption("Showing the first 100 invalid rows.")
        else:
            st.session_state["user_demand_forecasting_prepared"] = prepared_upload
            st.session_state[
                "user_demand_forecasting_preparation_signature"
            ] = preparation_signature
            for key in (
                "user_demand_forecasting_backtest",
                "user_demand_forecasting_backtest_signature",
                "user_demand_forecasting_forecasts",
                "user_demand_forecasting_forecast_warnings",
                "user_demand_forecasting_forecast_signature",
            ):
                st.session_state.pop(key, None)
            for key in list(st.session_state.keys()):
                if key.startswith(_MODEL_CHOICE_PREFIX):
                    st.session_state.pop(key, None)
            st.success("Data validated and prepared. No historical values were interpolated.")

prepared_upload = st.session_state.get("user_demand_forecasting_prepared")
stored_preparation_signature = st.session_state.get(
    "user_demand_forecasting_preparation_signature"
)
if not isinstance(prepared_upload, PreparedUserUpload):
    st.stop()
if stored_preparation_signature != preparation_signature:
    st.warning(
        "The upload mapping or preparation options have changed. Validate and "
        "prepare again before comparing models."
    )
    st.stop()

demand = prepared_upload.demand
diagnostics = prepared_upload.diagnostics
quality = demand.quality_report

st.divider()
st.subheader("Data quality and demand patterns")
with st.container(horizontal=True):
    st.metric("Input rows", f"{quality.input_rows:,}", border=True)
    st.metric("Valid rows", f"{quality.valid_rows:,}", border=True)
    st.metric("Missing periods", f"{quality.missing_periods:,}", border=True)
    st.metric("Series", f"{quality.number_of_products:,}", border=True)

if demand.aggregation_summary is not None:
    st.caption(
        f"Explicit transaction aggregation: {demand.aggregation_summary.input_rows:,} "
        f"rows summed into {demand.aggregation_summary.aggregated_period_rows:,} "
        "period/series observations. Every row was retained; exact duplicate "
        f"input rows: {demand.aggregation_summary.exact_duplicate_rows:,}."
    )
for warning in quality.validation_warnings:
    st.warning(warning, icon=":material/warning:")
if diagnostics.warnings:
    for warning in diagnostics.warnings:
        st.caption(warning)

quality_rows = pd.DataFrame(
    [
        {"Finding": "Input rows", "Value": str(quality.input_rows)},
        {"Finding": "Valid rows", "Value": str(quality.valid_rows)},
        {"Finding": "Invalid rows", "Value": str(quality.invalid_rows)},
        {
            "Finding": "Duplicate period/series keys",
            "Value": str(quality.duplicate_rows),
        },
        {
            "Finding": "Exact duplicate transaction rows",
            "Value": (
                str(demand.aggregation_summary.exact_duplicate_rows)
                if demand.aggregation_summary is not None
                else "Not checked in period-level mode"
            ),
        },
        {"Finding": "Date range", "Value": (
            f"{quality.date_range[0].date()} to {quality.date_range[1].date()}"
            if quality.date_range
            else "Unavailable"
        )},
        {"Finding": "Frequency", "Value": quality.frequency},
        {"Finding": "Missing periods", "Value": str(quality.missing_periods)},
        {
            "Finding": "Longest consecutive unknown gap",
            "Value": f"{_largest_unknown_run(demand.data)} {quality.frequency} periods",
        },
        {
            "Finding": "Series below minimum diagnostic history",
            "Value": (
                f"{sum('insufficient_history' in item.classifications for item in diagnostics.series)} "
                f"(minimum {diagnostics.thresholds.min_history_periods})"
            ),
        },
        {
            "Finding": "Missing before first supplied period",
            "Value": str(quality.missing_before_first_observation),
        },
        {
            "Finding": "Missing within supplied history",
            "Value": str(quality.missing_internal_periods),
        },
        {
            "Finding": "Missing after last supplied period",
            "Value": str(quality.missing_after_last_observation),
        },
    ]
)
with st.expander("Validation summary", expanded=False):
    st.dataframe(
        quality_rows,
        hide_index=True,
    )

diagnostic_table = diagnostics_frame(diagnostics)
st.dataframe(
    diagnostic_table,
    hide_index=True,
)
st.caption(
    "Pattern labels are screening heuristics, not statistical proof or a "
    "guarantee of future accuracy. Unknown periods are excluded from demand statistics."
)

series_ids = [item.product_id for item in diagnostics.series]
if "product_id" in demand.data.columns:
    selected_product_index = st.selectbox(
        "Series to inspect",
        options=list(range(len(series_ids))),
        format_func=lambda index: str(series_ids[index]),
        key="user_demand_forecasting_product_index",
    )
    selected_product_id = series_ids[selected_product_index]
    historical = demand.data.loc[
        demand.data["product_id"].eq(selected_product_id)
    ].copy()
else:
    selected_product_index = 0
    selected_product_id = None
    historical = demand.data.copy()

for item in diagnostics.series:
    if item.excluded_before_first_observation_periods or item.excluded_after_last_observation_periods:
        st.caption(
            f"Series {item.product_id!r}: "
            f"{item.excluded_before_first_observation_periods} periods before "
            "its first supplied observation and "
            f"{item.excluded_after_last_observation_periods} periods after its "
            "last supplied observation were excluded from its history because "
            "product-wide coverage was not declared."
        )

st.subheader(
    f"Historical demand — {selected_product_id}"
    if selected_product_id is not None
    else "Historical demand — single series"
)
st.line_chart(
    historical,
    x="period",
    y="demand_quantity",
)
st.dataframe(
    historical,
    hide_index=True,
)

st.divider()
st.subheader("Rolling-origin model comparison")
st.caption(
    "Models are compared on chronological expanding windows. Only models that "
    "meet the minimum origin and common-coverage requirements can be selected."
)
with st.form("user_demand_forecasting_comparison_form"):
    horizon = st.number_input(
        "Forecast horizon (periods)",
        min_value=1,
        max_value=365,
        value=3,
        step=1,
    )
    min_train_periods = st.number_input(
        "Minimum training periods per origin",
        min_value=2,
        max_value=1000,
        value=8,
        step=1,
    )
    min_successful_origins = st.number_input(
        "Minimum successful origins per model",
        min_value=1,
        max_value=100,
        value=2,
        step=1,
    )
    min_comparable_origins = st.number_input(
        "Minimum shared origins for selection",
        min_value=1,
        max_value=100,
        value=2,
        step=1,
    )
    selection_metric = st.selectbox(
        "Model selection metric",
        options=["mae", "rmse", "wape"],
        format_func=str.upper,
        index=0,
    )
    min_success_fraction = st.slider(
        "Minimum successful-origin rate",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )
    compare_clicked = st.form_submit_button(
        "Run rolling-origin comparison",
        type="primary",
    )

backtest_signature = (
    preparation_signature,
    int(horizon),
    int(min_train_periods),
    int(min_successful_origins),
    int(min_comparable_origins),
    selection_metric,
    float(min_success_fraction),
)
st.session_state["user_demand_forecasting_current_backtest_signature"] = (
    backtest_signature
)

if compare_clicked:
    try:
        backtest = compare_user_demand_models(
            demand,
            diagnostics,
            horizon=int(horizon),
            min_train_periods=int(min_train_periods),
            min_successful_origins=int(min_successful_origins),
            min_comparable_origins=int(min_comparable_origins),
            min_success_fraction=float(min_success_fraction),
            selection_metric=selection_metric,
        )
    except (TypeError, ValueError) as exc:
        st.error(f"Model comparison could not run: {exc}")
    else:
        st.session_state["user_demand_forecasting_backtest"] = backtest
        st.session_state[
            "user_demand_forecasting_backtest_signature"
        ] = backtest_signature
        st.session_state.pop("user_demand_forecasting_forecasts", None)
        st.session_state.pop("user_demand_forecasting_forecast_warnings", None)
        st.session_state.pop("user_demand_forecasting_forecast_signature", None)
        st.session_state.pop(
            "user_demand_forecasting_current_forecast_signature", None
        )
        for key in list(st.session_state.keys()):
            if key.startswith(_MODEL_CHOICE_PREFIX):
                st.session_state.pop(key, None)

backtest = st.session_state.get("user_demand_forecasting_backtest")
stored_backtest_signature = st.session_state.get(
    "user_demand_forecasting_backtest_signature"
)
if backtest is None:
    st.info("Run the rolling-origin comparison to see model metrics and eligibility.")
    st.stop()
if stored_backtest_signature != backtest_signature:
    for key in (
        "user_demand_forecasting_forecasts",
        "user_demand_forecasting_forecast_warnings",
        "user_demand_forecasting_forecast_signature",
        "user_demand_forecasting_current_forecast_signature",
    ):
        st.session_state.pop(key, None)
    st.warning(
        "The comparison settings changed. Run model comparison again before "
        "selecting or forecasting."
    )
    st.stop()

evaluation_table = build_evaluation_export(backtest)
st.dataframe(
    evaluation_table,
    hide_index=True,
)
skipped_table = evaluation_table.loc[
    evaluation_table["record_type"].eq("skipped_window")
]
st.caption(f"Skipped evaluation windows: {len(skipped_table):,}")
if not skipped_table.empty:
    st.warning(
        "These windows were skipped because actual target periods were "
        "unknown; they are not model failures or ineligible-model attempts."
    )
    st.dataframe(
        skipped_table[
            [
                "product_id",
                "origin_period",
                "evaluation_periods",
                "skipped_reason",
            ]
        ],
        hide_index=True,
    )
for selection in backtest.selections:
    product_label = (
        f"Product {selection.product_id}" if selection.product_id is not None else "Single series"
    )
    if selection.selected_model is None:
        st.warning(
            f"{product_label}: {selection.rationale} No validated recommendation "
            "is available for this series.",
            icon=":material/warning:",
        )
    else:
        st.success(
            f"{product_label}: recommended {selection.selected_model}. "
            f"{selection.rationale}",
            icon=":material/check_circle:",
        )

st.download_button(
    "Download evaluation CSV",
    data=evaluation_table.to_csv(index=False),
    file_name="user_demand_model_evaluation.csv",
    mime="text/csv",
)

selected_series = next(
    (
        selection
        for selection in backtest.selections
        if selection.product_id == selected_product_id
    ),
    None,
)
eligible_model_names = (
    {name for name, _ in selected_series.candidate_scores}
    if selected_series is not None
    else set()
)
valid_models = [
    summary.model_name
    for summary in backtest.model_summaries
    if summary.product_id == selected_product_id
    and summary.model_name in eligible_model_names
]
recommended = (
    selected_series.selected_model
    if selected_series is not None
    else None
)
selection_context = hashlib.sha256(
    repr(preparation_signature).encode("utf-8")
).hexdigest()
stored_model_choices = product_model_selections(
    st.session_state,
    context=selection_context,
)
selected_product_key = product_selection_key(selected_product_id)
has_saved_choice = selected_product_key in stored_model_choices
saved_choice = stored_model_choices.get(selected_product_key)
recommendation_label = (
    f"Use recommendation: {recommended}" if recommended is not None else None
)
choice_options = [_CHOOSE_MODEL, *valid_models]
if recommendation_label is not None:
    choice_options.append(recommendation_label)
stale_saved_choice = has_saved_choice and (
    saved_choice not in valid_models
    and saved_choice != USE_RECOMMENDATION
    and saved_choice != _CHOOSE_MODEL
)
stale_recommendation = (
    has_saved_choice
    and saved_choice == USE_RECOMMENDATION
    and recommendation_label is None
)
if stale_saved_choice:
    st.warning(
        f"The previously selected model {saved_choice!r} is no longer eligible "
        "for this product. Choose an eligible model or explicitly use the "
        "current recommendation."
    )
    default_choice = _CHOOSE_MODEL
elif stale_recommendation:
    st.warning(
        "The previously chosen recommendation is no longer available. Choose "
        "an eligible model or wait until a validated recommendation exists."
    )
    default_choice = _CHOOSE_MODEL
elif (
    has_saved_choice
    and saved_choice == USE_RECOMMENDATION
    and recommendation_label is not None
):
    default_choice = recommendation_label
elif has_saved_choice and saved_choice in valid_models:
    default_choice = saved_choice
elif has_saved_choice and saved_choice == _CHOOSE_MODEL:
    default_choice = _CHOOSE_MODEL
else:
    default_choice = recommendation_label or _CHOOSE_MODEL
widget_key = (
    f"{_MODEL_CHOICE_PREFIX}{selection_context[:12]}_"
    f"{selected_product_index}"
)
if (stale_saved_choice or stale_recommendation) and widget_key in st.session_state:
    st.session_state.pop(widget_key)
if (
    widget_key in st.session_state
    and st.session_state[widget_key] not in choice_options
):
    st.session_state.pop(widget_key)
model_choice = st.selectbox(
    "Model for the selected product",
    options=choice_options,
    index=choice_options.index(default_choice),
    key=widget_key,
    help=(
        "Only models that met the rolling-origin evaluation requirements on "
        "comparable origins are available."
    ),
)
if model_choice in valid_models:
    save_product_model_selection(
        st.session_state,
        context=selection_context,
        product_id=selected_product_id,
        model_name=model_choice,
    )
elif recommendation_label is not None and model_choice == recommendation_label:
    save_product_model_selection(
        st.session_state,
        context=selection_context,
        product_id=selected_product_id,
        model_name=USE_RECOMMENDATION,
    )
elif selected_product_key not in stored_model_choices:
    save_product_model_selection(
        st.session_state,
        context=selection_context,
        product_id=selected_product_id,
        model_name=None,
    )
selected_model_for_product = (
    recommended
    if model_choice == recommendation_label
    else model_choice if model_choice in valid_models else None
)
if selected_model_for_product is not None and selected_model_for_product != recommended:
    st.info(
        "You selected a model that passed comparable rolling-origin evaluation; "
        "the recommendation remains based on the configured selection metric."
    )

st.caption(
    "Forecasts are estimates, not guarantees. A series ending in an unknown "
    "period will not be forecast, and invalid or non-finite predictions are rejected."
)
chosen_models: dict[object | None, str | None] = {}
unresolved_model_choices: list[str] = []
for selection in backtest.selections:
    product_key = product_selection_key(selection.product_id)
    saved = stored_model_choices.get(product_key, _CHOOSE_MODEL)
    eligible_for_product = {
        model_name for model_name, _ in selection.candidate_scores
    }
    if saved == USE_RECOMMENDATION:
        chosen_models[selection.product_id] = selection.selected_model
        if selection.selected_model is None:
            unresolved_model_choices.append(
                f"{selection.product_id!r}: no validated recommendation remains"
            )
    elif saved in eligible_for_product:
        chosen_models[selection.product_id] = saved
    elif saved == _CHOOSE_MODEL and product_key not in stored_model_choices:
        chosen_models[selection.product_id] = selection.selected_model
    elif saved in (None, _CHOOSE_MODEL):
        chosen_models[selection.product_id] = None
    else:
        chosen_models[selection.product_id] = None
        unresolved_model_choices.append(
            f"{selection.product_id!r}: {saved!r} is no longer eligible"
        )
if unresolved_model_choices:
    st.warning(
        "Resolve these saved product choices before forecasting: "
        + "; ".join(unresolved_model_choices)
    )
chosen_models[selected_product_id] = selected_model_for_product
current_forecast_signature = (
    backtest_signature,
    tuple(chosen_models.items()),
)
st.session_state[
    "user_demand_forecasting_current_forecast_signature"
] = current_forecast_signature

if st.button(
    "Generate future forecasts",
    type="primary",
    disabled=(
        not any(model is not None for model in chosen_models.values())
        or bool(unresolved_model_choices)
        or model_choice == _CHOOSE_MODEL
    ),
):
    try:
        with st.spinner("Generating future forecasts..."):
            forecast_results, forecast_warnings = generate_user_forecasts(
                demand,
                diagnostics,
                backtest,
                horizon=int(horizon),
                selected_models=chosen_models,
            )
    except (TypeError, ValueError) as exc:
        st.error(f"Forecast generation failed: {exc}")
    else:
        st.session_state["user_demand_forecasting_forecasts"] = forecast_results
        st.session_state[
            "user_demand_forecasting_forecast_warnings"
        ] = forecast_warnings
        st.session_state[
            "user_demand_forecasting_forecast_signature"
        ] = current_forecast_signature
        set_operational_source(st.session_state, USER_UPLOAD_SOURCE)
        if forecast_results:
            st.success(
                "Forecast generation is complete. This upload is now the active "
                "source for Inventory Management, Supply Chain Risk, and "
                "Scenario Simulator."
            )
        else:
            st.warning(
                "Forecast generation finished without eligible forecast rows. "
                "The uploaded source is active so operational pages will not "
                "fall back to DataCo; forecast-dependent planning may be unavailable."
            )

forecasts = st.session_state.get("user_demand_forecasting_forecasts")
forecast_warnings = st.session_state.get(
    "user_demand_forecasting_forecast_warnings"
)
forecast_signature = st.session_state.get(
    "user_demand_forecasting_forecast_signature"
)
if forecasts is not None and forecast_signature is not None:
    expected_forecast_signature = (
        backtest_signature,
        tuple(chosen_models.items()),
    )
    if forecast_signature == expected_forecast_signature:
        forecast_export = build_forecast_export(forecasts, backtest)
        st.subheader("Future forecasts")
        if forecast_warnings is not None and not forecast_warnings.empty:
            st.warning(
                "Some product forecasts were withheld or could not be generated."
            )
            st.dataframe(
                forecast_warnings,
                hide_index=True,
            )
        if not forecast_export.empty:
            selected_forecast = (
                forecast_export.loc[forecast_export["product_id"].isna()]
                if selected_product_id is None
                else forecast_export.loc[
                    forecast_export["product_id"].eq(selected_product_id)
                ]
            )
            if not selected_forecast.empty:
                st.line_chart(
                    selected_forecast,
                    x="period",
                    y="predicted_demand",
                )
            st.dataframe(
                forecast_export,
                hide_index=True,
            )
        else:
            st.warning("No forecast rows were produced for the selected model choices.")
        st.download_button(
            "Download forecast CSV",
            data=forecast_export.to_csv(index=False),
            file_name="user_demand_forecasts.csv",
            mime="text/csv",
        )
        forecast_available = not forecast_export.empty
        st.subheader("Operational page readiness")
        readiness = pd.DataFrame(
            [
                {
                    "Page": "Inventory Management",
                    "Status": (
                        "Ready on page access"
                        if forecast_available
                        else "Unavailable"
                    ),
                    "Details": (
                        "Demand and forecast planning will calculate automatically. "
                        "Current stock and lead time remain optional user inputs."
                        if forecast_available
                        else "Requires at least one eligible forecast row."
                    ),
                },
                {
                    "Page": "Supply Chain Risk",
                    "Status": "Ready on page access",
                    "Details": (
                        "Demand diagnostics, unknown-period status, and backtest "
                        "indicators will calculate automatically; supplier and "
                        "logistics risk are not measured."
                    ),
                },
                {
                    "Page": "Scenario Simulator",
                    "Status": (
                        "Ready on page access"
                        if forecast_available
                        else "Unavailable"
                    ),
                    "Details": (
                        "Baseline and default +10%/-10% scenarios use saved forecasts."
                        if forecast_available
                        else "Requires at least one eligible forecast row."
                    ),
                },
            ]
        )
        st.dataframe(
            readiness,
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Forecast model choices changed. Generate forecasts again to refresh the results.")
