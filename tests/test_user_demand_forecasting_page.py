import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dashboard.user_demand_forecasting as user_demand_forecasting
from dashboard.operational_data import USER_UPLOAD_SOURCE
from dashboard.user_demand_forecasting import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_SIZE_MB,
    UserDemandForecastingError,
    UserDemandUploadError,
    build_evaluation_export,
    build_forecast_export,
    compare_user_demand_models,
    generate_user_forecasts,
    parse_user_demand_csv,
    prepare_user_upload,
    product_model_selections,
    product_selection_key,
    save_product_model_selection,
    store_uploaded_csv,
    suggest_transaction_aggregation,
    suggest_user_demand_columns,
    suggest_user_demand_frequency,
    USE_RECOMMENDATION,
    validate_user_demand_upload_size,
)


def test_csv_upload_preserves_string_identifiers_and_column_mapping():
    raw = b"period,units,sku\n2024-01-01,3,001-A\n2024-02-01,4,001-A\n"
    uploaded = parse_user_demand_csv("demand.CSV", raw)

    assert uploaded.loc[0, "sku"] == "001-A"
    assert uploaded.loc[0, "units"] == "3"
    result = prepare_user_upload(
        uploaded,
        date_column="period",
        demand_column="units",
        product_column="sku",
        frequency="monthly",
    )
    assert result.demand.data["product_id"].unique().tolist() == ["001-A"]
    assert result.demand.data["demand_quantity"].tolist() == [3.0, 4.0]


def test_column_suggestions_use_business_headers_and_validate_values():
    source = pd.DataFrame(
        {
            "Transaction Date": ["2024-01-03", "2024-01-18"],
            "Units Sold": ["2", "5"],
            "Item Code": ["001-A", "001-B"],
            "Product Category": ["Hardware", "Hardware"],
            "Sales Amount": ["30.00", "75.00"],
        }
    )

    suggestions = suggest_user_demand_columns(source)

    assert suggestions.date_column == "Transaction Date"
    assert suggestions.demand_column == "Units Sold"
    assert suggestions.product_column == "Item Code"
    assert suggestions.category_column == "Product Category"


def test_invalid_column_values_prevent_automatic_role_suggestions():
    source = pd.DataFrame(
        {
            "Order Date": ["2024-01-01", "not a date"],
            "Quantity": ["2", "-1"],
            "SKU": ["A", ""],
        }
    )

    suggestions = suggest_user_demand_columns(source)

    assert suggestions.date_column is None
    assert suggestions.demand_column is None
    assert suggestions.product_column is None


@pytest.mark.parametrize(
    ("name", "dates", "expected"),
    [
        ("Transaction Date", ["2024-01-01", "2024-01-02"], "daily"),
        ("Order Date", ["2024-01-01", "2024-01-08"], "weekly"),
        ("Date", ["2024-01-01", "2024-02-01"], "monthly"),
        ("Date", ["2024-01-31", "2024-02-29"], "monthly"),
    ],
)
def test_frequency_suggestions_follow_date_cadence(name, dates, expected):
    source = pd.DataFrame({name: dates})

    suggestion = suggest_user_demand_frequency(source, name)

    assert suggestion.frequency == expected


def test_transaction_mode_is_suggested_for_repeated_period_series_keys():
    source = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-18", "2024-02-02"],
            "sku": ["A", "A", "A"],
        }
    )

    assert suggest_transaction_aggregation(
        source,
        date_column="date",
        series_column="sku",
        frequency="monthly",
    )


def test_streamlit_mapping_suggestions_can_be_overridden_and_reset_on_upload():
    page = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "pages"
        / "08_user_demand_forecasting.py"
    )
    transaction_source = pd.DataFrame(
        {
            "Transaction Date": [
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
            ],
            "Units Sold": [2, 3, 4],
            "SKU": ["001-A", "001-A", "001-B"],
            "Product Category": ["Hardware", "Hardware", "Software"],
        }
    )
    app = AppTest.from_file(str(page), default_timeout=45).run()
    app.file_uploader[0].upload(
        "transactions.csv",
        transaction_source.to_csv(index=False).encode(),
        "text/csv",
    ).run()

    def selectbox(label):
        return next(element for element in app.selectbox if element.label == label)

    assert selectbox("Date or transaction date column").value == "Transaction Date"
    assert selectbox("Demand or sales quantity column").value == "Units Sold"
    assert selectbox("Product or grouping column (optional)").value == "SKU"
    assert app.session_state["user_demand_forecasting_frequency"] == "daily"
    aggregation = next(
        item
        for item in app.checkbox
        if item.label.startswith("Treat rows as additive transactions")
    )
    assert aggregation.value
    assert any("Product Category" in item.value for item in app.caption)

    selectbox("Product or grouping column (optional)").select(
        "Product Category"
    ).run()
    assert (
        selectbox("Product or grouping column (optional)").value
        == "Product Category"
    )

    period_source = pd.DataFrame(
        {
            "Month": ["2024-01-01", "2024-02-01"],
            "Demand": [12, 15],
            "Department": ["Hardware", "Hardware"],
        }
    )
    app.file_uploader[0].upload(
        "periods.csv",
        period_source.to_csv(index=False).encode(),
        "text/csv",
    ).run()

    assert selectbox("Date or transaction date column").value == "Month"
    assert selectbox("Demand or sales quantity column").value == "Demand"
    assert selectbox("Product or grouping column (optional)").value == "Department"
    assert app.session_state["user_demand_forecasting_frequency"] == "monthly"
    aggregation = next(
        item
        for item in app.checkbox
        if item.label.startswith("Treat rows as additive transactions")
    )
    assert not aggregation.value
    assert not app.exception


@pytest.mark.parametrize(
    ("frequency", "dates"),
    [
        ("daily", ["2024-01-01", "2024-01-02"]),
        ("weekly", ["2024-01-01", "2024-01-08"]),
        ("monthly", ["2024-01-01", "2024-02-01"]),
    ],
)
def test_frequency_selection_prepares_aligned_period_data(frequency, dates):
    source = pd.DataFrame({"when": dates, "amount": [2, 5]})
    result = prepare_user_upload(
        source,
        date_column="when",
        demand_column="amount",
        product_column=None,
        frequency=frequency,
    )
    assert result.demand.frequency == frequency
    assert result.demand.data["period"].tolist() == list(
        pd.to_datetime(dates)
    )


@pytest.mark.parametrize(
    ("frequency", "start", "end", "expected_start", "expected_end"),
    [
        ("daily", "2024-01-02", "2024-01-04", "2024-01-02", "2024-01-04"),
        ("weekly", "2024-01-03", "2024-01-17", "2024-01-01", "2024-01-15"),
        ("monthly", "2024-02-20", "2024-04-30", "2024-02-01", "2024-04-01"),
    ],
)
def test_user_upload_forwards_normalized_expected_coverage(
    frequency, start, end, expected_start, expected_end
):
    source_dates = {
        "daily": ["2024-01-02", "2024-01-04"],
        "weekly": ["2024-01-01", "2024-01-15"],
        "monthly": ["2024-02-01", "2024-04-01"],
    }[frequency]
    prepared = prepare_user_upload(
        pd.DataFrame({"date": source_dates, "demand": [2, 4]}),
        date_column="date",
        demand_column="demand",
        product_column=None,
        frequency=frequency,
        expected_start=start,
        expected_end=end,
    )

    assert prepared.demand.expected_coverage_start == pd.Timestamp(expected_start)
    assert prepared.demand.expected_coverage_end == pd.Timestamp(expected_end)


def test_transaction_rows_are_summed_only_when_explicitly_requested():
    source = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-18", "2024-02-02"],
            "qty": [2, 3, 5],
            "sku": ["A", "A", "A"],
        }
    )
    before = source.copy(deep=True)
    result = prepare_user_upload(
        source,
        date_column="date",
        demand_column="qty",
        product_column="sku",
        frequency="monthly",
        aggregate_transactions=True,
    )
    assert result.demand.data["demand_quantity"].tolist() == [5.0, 5.0]
    assert result.demand.aggregation_summary.input_rows == 3
    pd.testing.assert_frame_equal(source, before)


def test_transaction_aggregation_preserves_expected_coverage_metadata():
    source = pd.DataFrame(
        {
            "date": ["2024-02-03", "2024-04-15"],
            "qty": [2, 4],
            "sku": ["A", "A"],
        }
    )
    result = prepare_user_upload(
        source,
        date_column="date",
        demand_column="qty",
        product_column="sku",
        frequency="monthly",
        aggregate_transactions=True,
        expected_start="2024-01-01",
        expected_end="2024-05-01",
    )

    assert result.demand.data["period"].tolist() == list(
        pd.to_datetime(
            ["2024-02-01", "2024-03-01", "2024-04-01"]
        )
    )
    coverage = result.demand.coverage_by_product.iloc[0]
    assert coverage["missing_internal_periods"] == 1
    assert coverage["excluded_before_first_observation_periods"] == 1
    assert coverage["excluded_after_last_observation_periods"] == 1


def test_duplicate_aggregated_periods_are_actionable_validation_errors():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "qty": [2, 3],
        }
    )
    with pytest.raises(UserDemandForecastingError, match="invalid rows") as error:
        prepare_user_upload(
            source,
            date_column="date",
            demand_column="qty",
            product_column=None,
            frequency="monthly",
        )
    assert error.value.invalid_rows is not None
    assert error.value.invalid_rows["_validation_errors"].str.contains(
        "duplicate"
    ).all()


def test_unknown_and_confirmed_zero_periods_remain_distinct():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-03-01"],
            "qty": [2, 7],
        }
    )
    options = {
        "date_column": "date",
        "demand_column": "qty",
        "product_column": None,
        "frequency": "monthly",
    }
    unknown = prepare_user_upload(source, **options)
    assert unknown.demand.data.loc[1, "observation_status"] == (
        "unknown_unobserved_period"
    )
    assert pd.isna(unknown.demand.data.loc[1, "demand_quantity"])
    assert not bool(unknown.demand.data.loc[1, "training_eligible"])

    confirmed = prepare_user_upload(
        source,
        **options,
        complete_source_coverage=True,
        confirm_missing_periods_are_zero=True,
    )
    assert confirmed.demand.data.loc[1, "observation_status"] == (
        "confirmed_zero_demand"
    )
    assert confirmed.demand.data.loc[1, "demand_quantity"] == 0
    assert confirmed.demand.data.loc[1, "training_eligible"]

    with pytest.raises(UserDemandForecastingError, match="complete source coverage"):
        prepare_user_upload(
            source,
            **options,
            confirm_missing_periods_are_zero=True,
        )


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b" \n",
        b"date,demand\n",
        b"date,demand,demand\n2024-01-01,2,3\n",
        b",demand\n2024-01-01,2\n",
        b"date,demand\n2024-01-01,2\n2024-02-01,3,extra\n",
        "date,demand\n2024-01-01,caf\u00e9\n".encode("latin-1"),
    ],
)
def test_csv_upload_rejects_empty_malformed_or_unsupported_content(content):
    with pytest.raises(UserDemandUploadError):
        parse_user_demand_csv("input.csv", content)


def test_csv_upload_rejects_unsupported_extensions_and_oversized_files():
    with pytest.raises(UserDemandUploadError, match="\\.csv"):
        parse_user_demand_csv("input.xlsx", b"date,demand\n2024-01-01,1\n")


@pytest.mark.parametrize(
    ("size", "rejected"),
    [
        (MAX_UPLOAD_BYTES - 1, False),
        (MAX_UPLOAD_BYTES, False),
        (MAX_UPLOAD_BYTES + 1, True),
    ],
)
def test_upload_size_limit_boundaries(size, rejected):
    if rejected:
        with pytest.raises(UserDemandUploadError, match="100 MB"):
            validate_user_demand_upload_size(size)
    else:
        validate_user_demand_upload_size(size)


def test_csv_parser_checks_oversized_input_without_allocating_large_file(monkeypatch):
    monkeypatch.setattr(user_demand_forecasting, "MAX_UPLOAD_BYTES", 3)

    with pytest.raises(UserDemandUploadError, match="100 MB"):
        parse_user_demand_csv("large.csv", b"four")


def test_streamlit_upload_limit_is_configured_for_100_mb():
    project_root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (project_root / ".streamlit" / "config.toml").read_text(
            encoding="utf-8"
        )
    )

    assert MAX_UPLOAD_SIZE_MB == 100
    assert config["server"]["maxUploadSize"] == 100


def test_session_upload_storage_is_isolated_and_returns_copies():
    first_session = defaultdict(str)
    second_session = defaultdict(str)
    first = store_uploaded_csv(
        first_session,
        "one.csv",
        b"date,demand,sku\n2024-01-01,1,0001\n",
    )
    second = store_uploaded_csv(
        second_session,
        "two.csv",
        b"date,demand,sku\n2024-01-01,2,0002\n",
    )
    first.loc[0, "sku"] = "changed"

    assert first_session["user_demand_forecasting_source"].loc[0, "sku"] == "0001"
    assert second_session["user_demand_forecasting_source"].loc[0, "sku"] == "0002"


def test_manual_model_choices_persist_per_product_and_switch_independently():
    session = {}
    context = "dataset-context"
    save_product_model_selection(
        session,
        context=context,
        product_id="A",
        model_name="Naive",
    )
    save_product_model_selection(
        session,
        context=context,
        product_id="B",
        model_name="Moving Average",
    )

    choices = product_model_selections(session, context=context)
    assert choices[product_selection_key("A")] == "Naive"
    assert choices[product_selection_key("B")] == "Moving Average"


def test_new_dataset_context_does_not_inherit_manual_model_choices():
    session = {}
    save_product_model_selection(
        session,
        context="first-upload",
        product_id="shared-id",
        model_name="Naive",
    )

    choices = product_model_selections(session, context="second-upload")

    assert choices == {}


def test_new_upload_clears_persisted_model_choices():
    session = {}
    store_uploaded_csv(
        session,
        "first.csv",
        b"date,demand,sku\n2024-01-01,1,A\n",
    )
    save_product_model_selection(
        session,
        context="first-upload",
        product_id="A",
        model_name="Naive",
    )

    store_uploaded_csv(
        session,
        "second.csv",
        b"date,demand,sku\n2024-01-01,2,A\n",
    )

    assert "user_demand_forecasting_model_selections" not in session
    assert "user_demand_forecasting_model_selection_context" not in session


def test_recommendation_is_stored_as_an_explicit_distinct_choice():
    session = {}
    save_product_model_selection(
        session,
        context="dataset-context",
        product_id="A",
        model_name=USE_RECOMMENDATION,
    )

    assert product_model_selections(
        session,
        context="dataset-context",
    )[product_selection_key("A")] == USE_RECOMMENDATION


def test_new_invalid_upload_cannot_reuse_a_previously_valid_session_file():
    session = {}
    content = b"date,demand\n2024-01-01,1\n"
    store_uploaded_csv(session, "demand.csv", content)

    with pytest.raises(UserDemandUploadError):
        store_uploaded_csv(session, "demand.xlsx", content)

    assert "user_demand_forecasting_source" not in session
    assert "user_demand_forecasting_upload_hash" not in session


def test_streamlit_page_upload_and_prepare_flow_runs_in_session():
    page = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "pages"
        / "08_user_demand_forecasting.py"
    )
    csv_content = pd.DataFrame(
        {
            "period": pd.date_range("2020-01-01", periods=24, freq="MS"),
            "demand": [10 + index % 5 for index in range(24)],
        }
    ).to_csv(index=False).encode()
    app = AppTest.from_file(str(page), default_timeout=45).run()
    app.file_uploader[0].upload(
        "sample.csv",
        csv_content,
        "text/csv",
    ).run()
    app.button[0].click().run()
    assert any(
        "Data validated and prepared" in element.value for element in app.success
    )
    app.button[1].click().run()
    app.button[-1].click().run()

    assert not app.exception
    assert any("recommended" in element.value for element in app.success)
    assert {element.label for element in app.get("download_button")} == {
        "Download evaluation CSV",
        "Download forecast CSV",
    }
    assert (
        app.session_state["operational_dashboard_source"] == USER_UPLOAD_SOURCE
    )
    assert (
        app.session_state["operational_dashboard_source_selector"]
        == USER_UPLOAD_SOURCE
    )
    assert any(
        element.value == "Operational page readiness"
        for element in app.subheader
    )

    pages = Path(__file__).resolve().parents[1] / "dashboard" / "pages"
    for filename in (
        "05_inventory_management.py",
        "06_supply_chain_risk.py",
        "07_scenario_simulator.py",
    ):
        operational_app = AppTest.from_file(
            str(pages / filename),
            default_timeout=45,
        )
        for key, value in app.session_state.to_dict().items():
            operational_app.session_state[key] = value
        operational_app.run()

        assert not operational_app.exception, (
            filename,
            operational_app.exception,
        )
        assert operational_app.selectbox[0].value == USER_UPLOAD_SOURCE
        assert operational_app.dataframe
        assert not operational_app.get("file_uploader")
        assert not any(
            "run" in button.label.lower()
            or "calculate" in button.label.lower()
            for button in operational_app.button
        )


def test_streamlit_page_restores_manual_choice_after_switching_products():
    page = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "pages"
        / "08_user_demand_forecasting.py"
    )
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    source = pd.DataFrame(
        [
            {"period": date, "demand": 10 + index % 5, "sku": product}
            for product in ("A", "B")
            for index, date in enumerate(dates)
        ]
    )
    app = AppTest.from_file(str(page), default_timeout=45).run()
    app.file_uploader[0].upload(
        "products.csv",
        source.to_csv(index=False).encode(),
        "text/csv",
    ).run()
    app.button[0].click().run()
    app.button[1].click().run()

    def selectbox(label):
        return next(element for element in app.selectbox if element.label == label)

    selectbox("Model for the selected product").select("Naive").run()
    choices = app.session_state["user_demand_forecasting_model_selections"]
    assert choices[product_selection_key("A")] == "Naive"

    selectbox("Series to inspect").select(1).run()
    selectbox("Model for the selected product").select("Moving Average").run()
    choices = app.session_state["user_demand_forecasting_model_selections"]
    assert choices[product_selection_key("B")] == "Moving Average"

    selectbox("Series to inspect").select(0).run()
    assert selectbox("Model for the selected product").value == "Naive"
    assert not app.exception


def test_ineligible_saved_model_is_not_replaced_by_recommendation():
    page = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "pages"
        / "08_user_demand_forecasting.py"
    )
    source = pd.DataFrame(
        {
            "period": pd.date_range("2020-01-01", periods=36, freq="MS"),
            "demand": [10 + index % 5 for index in range(36)],
        }
    )
    app = AppTest.from_file(str(page)).run()
    app.file_uploader[0].upload(
        "single.csv",
        source.to_csv(index=False).encode(),
        "text/csv",
    ).run()
    app.button[0].click().run()
    app.button[1].click().run()
    selection_key = product_selection_key(None)
    app.session_state["user_demand_forecasting_model_selections"][
        selection_key
    ] = "Seasonal Naive"
    app.run()

    model_widget = next(
        element
        for element in app.selectbox
        if element.label == "Model for the selected product"
    )
    generate_button = next(
        element
        for element in app.button
        if element.label == "Generate future forecasts"
    )
    assert model_widget.value == "Choose a model"
    assert any("no longer eligible" in item.value for item in app.warning)
    assert generate_button.disabled
    assert (
        app.session_state["user_demand_forecasting_model_selections"][
            selection_key
        ]
        == "Seasonal Naive"
    )
    assert not app.exception


def test_preparation_does_not_mutate_uploaded_dataframe():
    source = pd.DataFrame(
        {
            "period": ["2024-01-01", "2024-02-01"],
            "units": ["3", "4"],
            "sku": ["A", "A"],
        }
    )
    before = source.copy(deep=True)
    prepare_user_upload(
        source,
        date_column="period",
        demand_column="units",
        product_column="sku",
        frequency="monthly",
    )
    pd.testing.assert_frame_equal(source, before)


def test_invalid_dates_demand_and_negative_values_are_not_discarded():
    source = pd.DataFrame(
        {
            "period": ["not-a-date", "2024-02-01", "2024-03-01"],
            "units": ["1", "unknown", "-2"],
        }
    )
    with pytest.raises(UserDemandForecastingError) as error:
        prepare_user_upload(
            source,
            date_column="period",
            demand_column="units",
            product_column=None,
            frequency="monthly",
        )
    assert len(error.value.invalid_rows) == 3
    assert error.value.invalid_rows["_validation_errors"].str.len().gt(0).all()


def test_missing_required_values_are_reported_with_invalid_rows():
    source = parse_user_demand_csv(
        "missing.csv",
        b"period,units,product\n,,   \n",
    )
    with pytest.raises(UserDemandForecastingError) as error:
        prepare_user_upload(
            source,
            date_column="period",
            demand_column="units",
            product_column="product",
            frequency="monthly",
        )
    messages = error.value.invalid_rows["_validation_errors"].iloc[0]
    assert "date is missing or invalid" in messages
    assert "demand is missing or non-numeric" in messages
    assert "product identifier is missing" in messages


def test_rolling_comparison_recommends_only_after_required_coverage():
    source = pd.DataFrame(
        {
            "period": pd.date_range("2020-01-01", periods=24, freq="MS"),
            "units": [10 + (index % 5) for index in range(24)],
        }
    )
    prepared = prepare_user_upload(
        source,
        date_column="period",
        demand_column="units",
        product_column=None,
        frequency="monthly",
    )
    backtest = compare_user_demand_models(
        prepared.demand,
        prepared.diagnostics,
        horizon=2,
        min_train_periods=8,
        min_successful_origins=2,
        min_comparable_origins=2,
    )
    selection = backtest.selections[0]
    assert selection.selected_model is not None
    assert len(selection.comparable_origins) >= 2
    assert any(summary.model_name == "Naive" for summary in backtest.model_summaries)
    assert all(
        summary.comparable_metrics is None
        or summary.comparable_origins >= 2
        for summary in backtest.model_summaries
    )

    forecasts, warnings = generate_user_forecasts(
        prepared.demand,
        prepared.diagnostics,
        backtest,
        horizon=2,
        selected_models={None: selection.selected_model},
    )
    assert len(forecasts) == 2
    assert forecasts[0].forecast_horizon == 1
    assert forecasts[0].forecast_period == pd.Timestamp("2022-01-01")
    assert warnings.empty

    forecast_export = build_forecast_export(forecasts, backtest)
    assert list(forecast_export.columns) == [
        "product_id",
        "period",
        "predicted_demand",
        "model_name",
        "forecast_horizon",
        "frequency",
        "demand_pattern",
        "selection_metric",
        "selection_rationale",
    ]
    evaluation_export = build_evaluation_export(backtest)
    assert {
        "product_id",
        "model_name",
        "mae",
        "rmse",
        "wape",
        "actual_total",
        "forecast_total",
        "successful_origins",
        "failed_origins",
        "comparable_origins",
    }.issubset(evaluation_export.columns)


def test_no_model_is_recommended_when_origin_requirements_cannot_be_met():
    source = pd.DataFrame(
        {
            "period": pd.date_range("2020-01-01", periods=16, freq="MS"),
            "units": range(1, 17),
        }
    )
    prepared = prepare_user_upload(
        source,
        date_column="period",
        demand_column="units",
        product_column=None,
        frequency="monthly",
    )
    backtest = compare_user_demand_models(
        prepared.demand,
        prepared.diagnostics,
        horizon=1,
        min_train_periods=8,
        min_successful_origins=50,
        min_comparable_origins=2,
    )
    assert backtest.selections[0].selected_model is None
    assert "No model reached" in backtest.selections[0].rationale
    forecasts, warnings = generate_user_forecasts(
        prepared.demand,
        prepared.diagnostics,
        backtest,
        horizon=1,
        selected_models={None: None},
    )
    assert forecasts == ()
    assert warnings.iloc[0]["warning"].startswith("No model met")


def test_skipped_unknown_target_windows_are_included_in_evaluation_export():
    source = pd.DataFrame(
        {
            "period": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-05-01",
                    "2020-06-01",
                    "2020-07-01",
                    "2020-08-01",
                    "2020-09-01",
                ]
            ),
            "units": range(1, 9),
        }
    )
    prepared = prepare_user_upload(
        source,
        date_column="period",
        demand_column="units",
        product_column=None,
        frequency="monthly",
    )
    backtest = compare_user_demand_models(
        prepared.demand,
        prepared.diagnostics,
        horizon=1,
        min_train_periods=2,
        min_successful_origins=1,
        min_comparable_origins=1,
    )

    export = build_evaluation_export(backtest)
    skipped = export.loc[export["record_type"].eq("skipped_window")]
    assert len(skipped) == len(backtest.skipped_origins) == 1
    assert "unknown periods" in skipped.iloc[0]["skipped_reason"]
    assert skipped.iloc[0]["evaluation_periods"] == "2020-04-01T00:00:00"
    assert pd.isna(skipped.iloc[0]["model_name"])
    csv_contents = export.to_csv(index=False)
    assert "skipped_window" in csv_contents
    assert "unknown periods" in csv_contents


def test_streamlit_page_displays_skipped_window_count_and_reason():
    page = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "pages"
        / "08_user_demand_forecasting.py"
    )
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    source = pd.DataFrame(
        {
            "period": dates.delete(10),
            "demand": list(range(10)) + list(range(11, 24)),
        }
    )
    app = AppTest.from_file(str(page)).run()
    app.file_uploader[0].upload(
        "with-gap.csv",
        source.to_csv(index=False).encode(),
        "text/csv",
    ).run()
    app.button[0].click().run()
    app.button[1].click().run()

    assert any("Skipped evaluation windows:" in item.value for item in app.caption)
    assert any("target periods were unknown" in item.value for item in app.warning)
    skipped_table = app.dataframe[-1].value
    assert skipped_table["skipped_reason"].str.contains(
        "unknown periods"
    ).all()
    assert not app.exception


def test_streamlit_page_applies_explicit_expected_coverage_bounds():
    page = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "pages"
        / "08_user_demand_forecasting.py"
    )
    source = pd.DataFrame(
        {
            "period": ["2024-02-01", "2024-04-01"],
            "demand": [2, 4],
        }
    )
    app = AppTest.from_file(str(page)).run()
    app.file_uploader[0].upload(
        "bounded.csv",
        source.to_csv(index=False).encode(),
        "text/csv",
    ).run()
    next(
        item
        for item in app.checkbox
        if item.label.startswith("Set expected coverage")
    ).check()
    app.date_input[0].set_value(date(2024, 1, 1))
    app.date_input[1].set_value(date(2024, 5, 1))
    app.button[0].click().run()

    prepared = app.session_state["user_demand_forecasting_prepared"].demand
    assert prepared.data["observation_status"].tolist() == [
        "unknown_unobserved_period",
        "observed_demand",
        "unknown_unobserved_period",
        "observed_demand",
        "unknown_unobserved_period",
    ]
    assert not app.exception


def test_multi_product_workflow_withholds_forecast_for_unknown_tail():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    source = pd.DataFrame(
        [
            {"period": date, "units": 10 + index, "sku": "short-history"}
            for index, date in enumerate(dates[:18])
        ]
        + [
            {"period": date, "units": 20 + index, "sku": "complete-history"}
            for index, date in enumerate(dates)
        ]
    )
    prepared = prepare_user_upload(
        source,
        date_column="period",
        demand_column="units",
        product_column="sku",
        frequency="monthly",
        expected_start=dates[0],
        expected_end=dates[-1],
        products_expected_throughout_coverage=True,
    )
    backtest = compare_user_demand_models(
        prepared.demand,
        prepared.diagnostics,
        horizon=1,
        min_train_periods=8,
        min_successful_origins=2,
        min_comparable_origins=2,
    )
    selections = {
        selection.product_id: selection.selected_model
        for selection in backtest.selections
    }
    forecasts, warnings = generate_user_forecasts(
        prepared.demand,
        prepared.diagnostics,
        backtest,
        horizon=1,
        selected_models=selections,
    )
    assert {forecast.product_id for forecast in forecasts} == {"complete-history"}
    short_warning = warnings.loc[
        warnings["product_id"].eq("short-history"), "warning"
    ].iloc[0]
    assert "Latest period is unknown" in short_warning


def test_forecast_selection_rejects_unvalidated_models_and_horizon_mismatch():
    source = pd.DataFrame(
        {
            "period": pd.date_range("2020-01-01", periods=18, freq="MS"),
            "units": range(1, 19),
        }
    )
    prepared = prepare_user_upload(
        source,
        date_column="period",
        demand_column="units",
        product_column=None,
        frequency="monthly",
    )
    backtest = compare_user_demand_models(
        prepared.demand,
        prepared.diagnostics,
        horizon=1,
        min_train_periods=8,
    )
    with pytest.raises(UserDemandForecastingError, match="not validated"):
        generate_user_forecasts(
            prepared.demand,
            prepared.diagnostics,
            backtest,
            horizon=1,
            selected_models={None: "Seasonal Naive"},
        )
    selection = backtest.selections[0]
    with pytest.raises(UserDemandForecastingError, match="horizon"):
        generate_user_forecasts(
            prepared.demand,
            prepared.diagnostics,
            backtest,
            horizon=2,
            selected_models={None: selection.selected_model},
        )
