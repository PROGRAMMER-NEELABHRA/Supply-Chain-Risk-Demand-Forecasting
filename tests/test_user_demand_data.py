import pandas as pd
import pytest

from src.data.user_demand import (
    DemandDataValidationError,
    prepare_user_demand_data,
)


def prepare(dataframe, **kwargs):
    defaults = {
        "date_column": "date",
        "demand_column": "demand",
        "frequency": "monthly",
    }
    defaults.update(kwargs)
    return prepare_user_demand_data(dataframe, **defaults)


def test_monthly_single_product_materializes_unknown_period():
    source = pd.DataFrame(
        {"date": ["2024-01-01", "2024-03-01"], "demand": [10, 30]}
    )

    result = prepare(source)

    assert result.data["period"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
    )
    assert result.data["demand_quantity"].iloc[[0, 2]].tolist() == [10, 30]
    assert pd.isna(result.data["demand_quantity"].iloc[1])
    assert result.data["observation_status"].tolist() == [
        "observed_demand",
        "unknown_unobserved_period",
        "observed_demand",
    ]
    assert result.quality_report.missing_periods == 1
    assert result.quality_report.number_of_products == 1


def test_monthly_multi_product_preserves_string_and_numeric_identifiers():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "demand": [2, 4],
            "sku": pd.Series(["001", 7], dtype=object),
        }
    )

    result = prepare(source, product_column="sku")

    assert set(result.data["product_id"]) == {"001", 7}
    assert result.quality_report.number_of_products == 2
    assert result.quality_report.missing_periods == 0


def test_daily_input_is_normalized_to_daily_periods():
    source = pd.DataFrame(
        {"date": ["2024-04-01", "2024-04-02"], "demand": [1, 2]}
    )

    result = prepare(source, frequency="daily")

    assert result.data["period"].tolist() == list(
        pd.to_datetime(["2024-04-01", "2024-04-02"])
    )
    assert result.quality_report.frequency == "daily"


def test_weekly_input_uses_monday_period_start():
    source = pd.DataFrame(
        {"date": ["2024-04-01", "2024-04-15"], "demand": [3, 5]}
    )

    result = prepare(source, frequency="weekly")

    assert result.data["period"].tolist() == list(
        pd.to_datetime(["2024-04-01", "2024-04-08", "2024-04-15"])
    )
    assert result.data["observation_status"].tolist() == [
        "observed_demand",
        "unknown_unobserved_period",
        "observed_demand",
    ]


def test_custom_column_mapping_is_supported():
    source = pd.DataFrame(
        {"when": ["2024-01-01"], "units_sold": [8], "item": ["A-01"]}
    )

    result = prepare(
        source,
        date_column="when",
        demand_column="units_sold",
        product_column="item",
    )

    assert result.data.loc[0, "product_id"] == "A-01"
    assert result.data.loc[0, "demand_quantity"] == 8


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (
            pd.DataFrame({"date": [None], "demand": [1]}),
            "date is missing or invalid",
        ),
        (
            pd.DataFrame({"date": ["not-a-date"], "demand": [1]}),
            "date is missing or invalid",
        ),
        (
            pd.DataFrame({"date": ["2024-01-01"], "demand": [None]}),
            "demand is missing or non-numeric",
        ),
        (
            pd.DataFrame({"date": ["2024-01-01"], "demand": ["many"]}),
            "demand is missing or non-numeric",
        ),
        (
            pd.DataFrame({"date": ["2024-01-01"], "demand": [-1]}),
            "demand cannot be negative",
        ),
        (
            pd.DataFrame({"date": ["2024-01-01"], "demand": [float("inf")]}),
            "demand must be finite",
        ),
        (
            pd.DataFrame({"date": ["2024-01-01"], "demand": [float("-inf")]}),
            "demand must be finite",
        ),
        (
            pd.DataFrame({"date": ["2024-01-01"], "demand": [float("nan")]}),
            "demand is missing or non-numeric",
        ),
    ],
)
def test_invalid_date_or_demand_rows_are_reported_not_dropped(source, error):
    with pytest.raises(DemandDataValidationError) as raised:
        prepare(source)

    assert raised.value.quality_report.invalid_rows == 1
    assert error in raised.value.invalid_rows.loc[0, "_validation_errors"]


def test_duplicate_product_period_is_rejected():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "demand": [1, 2],
            "sku": ["A", "A"],
        }
    )

    with pytest.raises(DemandDataValidationError) as raised:
        prepare(source, product_column="sku")

    assert raised.value.quality_report.duplicate_rows == 1
    assert raised.value.quality_report.invalid_rows == 1
    assert "duplicate date/product period" in raised.value.invalid_rows.iloc[0][
        "_validation_errors"
    ]


def test_missing_periods_remain_unknown_by_default():
    source = pd.DataFrame(
        {"date": ["2024-01-01", "2024-03-01"], "demand": [1, 3]}
    )

    result = prepare(source)

    missing = result.data.loc[
        result.data["observation_status"] == "unknown_unobserved_period"
    ]
    assert len(missing) == 1
    assert pd.isna(missing.iloc[0]["demand_quantity"])
    assert "no demand values were imputed" in result.quality_report.validation_warnings[-1]


def test_missing_periods_become_zero_only_with_both_confirmations():
    source = pd.DataFrame(
        {"date": ["2024-01-01", "2024-03-01"], "demand": [1, 3]}
    )

    result = prepare(
        source,
        complete_source_coverage=True,
        confirm_missing_periods_are_zero=True,
    )

    missing = result.data.loc[result.data["period"] == pd.Timestamp("2024-02-01")]
    assert missing.iloc[0]["demand_quantity"] == 0
    assert missing.iloc[0]["observation_status"] == "confirmed_zero_demand"
    assert result.quality_report.missing_periods == 1


def test_zero_confirmation_requires_complete_coverage():
    source = pd.DataFrame(
        {"date": ["2024-01-01"], "demand": [1]}
    )

    with pytest.raises(ValueError, match="complete source coverage"):
        prepare(source, confirm_missing_periods_are_zero=True)


def test_complete_coverage_without_zero_confirmation_keeps_period_unknown():
    source = pd.DataFrame(
        {"date": ["2024-01-01", "2024-03-01"], "demand": [1, 3]}
    )

    result = prepare(source, complete_source_coverage=True)

    missing = result.data.loc[result.data["period"] == pd.Timestamp("2024-02-01")]
    assert pd.isna(missing.iloc[0]["demand_quantity"])
    assert missing.iloc[0]["observation_status"] == "unknown_unobserved_period"


def test_observed_zero_is_labelled_observed_zero_demand():
    source = pd.DataFrame(
        {"date": ["2024-01-01"], "demand": [0]}
    )

    result = prepare(source)

    assert result.data.loc[0, "observation_status"] == "observed_zero_demand"
    assert result.data.loc[0, "demand_quantity"] == 0


def test_input_dataframe_is_not_mutated_and_unsorted_input_is_reported():
    source = pd.DataFrame(
        {"date": ["2024-02-01", "2024-01-01"], "demand": ["2", "1"]}
    )
    original = source.copy(deep=True)

    result = prepare(source)

    pd.testing.assert_frame_equal(source, original)
    assert result.data["period"].is_monotonic_increasing
    assert any("unsorted" in warning for warning in result.quality_report.validation_warnings)


@pytest.mark.parametrize(
    ("frequency", "date"),
    [("monthly", "2024-01-02"), ("weekly", "2024-04-02")],
)
def test_invalid_period_alignment_is_rejected(frequency, date):
    source = pd.DataFrame({"date": [date], "demand": [1]})

    with pytest.raises(DemandDataValidationError, match="invalid rows"):
        prepare(source, frequency=frequency)


def test_missing_columns_and_empty_input_are_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        prepare(pd.DataFrame(columns=["date", "demand"]))
    with pytest.raises(ValueError, match="Missing mapped demand columns"):
        prepare(pd.DataFrame({"date": ["2024-01-01"]}))


def test_missing_product_identifier_is_reported():
    source = pd.DataFrame(
        {"date": ["2024-01-01"], "demand": [1], "sku": [""]}
    )

    with pytest.raises(DemandDataValidationError, match="invalid rows"):
        prepare(source, product_column="sku")


@pytest.mark.parametrize(
    ("frequency", "start", "end", "expected_start", "expected_end"),
    [
        ("daily", "2024-01-02 18:00", "2024-01-04 23:00", "2024-01-02", "2024-01-04"),
        ("weekly", "2024-01-03", "2024-01-17", "2024-01-01", "2024-01-15"),
        ("monthly", "2024-02-20", "2024-04-30", "2024-02-01", "2024-04-01"),
    ],
)
def test_expected_bounds_normalize_to_selected_frequency(
    frequency, start, end, expected_start, expected_end
):
    source_dates = {
        "daily": ["2024-01-02", "2024-01-04"],
        "weekly": ["2024-01-01", "2024-01-15"],
        "monthly": ["2024-02-01", "2024-04-01"],
    }[frequency]
    result = prepare(
        pd.DataFrame({"date": source_dates, "demand": [2, 4]}),
        frequency=frequency,
        expected_start=start,
        expected_end=end,
    )

    assert result.quality_report.expected_coverage_start == pd.Timestamp(expected_start)
    assert result.quality_report.expected_coverage_end == pd.Timestamp(expected_end)
    assert result.data["period"].min() == result.quality_report.expected_coverage_start
    assert result.data["period"].max() == result.quality_report.expected_coverage_end


def test_expected_bounds_classify_missing_before_internal_and_after_periods():
    source = pd.DataFrame(
        {"date": ["2024-02-01", "2024-04-01"], "demand": [2, 4]}
    )

    result = prepare(
        source,
        expected_start="2024-01-01",
        expected_end="2024-05-01",
    )
    coverage = result.coverage_by_product.iloc[0]

    assert result.data["observation_status"].tolist() == [
        "unknown_unobserved_period",
        "observed_demand",
        "unknown_unobserved_period",
        "observed_demand",
        "unknown_unobserved_period",
    ]
    assert (
        coverage["missing_before_first_observation"],
        coverage["missing_internal_periods"],
        coverage["missing_after_last_observation"],
    ) == (1, 1, 1)
    assert (
        result.quality_report.missing_before_first_observation,
        result.quality_report.missing_internal_periods,
        result.quality_report.missing_after_last_observation,
    ) == (1, 1, 1)


def test_expected_bounds_fill_missing_periods_only_with_both_confirmations():
    source = pd.DataFrame(
        {"date": ["2024-02-01", "2024-04-01"], "demand": [2, 4]}
    )
    result = prepare(
        source,
        expected_start="2024-01-01",
        expected_end="2024-05-01",
        complete_source_coverage=True,
        confirm_missing_periods_are_zero=True,
    )

    assert result.data["demand_quantity"].tolist() == [0, 2, 0, 4, 0]
    assert result.data["observation_status"].tolist() == [
        "confirmed_zero_demand",
        "observed_demand",
        "confirmed_zero_demand",
        "observed_demand",
        "confirmed_zero_demand",
    ]


def test_product_lifecycle_bounds_do_not_create_artificial_prelaunch_history():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01"],
            "demand": [1, 2, 3, 4],
            "sku": ["A", "A", "B", "B"],
        }
    )
    result = prepare(
        source,
        product_column="sku",
        expected_start="2024-01-01",
        expected_end="2024-04-01",
    )

    product_b = result.data.loc[result.data["product_id"].eq("B")]
    coverage_b = result.coverage_by_product.set_index("product_id").loc["B"]
    assert product_b["period"].tolist() == list(
        pd.to_datetime(["2024-02-01", "2024-03-01"])
    )
    assert coverage_b["excluded_before_first_observation_periods"] == 1
    assert coverage_b["excluded_after_last_observation_periods"] == 1
    assert coverage_b["missing_before_first_observation"] == 0
    assert coverage_b["missing_after_last_observation"] == 0


def test_product_wide_expected_window_materializes_boundary_unknowns():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-02-01"],
            "demand": [1, 2, 3],
            "sku": ["A", "A", "B"],
        }
    )
    result = prepare(
        source,
        product_column="sku",
        expected_start="2024-01-01",
        expected_end="2024-03-01",
        products_expected_throughout_coverage=True,
    )

    product_b = result.data.loc[result.data["product_id"].eq("B")]
    assert product_b["observation_status"].tolist() == [
        "unknown_unobserved_period",
        "observed_demand",
        "unknown_unobserved_period",
    ]
    assert product_b["demand_quantity"].isna().tolist() == [True, False, True]

    confirmed = prepare(
        source,
        product_column="sku",
        expected_start="2024-01-01",
        expected_end="2024-03-01",
        products_expected_throughout_coverage=True,
        complete_source_coverage=True,
        confirm_missing_periods_are_zero=True,
    )
    confirmed_b = confirmed.data.loc[confirmed.data["product_id"].eq("B")]
    assert confirmed_b["observation_status"].tolist() == [
        "confirmed_zero_demand",
        "observed_demand",
        "confirmed_zero_demand",
    ]


def test_invalid_expected_bounds_and_observations_outside_window_are_rejected():
    source = pd.DataFrame(
        {"date": ["2024-01-01", "2024-03-01"], "demand": [1, 3]}
    )
    with pytest.raises(ValueError, match="must not be after"):
        prepare(source, expected_start="2024-04-01", expected_end="2024-02-01")
    with pytest.raises(ValueError, match="fall outside"):
        prepare(source, expected_start="2024-02-01", expected_end="2024-03-01")
    with pytest.raises(ValueError, match="valid dates"):
        prepare(source, expected_start="not-a-date", expected_end="2024-03-01")


def test_product_wide_coverage_requires_explicit_bounds():
    source = pd.DataFrame(
        {"date": ["2024-01-01"], "demand": [1], "sku": ["A"]}
    )
    with pytest.raises(ValueError, match="both expected coverage bounds"):
        prepare(
            source,
            product_column="sku",
            products_expected_throughout_coverage=True,
        )


def test_omitting_expected_bounds_remains_supported_without_prelaunch_rows():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-02-01"],
            "demand": [1, 2, 3],
            "sku": ["A", "A", "B"],
        }
    )
    result = prepare(source, product_column="sku")

    product_b = result.data.loc[result.data["product_id"].eq("B")]
    assert product_b["period"].tolist() == [pd.Timestamp("2024-02-01")]
    assert result.quality_report.expected_coverage_start == pd.Timestamp("2024-01-01")
    assert result.quality_report.expected_coverage_end == pd.Timestamp("2024-02-01")
