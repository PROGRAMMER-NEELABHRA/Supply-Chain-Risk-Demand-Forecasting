import pandas as pd
import pytest

from src.data.user_demand import (
    PreparedDemandData,
    prepare_user_demand_data,
)
from src.preprocessing.user_demand_preparation import (
    aggregate_transaction_demand,
    prepare_forecast_demand,
)


def prepare_periods(
    dataframe,
    *,
    frequency="monthly",
    product_column=None,
    complete_source_coverage=False,
    confirm_missing_periods_are_zero=False,
):
    prepared = prepare_user_demand_data(
        dataframe,
        date_column="date",
        demand_column="demand",
        product_column=product_column,
        frequency=frequency,
        complete_source_coverage=complete_source_coverage,
        confirm_missing_periods_are_zero=confirm_missing_periods_are_zero,
    )
    return prepared


def aggregate(dataframe, *, frequency, product_column=None):
    return aggregate_transaction_demand(
        dataframe,
        date_column="date",
        demand_column="quantity",
        product_column=product_column,
        frequency=frequency,
        aggregation="sum_transactions",
    )


@pytest.mark.parametrize(
    ("frequency", "dates", "expected_periods", "expected_totals"),
    [
        (
            "daily",
            ["2024-01-01 09:00", "2024-01-01 12:00", "2024-01-02"],
            ["2024-01-01", "2024-01-02"],
            [5, 4],
        ),
        (
            "weekly",
            ["2024-01-01", "2024-01-03", "2024-01-08"],
            ["2024-01-01", "2024-01-08"],
            [5, 4],
        ),
        (
            "monthly",
            ["2024-01-01", "2024-01-29", "2024-02-01"],
            ["2024-01-01", "2024-02-01"],
            [5, 4],
        ),
    ],
)
def test_transaction_rows_sum_by_selected_frequency(
    frequency, dates, expected_periods, expected_totals
):
    transactions = pd.DataFrame(
        {"date": dates, "quantity": [2, 3, 4]}
    )

    result = aggregate(transactions, frequency=frequency)

    assert result.data["period"].tolist() == list(pd.to_datetime(expected_periods))
    assert result.data["demand_quantity"].tolist() == expected_totals
    assert result.aggregation_summary.input_rows == 3
    assert result.aggregation_summary.aggregated_period_rows == 2
    assert result.aggregation_summary.method == "sum_transactions"


def test_transaction_aggregation_supports_multiple_mixed_type_products():
    transactions = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "quantity": [2, 3, 7],
            "sku": pd.Series(["001", "001", 7], dtype=object),
        }
    )

    result = aggregate(transactions, frequency="monthly", product_column="sku")

    monthly = result.data.loc[result.data["training_eligible"]]
    assert set(monthly["product_id"]) == {"001", 7}
    assert monthly.set_index("product_id")["demand_quantity"].to_dict() == {
        "001": 5,
        7: 7,
    }
    assert result.coverage_by_product["product_id"].tolist() == ["001", 7]


def test_single_product_period_data_gets_training_and_coverage_metadata():
    prepared = prepare_periods(
        pd.DataFrame(
            {"date": ["2024-01-01", "2024-02-01"], "demand": [2, 0]}
        )
    )

    result = prepare_forecast_demand(prepared)

    assert "product_id" not in result.data.columns
    assert result.data["training_eligible"].tolist() == [True, True]
    assert result.data["observation_status"].tolist() == [
        "observed_demand",
        "observed_zero_demand",
    ]
    assert result.frequency == "monthly"
    assert result.coverage_by_product.iloc[0]["periods_expected"] == 2
    assert result.coverage_by_product.iloc[0]["observed_periods"] == 2


def test_unknown_and_confirmed_zero_periods_remain_distinct():
    source = pd.DataFrame(
        {"date": ["2024-01-01", "2024-03-01"], "demand": [2, 0]}
    )
    unknown = prepare_forecast_demand(prepare_periods(source))
    confirmed = prepare_forecast_demand(
        prepare_periods(
            source,
            complete_source_coverage=True,
            confirm_missing_periods_are_zero=True,
        )
    )

    unknown_february = unknown.data.loc[
        unknown.data["period"] == pd.Timestamp("2024-02-01")
    ].iloc[0]
    confirmed_february = confirmed.data.loc[
        confirmed.data["period"] == pd.Timestamp("2024-02-01")
    ].iloc[0]
    observed_march = confirmed.data.loc[
        confirmed.data["period"] == pd.Timestamp("2024-03-01")
    ].iloc[0]
    assert unknown_february["observation_status"] == "unknown_unobserved_period"
    assert pd.isna(unknown_february["demand_quantity"])
    assert not unknown_february["training_eligible"]
    assert confirmed_february["observation_status"] == "confirmed_zero_demand"
    assert confirmed_february["demand_quantity"] == 0
    assert confirmed_february["training_eligible"]
    assert observed_march["observation_status"] == "observed_zero_demand"
    summary = confirmed.coverage_by_product.iloc[0]
    assert summary["confirmed_zero_periods"] == 1
    assert summary["observed_periods"] == 2
    assert summary["unknown_periods"] == 0


def test_multi_product_coverage_is_reported_per_product():
    source = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-02-01"],
            "demand": [1, 2, 3],
            "sku": ["A", "A", "B"],
        }
    )

    result = prepare_forecast_demand(
        prepare_periods(source, product_column="sku")
    )

    coverage = result.coverage_by_product.set_index("product_id")
    assert coverage.loc["A", "unknown_periods"] == 0
    assert coverage.loc["B", "unknown_periods"] == 0
    assert coverage.loc["B", "coverage_ratio"] == 1.0
    assert coverage.loc["B", "periods_expected"] == 1


def test_transaction_duplicates_are_only_combined_in_explicit_sum_mode():
    transactions = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "quantity": [2, 3],
            "sku": ["A", "A"],
        }
    )

    with pytest.raises(TypeError):
        aggregate_transaction_demand(
            transactions,
            date_column="date",
            demand_column="quantity",
            product_column="sku",
            frequency="monthly",
        )
    result = aggregate(transactions, frequency="monthly", product_column="sku")
    assert result.data.loc[0, "demand_quantity"] == 5


def test_exact_duplicate_transaction_rows_are_reported_but_retained():
    transactions = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-03", "2024-01-18"],
            "quantity": [2, 2, 5],
            "sku": ["A", "A", "A"],
        }
    )

    result = aggregate(
        transactions,
        frequency="monthly",
        product_column="sku",
    )

    assert result.data.loc[0, "demand_quantity"] == 9
    assert result.aggregation_summary.exact_duplicate_rows == 1
    assert any(
        "retained in the sums" in warning
        for warning in result.quality_report.validation_warnings
    )


@pytest.mark.parametrize(
    ("frequency", "dates", "expected_periods"),
    [
        (
            "daily",
            ["2024-02-28 23:30", "2024-02-29 01:00"],
            ["2024-02-28", "2024-02-29"],
        ),
        (
            "weekly",
            ["2024-03-31", "2024-04-01"],
            ["2024-03-25", "2024-04-01"],
        ),
        (
            "monthly",
            ["2024-02-29", "2024-03-01"],
            ["2024-02-01", "2024-03-01"],
        ),
    ],
)
def test_frequency_boundaries_map_to_expected_period_starts(
    frequency, dates, expected_periods
):
    transactions = pd.DataFrame({"date": dates, "quantity": [1, 2]})

    result = aggregate(transactions, frequency=frequency)

    assert result.data["period"].tolist() == list(pd.to_datetime(expected_periods))


def test_no_interpolated_or_synthetic_historical_values_are_created():
    source = pd.DataFrame(
        {"date": ["2024-01-01", "2024-04-01"], "demand": [10, 40]}
    )
    prepared = prepare_forecast_demand(prepare_periods(source))

    gap = prepared.data.loc[
        prepared.data["observation_status"] == "unknown_unobserved_period"
    ]
    assert len(gap) == 2
    assert gap["demand_quantity"].isna().all()
    assert not gap["training_eligible"].any()
    assert prepared.data.loc[
        prepared.data["training_eligible"], "demand_quantity"
    ].tolist() == [10, 40]


def test_preparation_does_not_mutate_phase_one_input():
    prepared = prepare_periods(
        pd.DataFrame(
            {"date": ["2024-02-01", "2024-01-01"], "demand": [2, 1]}
        )
    )
    original = prepared.data.copy(deep=True)

    result = prepare_forecast_demand(prepared)

    pd.testing.assert_frame_equal(prepared.data, original)
    assert result.data["period"].is_monotonic_increasing


def test_duplicate_keys_in_manually_constructed_phase_one_data_are_rejected():
    prepared = prepare_periods(
        pd.DataFrame({"date": ["2024-01-01"], "demand": [1]})
    )
    duplicate = pd.concat([prepared.data, prepared.data], ignore_index=True)
    prepared = PreparedDemandData(
        data=duplicate,
        quality_report=prepared.quality_report,
    )

    with pytest.raises(ValueError, match="duplicate period/product"):
        prepare_forecast_demand(prepared)


def test_transaction_aggregation_rejects_bad_demand_without_coercion():
    transactions = pd.DataFrame(
        {"date": ["2024-01-01"], "quantity": ["not a quantity"]}
    )

    with pytest.raises(ValueError, match="non-numeric"):
        aggregate(transactions, frequency="monthly")


def test_transaction_aggregation_does_not_mutate_input():
    transactions = pd.DataFrame(
        {
            "date": ["2024-01-02 12:00", "2024-01-01 08:00"],
            "quantity": [2, 1],
            "sku": ["A", "A"],
        }
    )
    original = transactions.copy(deep=True)

    aggregate(transactions, frequency="monthly", product_column="sku")

    pd.testing.assert_frame_equal(transactions, original)


def test_preparation_accepts_daily_weekly_and_monthly_contracts():
    cases = [
        ("daily", ["2024-01-01", "2024-01-02"]),
        ("weekly", ["2024-01-01", "2024-01-08"]),
        ("monthly", ["2024-01-01", "2024-02-01"]),
    ]
    for frequency, dates in cases:
        prepared = prepare_periods(
            pd.DataFrame({"date": dates, "demand": [1, 2]}),
            frequency=frequency,
        )
        result = prepare_forecast_demand(prepared)
        assert result.frequency == frequency
        assert result.data["period"].tolist() == list(pd.to_datetime(dates))
