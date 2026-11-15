import pandas as pd
import pytest

from src.analysis import build_monthly_demand


def test_aggregate_monthly_demand_does_not_mutate_raw_input():
    transactions = pd.DataFrame(
        {
            "order_date": ["2017-01-02", "2017-02-01"],
            "product_id": [1, 1],
            "quantity": [2, 3],
        }
    )
    original = transactions.copy(deep=True)

    result = build_monthly_demand.aggregate_monthly_demand(transactions)

    pd.testing.assert_frame_equal(transactions, original)
    assert result["demand_units"].tolist() == [2, 3]
    assert result["month"].tolist() == list(
        pd.to_datetime(["2017-01-01", "2017-02-01"])
    )


@pytest.mark.parametrize(
    ("transactions", "message"),
    [
        (pd.DataFrame(), "Missing raw data columns"),
        (
            pd.DataFrame(
                {
                    "order_date": ["2017-01-01"],
                    "product_id": [1],
                }
            ),
            "Missing raw data columns",
        ),
        (
            pd.DataFrame(
                {
                    "order_date": ["2017-01-01"],
                    "product_id": ["invalid"],
                    "quantity": [2],
                }
            ),
            "product IDs must be numeric",
        ),
        (
            pd.DataFrame(
                {
                    "order_date": ["not-a-date"],
                    "product_id": [1],
                    "quantity": [2],
                }
            ),
            "invalid order dates",
        ),
        (
            pd.DataFrame(
                {
                    "order_date": ["2017-01-01"],
                    "product_id": [1],
                    "quantity": ["not-numeric"],
                }
            ),
            "quantities contain invalid values",
        ),
        (
            pd.DataFrame(
                {
                    "order_date": ["2017-01-01"],
                    "product_id": [1],
                    "quantity": [-1],
                }
            ),
            "quantities must be positive",
        ),
        (
            pd.DataFrame(
                {
                    "order_date": ["2017-01-01", "2017-01-01"],
                    "product_id": [1, 1],
                    "quantity": [2, 2],
                }
            ),
            "duplicate records",
        ),
    ],
)
def test_aggregate_monthly_demand_rejects_invalid_raw_input(
    transactions, message
):
    with pytest.raises(ValueError, match=message):
        build_monthly_demand.aggregate_monthly_demand(transactions)


@pytest.mark.parametrize(
    ("months", "input_is_strings"),
    [
        (["2020-01-01", "2020-02-01"], True),
        (pd.to_datetime(["2020-01-01", "2020-02-01"]), False),
    ],
)
def test_quality_artifacts_accept_date_strings_and_datetime_months(
    months, input_is_strings, tmp_path, monkeypatch
):
    monthly = pd.DataFrame(
        {
            "product_id": [1, 1],
            "month": months,
            "demand_units": [10, 20],
            "transaction_rows": [1, 1],
        }
    )
    original = monthly.copy(deep=True)

    summary, panel, gap, labelled, *_ = build_artifacts(
        monthly, tmp_path, monkeypatch
    )

    if input_is_strings:
        assert original["month"].tolist() == ["2020-01-01", "2020-02-01"]
        assert all(isinstance(value, str) for value in original["month"])
    else:
        assert pd.api.types.is_datetime64_any_dtype(original["month"].dtype)
    pd.testing.assert_frame_equal(monthly, original)
    assert summary["observed_product_months"] == 2
    assert panel.loc[
        panel["record_status"] == "observed", "month"
    ].tolist() == ["2020-01-01", "2020-02-01"]
    assert panel.loc[
        panel["record_status"] == "observed", "demand_units"
    ].tolist() == [10, 20]
    assert gap.empty
    assert labelled["demand_units"].tolist() == [10, 20]


@pytest.mark.parametrize(
    ("months", "message"),
    [
        (
            pd.Series([1577836800000000000], dtype="int64"),
            "date strings or datetime values",
        ),
        (
            pd.to_datetime(["2020-01-01 12:00"]),
            "month starts",
        ),
        (
            pd.to_datetime(["2020-01-01T00:00:00+00:00"]),
            "timezone-naive",
        ),
    ],
)
def test_quality_artifacts_reject_unsupported_month_values(
    months, message, tmp_path, monkeypatch
):
    monthly = pd.DataFrame(
        {
            "product_id": [1],
            "month": months,
            "demand_units": [10],
            "transaction_rows": [1],
        }
    )
    quality_dir = tmp_path / "quality"
    forecast_dir = tmp_path / "forecasts"
    monkeypatch.setattr(build_monthly_demand, "QUALITY_OUTPUT_DIR", quality_dir)
    monkeypatch.setattr(
        build_monthly_demand, "FORECAST_OUTPUT_DIR", forecast_dir
    )

    with pytest.raises(ValueError, match=message):
        build_monthly_demand.build_quality_artifacts(monthly)

    assert not quality_dir.exists()
    assert not forecast_dir.exists()


def test_quality_artifacts_accept_valid_transaction_row_counts(
    tmp_path, monkeypatch
):
    monthly = monthly_data(["2020-01-01", "2020-02-01"], [10, 20])
    monthly["transaction_rows"] = [2, 3]

    summary, *_ = build_artifacts(monthly, tmp_path, monkeypatch)

    assert summary["raw_transaction_records"] == 5


def test_quality_summary_records_raw_source_provenance(tmp_path, monkeypatch):
    monthly = monthly_data(["2017-08-01"], [10])
    quality_dir = tmp_path / "quality"
    forecast_dir = tmp_path / "forecasts"
    monkeypatch.setattr(build_monthly_demand, "QUALITY_OUTPUT_DIR", quality_dir)
    monkeypatch.setattr(
        build_monthly_demand, "FORECAST_OUTPUT_DIR", forecast_dir
    )
    metadata = {
        "source_filename": "integrated_supply_chain_until_aug2017.csv",
        "source_row_count": 166773,
        "source_product_count": 87,
        "source_first_order_date": "2015-01-01",
        "source_last_order_date": "2017-08-31",
    }

    build_monthly_demand.build_quality_artifacts(
        monthly,
        source_metadata=metadata,
    )

    summary = pd.read_csv(forecast_dir / "data_quality_summary.csv")
    values = summary.set_index("metric")["value"]
    for metric, expected in metadata.items():
        if isinstance(expected, int):
            assert float(values[metric]) == expected
        else:
            assert values[metric] == expected


@pytest.mark.parametrize(
    ("transaction_rows", "message"),
    [
        (0, "positive integers"),
        (-1, "positive integers"),
        (1.5, "positive integers"),
        (float("nan"), "null values"),
        (float("inf"), "finite numeric values"),
        ("invalid", "numeric values"),
        (None, "null values"),
    ],
)
def test_invalid_transaction_row_counts_preserve_existing_artifacts(
    transaction_rows, message, tmp_path, monkeypatch
):
    monthly = monthly_data(["2020-01-01"], [10])
    monthly["transaction_rows"] = [transaction_rows]
    quality_dir = tmp_path / "quality"
    forecast_dir = tmp_path / "forecasts"
    quality_dir.mkdir()
    forecast_dir.mkdir()
    sentinels = {
        quality_dir / "monthly_demand_observed_rebuilt.csv": b"rebuilt sentinel",
        quality_dir / "product_month_quality_panel.csv": b"panel sentinel",
        quality_dir / "internal_gaps_imputed_review.csv": b"gap sentinel",
        quality_dir
        / "monthly_demand_observed_plus_labelled_imputation.csv": (
            b"labelled sentinel"
        ),
        forecast_dir / "data_quality_summary.csv": b"summary sentinel",
    }
    for path, content in sentinels.items():
        path.write_bytes(content)
    monkeypatch.setattr(build_monthly_demand, "QUALITY_OUTPUT_DIR", quality_dir)
    monkeypatch.setattr(
        build_monthly_demand, "FORECAST_OUTPUT_DIR", forecast_dir
    )

    with pytest.raises(ValueError, match=message):
        build_monthly_demand.build_quality_artifacts(monthly)

    for path, content in sentinels.items():
        assert path.read_bytes() == content
    assert set(quality_dir.iterdir()) == {
        path for path in sentinels if path.parent == quality_dir
    }
    assert set(forecast_dir.iterdir()) == {
        path for path in sentinels if path.parent == forecast_dir
    }


def test_invalid_transaction_row_count_does_not_create_output_directories(
    tmp_path, monkeypatch
):
    monthly = monthly_data(["2020-01-01"], [10])
    monthly["transaction_rows"] = [0]
    quality_dir = tmp_path / "quality"
    forecast_dir = tmp_path / "forecasts"
    monkeypatch.setattr(build_monthly_demand, "QUALITY_OUTPUT_DIR", quality_dir)
    monkeypatch.setattr(
        build_monthly_demand, "FORECAST_OUTPUT_DIR", forecast_dir
    )

    with pytest.raises(ValueError, match="positive integers"):
        build_monthly_demand.build_quality_artifacts(monthly)

    assert not quality_dir.exists()
    assert not forecast_dir.exists()


def build_artifacts(monthly, tmp_path, monkeypatch):
    original = monthly.copy(deep=True)
    quality_dir = tmp_path / "quality"
    forecast_dir = tmp_path / "forecasts"
    monkeypatch.setattr(build_monthly_demand, "QUALITY_OUTPUT_DIR", quality_dir)
    monkeypatch.setattr(
        build_monthly_demand, "FORECAST_OUTPUT_DIR", forecast_dir
    )

    summary = build_monthly_demand.build_quality_artifacts(monthly)

    pd.testing.assert_frame_equal(monthly, original)
    return (
        summary,
        pd.read_csv(quality_dir / "product_month_quality_panel.csv"),
        pd.read_csv(quality_dir / "internal_gaps_imputed_review.csv"),
        pd.read_csv(
            quality_dir
            / "monthly_demand_observed_plus_labelled_imputation.csv"
        ),
        quality_dir,
        forecast_dir,
    )


def monthly_data(months, demands, *, product_id=1):
    return pd.DataFrame(
        {
            "product_id": [product_id] * len(months),
            "month": pd.to_datetime(months),
            "demand_units": demands,
            "transaction_rows": [1] * len(months),
        }
    )


def test_complete_product_history_has_no_synthetic_gaps(
    tmp_path, monkeypatch
):
    monthly = monthly_data(
        ["2020-01-01", "2020-02-01", "2020-03-01"],
        [10, 20, 30],
    )

    summary, panel, gap, labelled, *_ = build_artifacts(
        monthly, tmp_path, monkeypatch
    )

    assert summary["internal_missing_months"] == 0
    assert panel["record_status"].eq("observed").all()
    assert gap.empty
    assert labelled["record_status"].eq("observed").all()
    assert labelled["demand_units"].tolist() == [10, 20, 30]


@pytest.mark.parametrize(
    ("months", "demands", "expected_missing", "expected_estimates"),
    [
        (
            ["2020-01-01", "2020-03-01"],
            [10, 30],
            ["2020-02-01"],
            [20],
        ),
        (
            ["2020-01-01", "2020-04-01"],
            [10, 40],
            ["2020-02-01", "2020-03-01"],
            [20, 30],
        ),
    ],
)
def test_internal_gaps_are_interpolated_and_labelled(
    months,
    demands,
    expected_missing,
    expected_estimates,
    tmp_path,
    monkeypatch,
):
    monthly = monthly_data(months, demands)

    summary, panel, gap, labelled, *_ = build_artifacts(
        monthly, tmp_path, monkeypatch
    )

    assert summary["internal_missing_months"] == len(expected_missing)
    assert gap["month"].tolist() == expected_missing
    assert gap["demand_units_imputed"].tolist() == expected_estimates
    assert gap["imputation_method"].eq(
        "linear_interpolation_synthetic_not_observed"
    ).all()

    synthetic = labelled.loc[
        labelled["record_status"] == "synthetic_internal_gap"
    ]
    assert synthetic["month"].tolist() == expected_missing
    assert synthetic["demand_units"].isna().all()
    assert synthetic["demand_units_imputed"].tolist() == expected_estimates
    assert synthetic["imputation_method"].eq(
        "linear_interpolation_synthetic_not_observed"
    ).all()

    observed_panel = panel.loc[panel["record_status"] == "observed"]
    assert observed_panel["demand_units"].tolist() == demands
    observed_labelled = labelled.loc[labelled["record_status"] == "observed"]
    assert observed_labelled["demand_units"].tolist() == demands
    assert observed_labelled["demand_units_imputed"].tolist() == demands


def test_lifecycle_boundaries_are_classified_and_never_filled_with_zero(
    tmp_path, monkeypatch
):
    monthly = pd.concat(
        [
            monthly_data(["2020-01-01", "2020-06-01"], [5, 6], product_id=2),
            monthly_data(["2020-03-01", "2020-04-01"], [30, 40]),
        ],
        ignore_index=True,
    )

    _, panel, gap, labelled, *_ = build_artifacts(
        monthly, tmp_path, monkeypatch
    )

    product = panel.loc[panel["product_id"] == 1].set_index("month")
    assert product.loc["2020-01-01", "record_status"] == (
        "before_first_observation_not_assumed_zero"
    )
    assert product.loc["2020-02-01", "record_status"] == (
        "before_first_observation_not_assumed_zero"
    )
    assert product.loc["2020-05-01", "record_status"] == (
        "after_last_observation_not_assumed_zero"
    )
    assert product.loc["2020-06-01", "record_status"] == (
        "after_last_observation_not_assumed_zero"
    )
    boundary = product.loc[
        product["record_status"].isin(
            [
                "before_first_observation_not_assumed_zero",
                "after_last_observation_not_assumed_zero",
            ]
        )
    ]
    assert boundary["demand_units"].isna().all()
    assert boundary["demand_units_imputed"].isna().all()
    assert not boundary["demand_units"].eq(0).any()
    assert not gap["product_id"].eq(1).any()
    product_labelled = labelled.loc[labelled["product_id"] == 1]
    assert not product_labelled["month"].isin(
        ["2020-01-01", "2020-02-01", "2020-05-01", "2020-06-01"]
    ).any()


def test_single_observation_is_not_interpolated(
    tmp_path, monkeypatch
):
    monthly = pd.concat(
        [
            monthly_data(["2020-01-01", "2020-03-01"], [10, 30], product_id=2),
            monthly_data(["2020-02-01"], [20]),
        ],
        ignore_index=True,
    )

    _, panel, gap, labelled, *_ = build_artifacts(
        monthly, tmp_path, monkeypatch
    )

    product = panel.loc[panel["product_id"] == 1]
    assert product.loc[
        product["month"] == "2020-02-01", "record_status"
    ].item() == "observed"
    assert product["demand_units_imputed"].isna().all()
    assert not gap["product_id"].eq(1).any()
    assert labelled.loc[labelled["product_id"] == 1].shape[0] == 1


@pytest.mark.parametrize(
    ("monthly", "message"),
    [
        (
            pd.DataFrame(
                columns=[
                    "month",
                    "product_id",
                    "demand_units",
                    "transaction_rows",
                ]
            ),
            "cannot be empty",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-01"],
                    "product_id": [1],
                    "transaction_rows": [1],
                }
            ),
            "Missing monthly demand columns",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-01"],
                    "product_id": [1],
                    "demand_units": [10],
                }
            ),
            "transaction_rows",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-01", "2020-01-01"],
                    "product_id": [1, 1],
                    "demand_units": [10, 11],
                    "transaction_rows": [1, 1],
                }
            ),
            "Duplicate product-month",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-01"],
                    "product_id": ["invalid"],
                    "demand_units": [10],
                    "transaction_rows": [1],
                }
            ),
            "Product IDs contain invalid values",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["not-a-date"],
                    "product_id": [1],
                    "demand_units": [10],
                    "transaction_rows": [1],
                }
            ),
            "invalid dates",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-02"],
                    "product_id": [1],
                    "demand_units": [10],
                    "transaction_rows": [1],
                }
            ),
            "month starts",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-01 12:00"],
                    "product_id": [1],
                    "demand_units": [10],
                    "transaction_rows": [1],
                }
            ),
            "month starts",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-01"],
                    "product_id": [1],
                    "demand_units": ["not-numeric"],
                    "transaction_rows": [1],
                }
            ),
            "non-finite",
        ),
        (
            pd.DataFrame(
                {
                    "month": ["2020-01-01"],
                    "product_id": [1],
                    "demand_units": [-1],
                    "transaction_rows": [1],
                }
            ),
            "cannot be negative",
        ),
    ],
)
def test_invalid_monthly_input_fails_before_writing_artifacts(
    monthly, message, tmp_path, monkeypatch
):
    quality_dir = tmp_path / "quality"
    forecast_dir = tmp_path / "forecasts"
    monkeypatch.setattr(build_monthly_demand, "QUALITY_OUTPUT_DIR", quality_dir)
    monkeypatch.setattr(
        build_monthly_demand, "FORECAST_OUTPUT_DIR", forecast_dir
    )

    with pytest.raises(ValueError, match=message):
        build_monthly_demand.build_quality_artifacts(monthly)

    assert not quality_dir.exists()
    assert not forecast_dir.exists()


def test_main_invalid_aggregated_demand_preserves_existing_outputs(
    tmp_path, monkeypatch
):
    raw_transactions = pd.DataFrame(
        {
            "order_date": ["2020-01-01", "2020-01-02"],
            "product_id": [1, 1],
            "quantity": [1e308, 1e308],
        }
    )
    output_dir = tmp_path / "processed"
    quality_dir = tmp_path / "quality"
    forecast_dir = tmp_path / "forecasts"
    output_dir.mkdir()
    quality_dir.mkdir()
    forecast_dir.mkdir()

    sentinels = {
        output_dir / "monthly_demand_observed.csv": b"processed sentinel",
        quality_dir / "monthly_demand_observed_rebuilt.csv": b"rebuilt sentinel",
        quality_dir / "product_month_quality_panel.csv": b"panel sentinel",
        quality_dir / "internal_gaps_imputed_review.csv": b"gap sentinel",
        quality_dir
        / "monthly_demand_observed_plus_labelled_imputation.csv": (
            b"labelled sentinel"
        ),
        forecast_dir / "data_quality_summary.csv": b"summary sentinel",
    }
    for path, content in sentinels.items():
        path.write_bytes(content)

    monkeypatch.setattr(
        build_monthly_demand,
        "load_csv",
        lambda _path: raw_transactions.copy(deep=True),
    )
    monkeypatch.setattr(build_monthly_demand, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(build_monthly_demand, "QUALITY_OUTPUT_DIR", quality_dir)
    monkeypatch.setattr(
        build_monthly_demand, "FORECAST_OUTPUT_DIR", forecast_dir
    )

    with pytest.raises(ValueError, match="Demand contains non-finite values"):
        build_monthly_demand.main()

    for path, content in sentinels.items():
        assert path.read_bytes() == content
    assert set(output_dir.iterdir()) == {
        output_dir / "monthly_demand_observed.csv"
    }
    assert set(quality_dir.iterdir()) == {
        path for path in sentinels if path.parent == quality_dir
    }
    assert set(forecast_dir.iterdir()) == {
        forecast_dir / "data_quality_summary.csv"
    }
