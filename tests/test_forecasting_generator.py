import pandas as pd
import pytest

from src.forecasting import generate_future_forecasts as generator


class IncrementingModel:
    n_estimators = 200
    max_depth = 5
    random_state = 42

    def predict(self, features):
        return [float(features["lag_1"].iloc[0]) + 10]


class NegativeModel(IncrementingModel):
    def predict(self, features):
        return [-5.0]


def observed_product(start="2020-01-01", periods=15, *, omit=()):
    months = pd.date_range(start, periods=periods, freq="MS")
    retained = [index for index in range(periods) if index not in omit]
    return pd.DataFrame(
        {
            "product_id": 7,
            "month": months[retained],
            "demand_units": [index + 1 for index in retained],
        }
    )


def test_forecast_recursively_uses_previous_predictions(monkeypatch):
    monkeypatch.setattr(
        generator, "train_random_forest", lambda features, target: IncrementingModel()
    )

    forecasts, warning = generator.forecast_one_product(
        observed_product(), pd.Timestamp("2021-03-01")
    )

    assert warning is None
    assert [row["forecast_demand"] for row in forecasts] == [
        25.0,
        35.0,
        45.0,
    ]
    assert [row["forecast_horizon_month"] for row in forecasts] == [1, 2, 3]
    assert [row["forecast_month"] for row in forecasts] == list(
        pd.date_range("2021-04-01", periods=3, freq="MS")
    )
    assert {row["forecast_method"] for row in forecasts} == {
        "random_forest_recursive"
    }


def test_internal_gap_uses_baseline_without_imputation(monkeypatch):
    monkeypatch.setattr(
        generator,
        "train_random_forest",
        lambda features, target: pytest.fail("RF must not train across a gap"),
    )
    product = observed_product(omit=(7,))

    forecasts, reason = generator.forecast_one_product(
        product, product["month"].max()
    )

    assert len(forecasts) == 3
    assert "internal missing months" in reason
    assert {row["forecast_method"] for row in forecasts} == {
        "moving_average_baseline"
    }
    assert {row["internal_missing_months"] for row in forecasts} == {1}
    assert {row["forecast_confidence"] for row in forecasts} == {"MEDIUM"}


def test_skip_reason_reports_both_short_history_and_internal_gap():
    product = observed_product(periods=10, omit=(7,))

    forecasts, reason = generator.forecast_one_product(
        product, product["month"].max()
    )

    assert len(forecasts) == 3
    assert "internal missing months" in reason
    assert {row["forecast_method"] for row in forecasts} == {
        "moving_average_baseline"
    }


def test_stale_history_within_configured_limit_is_warned(monkeypatch):
    monkeypatch.setattr(
        generator, "train_random_forest", lambda features, target: IncrementingModel()
    )
    product = observed_product()
    cutoff = product["month"].max() + pd.offsets.MonthBegin(1)

    forecasts, warning = generator.forecast_one_product(
        product, cutoff
    )

    assert len(forecasts) == 3
    assert "stale history" in warning.lower()
    assert forecasts[0]["months_since_last_record"] == 1
    assert forecasts[0]["data_quality_status"] == (
        "STALE_HISTORY_LOW_CONFIDENCE"
    )
    assert forecasts[0]["forecast_confidence"] == "LOW"


def test_stale_history_beyond_three_months_is_forecast_with_warning(
    monkeypatch,
):
    monkeypatch.setattr(
        generator, "train_random_forest", lambda features, target: IncrementingModel()
    )
    product = observed_product()
    cutoff = product["month"].max() + pd.offsets.MonthBegin(6)

    forecasts, warning = generator.forecast_one_product(product, cutoff)

    assert len(forecasts) == 3
    assert forecasts[0]["months_since_last_record"] == 6
    assert forecasts[0]["data_quality_status"] == (
        "STALE_HISTORY_LOW_CONFIDENCE"
    )
    assert "Stale history" in warning


def test_insufficient_history_uses_moving_average_baseline():
    product = observed_product(periods=10)

    forecasts, warning = generator.forecast_one_product(
        product, product["month"].max()
    )

    assert len(forecasts) == 3
    assert {row["forecast_method"] for row in forecasts} == {
        "moving_average_baseline"
    }
    assert {row["forecast_confidence"] for row in forecasts} == {"MEDIUM"}
    assert "baseline estimate" in warning


def test_moving_average_uses_only_latest_actual_records():
    product = observed_product(periods=15, omit=(7,))

    forecasts, _ = generator.forecast_one_product(
        product, product["month"].max()
    )

    expected_mean = product["demand_units"].tail(3).mean()
    assert [row["forecast_demand"] for row in forecasts] == [
        expected_mean,
        expected_mean,
        expected_mean,
    ]
    assert {row["internal_missing_months"] for row in forecasts} == {1}


def test_exactly_twelve_observations_use_moving_average(monkeypatch):
    monkeypatch.setattr(
        generator,
        "train_random_forest",
        lambda features, target: pytest.fail("12 months cannot train RF"),
    )
    product = observed_product(periods=12)

    forecasts, _ = generator.forecast_one_product(
        product, product["month"].max()
    )

    assert len(forecasts) == 3
    assert {row["forecast_method"] for row in forecasts} == {
        "moving_average_baseline"
    }


def test_one_observation_uses_low_confidence_naive_fallback():
    product = observed_product(periods=1)

    forecasts, warning = generator.forecast_one_product(
        product, product["month"].max()
    )

    assert [row["forecast_demand"] for row in forecasts] == [1, 1, 1]
    assert {row["forecast_method"] for row in forecasts} == {
        "naive_last_observation_baseline"
    }
    assert {row["forecast_confidence"] for row in forecasts} == {"LOW"}
    assert "baseline estimate" in warning


def test_negative_model_predictions_are_clipped_to_zero(monkeypatch):
    monkeypatch.setattr(
        generator, "train_random_forest", lambda features, target: NegativeModel()
    )

    forecasts, _ = generator.forecast_one_product(
        observed_product(), pd.Timestamp("2021-03-01")
    )

    assert [row["forecast_demand"] for row in forecasts] == [0, 0, 0]


def test_all_observed_products_receive_three_forecasts(monkeypatch, tmp_path):
    products = []
    cutoff = pd.Timestamp("2020-12-01")
    for product_id in range(118):
        periods = 1 if product_id == 0 else (8 if product_id == 1 else 15)
        months = pd.date_range("2019-01-01", periods=periods, freq="MS")
        products.append(
            pd.DataFrame(
                {
                    "product_id": product_id + 1,
                    "month": months,
                    "demand_units": range(1, periods + 1),
                }
            )
        )
    data = pd.concat(products, ignore_index=True)
    monkeypatch.setattr(generator, "load_data", lambda: data)
    monkeypatch.setattr(generator, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        generator,
        "train_random_forest",
        lambda features, target: IncrementingModel(),
    )

    generator.main()

    forecasts = pd.read_csv(tmp_path / "future_demand_forecasts.csv")
    skipped = pd.read_csv(tmp_path / "forecast_products_skipped.csv")
    warnings = pd.read_csv(tmp_path / "forecast_data_warnings.csv")
    summary = pd.read_csv(tmp_path / "forecast_method_summary.csv")
    warning_summary = pd.read_csv(
        tmp_path / "forecast_warning_summary.csv"
    )

    assert forecasts["product_id"].nunique() == 118
    assert len(forecasts) == 354
    assert forecasts.groupby("product_id").size().eq(3).all()
    assert skipped.empty
    assert {"random_forest_recursive", "moving_average_baseline",
            "naive_last_observation_baseline"} <= set(
                forecasts["forecast_method"]
            )
    assert warnings.columns.tolist() == generator.WARNING_COLUMNS
    assert {"warning_category", "warning_message"} <= set(warnings.columns)
    assert warnings["product_id"].is_unique
    assert len(warnings) > 0
    assert set(warning_summary["warning_category"]) == set(
        generator.WARNING_CATEGORIES
    )
    assert forecasts.loc[
        forecasts["months_since_last_record"] > 0,
        "data_quality_status",
    ].eq("STALE_HISTORY_LOW_CONFIDENCE").all()
    assert set(summary["summary_type"]) == {
        "forecast_method",
        "forecast_confidence",
    }
    method_summary = summary.loc[
        summary["summary_type"].eq("forecast_method")
    ]
    confidence_summary = summary.loc[
        summary["summary_type"].eq("forecast_confidence")
    ]
    assert method_summary["product_count"].sum() == 118
    assert set(confidence_summary["category"]) == {"HIGH", "MEDIUM", "LOW"}
    assert confidence_summary["forecast_count"].sum() == 354
    assert forecasts["forecast_demand"].map(pd.notna).all()
    assert (forecasts["forecast_demand"] >= 0).all()


def test_forecast_fields_include_required_contract():
    forecasts, _ = generator.forecast_one_product(
        observed_product(periods=1), pd.Timestamp("2021-03-01")
    )

    required = {
        "product_id",
        "forecast_month",
        "forecast_demand",
        "model_name",
        "forecast_method",
        "forecast_confidence",
        "data_quality_status",
        "first_recorded_month",
        "last_recorded_month",
        "observed_history_months",
        "months_since_last_record",
        "internal_missing_months",
        "forecast_origin_month",
        "forecast_horizon_month",
    }
    assert required <= set(forecasts[0])


def test_structured_warnings_preserve_each_applicable_category():
    forecasts = pd.DataFrame(
        [
            {
                "product_id": 1,
                "forecast_method": "random_forest_recursive",
                "forecast_confidence": "LOW",
                "last_recorded_month": pd.Timestamp("2020-01-01"),
                "months_since_last_record": 5,
                "internal_missing_months": 1,
            },
            {
                "product_id": 1,
                "forecast_method": "random_forest_recursive",
                "forecast_confidence": "LOW",
                "last_recorded_month": pd.Timestamp("2020-01-01"),
                "months_since_last_record": 5,
                "internal_missing_months": 1,
            },
            {
                "product_id": 2,
                "forecast_method": "moving_average_baseline",
                "forecast_confidence": "MEDIUM",
                "last_recorded_month": pd.Timestamp("2020-06-01"),
                "months_since_last_record": 0,
                "internal_missing_months": 0,
            },
            {
                "product_id": 3,
                "forecast_method": "naive_last_observation_baseline",
                "forecast_confidence": "LOW",
                "last_recorded_month": pd.Timestamp("2019-12-01"),
                "months_since_last_record": 6,
                "internal_missing_months": 2,
            },
            {
                "product_id": 4,
                "forecast_method": "random_forest_recursive",
                "forecast_confidence": "MEDIUM",
                "last_recorded_month": pd.Timestamp("2020-06-01"),
                "months_since_last_record": 0,
                "internal_missing_months": 1,
            },
        ]
    )

    warnings = generator.build_warning_records(forecasts)
    summary = generator.build_warning_summary(warnings).set_index(
        "warning_category"
    )

    assert warnings.columns.tolist() == generator.WARNING_COLUMNS
    assert warnings["product_id"].is_unique
    assert summary.loc["STALE_HISTORY", "product_count"] == 2
    assert summary.loc["STALE_HISTORY", "warning_count"] == 2
    assert summary.loc["BASELINE_METHOD", "product_count"] == 1
    assert summary.loc["INTERNAL_MISSING_MONTHS", "product_count"] == 1
    assert set(warnings["warning_category"]) == {
        "STALE_HISTORY",
        "BASELINE_METHOD",
        "INTERNAL_MISSING_MONTHS",
    }
    stale_with_multiple_warnings = warnings.loc[
        warnings["product_id"].eq(3), "warning_message"
    ].iloc[0]
    assert "baseline estimate" in stale_with_multiple_warnings
    assert "internal missing months" in stale_with_multiple_warnings
    assert warnings.loc[
        warnings["warning_category"].eq("BASELINE_METHOD"),
        "warning_message",
    ].str.contains("rather than a Random Forest prediction").all()
