import pandas as pd

from src.forecasting import forecasting_improvement_experiment_v2 as experiment


def observed_series(product_id, start, periods, *, omit=(), values=None):
    months = pd.date_range(start, periods=periods, freq="MS")
    retained = [index for index in range(periods) if index not in omit]
    demand = values or [float(index + product_id) for index in range(periods)]
    return pd.DataFrame(
        {
            "month": months[retained],
            "product_id": product_id,
            "demand_units": [demand[index] for index in retained],
        }
    )


def test_origin_eligibility_requires_13_contiguous_observed_months():
    origin = pd.Timestamp("2021-01-01")
    data = pd.concat(
        [
            observed_series(1, "2020-01-01", 13),
            observed_series(2, "2020-01-01", 13, omit=(6,)),
            observed_series(3, "2021-02-01", 15),
        ],
        ignore_index=True,
    )

    eligible = experiment.eligible_histories(data, origin)

    assert set(eligible) == {1}
    assert len(eligible[1]) == 13


def test_future_rows_cannot_change_earlier_eligibility_or_forecasts():
    origin = pd.Timestamp("2021-01-01")
    past = pd.concat(
        [
            observed_series(1, "2020-01-01", 13),
            observed_series(2, "2020-02-01", 12),
        ],
        ignore_index=True,
    )
    future = pd.concat(
        [
            observed_series(1, "2021-02-01", 3, values=[5000, 6000, 7000]),
            observed_series(2, "2021-02-01", 3, values=[8000, 9000, 10000]),
            observed_series(3, "2021-02-01", 3, values=[11, 12, 13]),
        ],
        ignore_index=True,
    )

    before = experiment.predict_at_origin(past, origin)
    after = experiment.predict_at_origin(
        pd.concat([past, future], ignore_index=True), origin
    )
    prediction_columns = [
        "method",
        "origin_month",
        "history_months",
        "product_id",
        "horizon",
        "target_month",
        "forecast",
    ]

    pd.testing.assert_frame_equal(
        before[prediction_columns]
        .sort_values(prediction_columns[:-1])
        .reset_index(drop=True),
        after[prediction_columns]
        .sort_values(prediction_columns[:-1])
        .reset_index(drop=True),
    )
    assert set(experiment.eligible_histories(past, origin)) == set(
        experiment.eligible_histories(
            pd.concat([past, future], ignore_index=True), origin
        )
    )


def test_missing_future_month_stays_unknown_and_is_not_scored_as_zero():
    origin = pd.Timestamp("2021-01-01")
    history = observed_series(1, "2020-01-01", 13)
    future = pd.DataFrame(
        {
            "month": pd.to_datetime(["2021-02-01", "2021-04-01"]),
            "product_id": [1, 1],
            "demand_units": [20.0, 25.0],
        }
    )

    result = experiment.predict_at_origin(
        pd.concat([history, future], ignore_index=True), origin
    )
    missing_month = result.loc[
        result["target_month"].eq(pd.Timestamp("2021-03-01"))
    ]

    assert len(missing_month) == len(experiment.METHODS)
    assert not missing_month["actual_observed"].any()
    assert missing_month["actual"].isna().all()
    assert missing_month["forecast"].notna().all()


def test_all_methods_share_identical_origin_cohort_and_actuals():
    origin = pd.Timestamp("2021-01-01")
    history = pd.concat(
        [
            observed_series(1, "2020-01-01", 13),
            observed_series(2, "2020-01-01", 13),
        ],
        ignore_index=True,
    )
    future = pd.concat(
        [
            observed_series(1, "2021-02-01", 3),
            observed_series(2, "2021-02-01", 3),
        ],
        ignore_index=True,
    )
    result = experiment.predict_at_origin(
        pd.concat([history, future], ignore_index=True), origin
    )

    reference = (
        result.loc[result["method"].eq(experiment.METHODS[0])]
        .sort_values(["product_id", "horizon"])
        .reset_index(drop=True)
    )
    for method in experiment.METHODS[1:]:
        compared = (
            result.loc[result["method"].eq(method)]
            .sort_values(["product_id", "horizon"])
            .reset_index(drop=True)
        )
        assert compared[
            ["product_id", "horizon", "target_month", "actual"]
        ].equals(
            reference[["product_id", "horizon", "target_month", "actual"]]
        )
        assert compared["eligible_products"].eq(2).all()


def test_oracle_is_separate_and_marked_non_deployable(monkeypatch):
    origin = pd.Timestamp("2021-01-01")
    monkeypatch.setattr(
        experiment,
        "STABLE_END",
        pd.Timestamp("2021-04-01"),
    )
    monkeypatch.setattr(
        experiment,
        "paired_block_bootstrap",
        lambda predictions: (0.0, 0.0, 0.0),
    )
    data = pd.concat(
        [
            observed_series(1, "2020-01-01", 13),
            observed_series(1, "2021-02-01", 3),
        ],
        ignore_index=True,
    )
    result = experiment.predict_at_origin(data, origin, include_oracle=True)
    oracle = result.loc[result["method"].eq(experiment.ORACLE_METHOD)]

    assert len(oracle) == experiment.HORIZON
    assert oracle["method_status"].eq(
        "non_deployable_oracle_uses_future_data"
    ).all()
    summary = experiment.build_summary(result)
    assert experiment.ORACLE_METHOD not in set(summary["method"])
