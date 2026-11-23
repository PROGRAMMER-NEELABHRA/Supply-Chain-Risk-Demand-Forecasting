import pandas as pd

from src.analysis import build_monthly_demand
from src.data.loaders import load_monthly_demand


def test_load_monthly_demand_sorts_and_parses_dates(tmp_path):
    source = tmp_path / "monthly_demand.csv"
    pd.DataFrame(
        {
            "month": ["2020-02-01", "2020-01-01"],
            "product_id": [2, 1],
            "demand_units": [20, 10],
        }
    ).to_csv(source, index=False)

    result = load_monthly_demand(source)

    assert result["month"].tolist() == list(
        pd.to_datetime(["2020-01-01", "2020-02-01"])
    )
    assert result["product_id"].tolist() == [1, 2]


def test_raw_transactions_aggregate_reproducibly_by_product_month():
    transactions = pd.DataFrame(
        {
            "order_date": [
                "2017-01-02",
                "2017-01-29",
                "2017-02-01",
            ],
            "product_id": [1, 1, 1],
            "quantity": [2, 3, 5],
            "order_item_id": [11, 12, 13],
            "order_id": [100, 101, 101],
        }
    )

    result = build_monthly_demand.aggregate_monthly_demand(transactions)

    assert result["demand_units"].tolist() == [5, 5]
    assert result["transaction_rows"].tolist() == [2, 1]
    assert result["unique_orders"].tolist() == [2, 1]


def test_quality_panel_classifies_gaps_and_labels_synthetic_values(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        build_monthly_demand,
        "QUALITY_OUTPUT_DIR",
        tmp_path / "quality",
    )
    monkeypatch.setattr(
        build_monthly_demand,
        "FORECAST_OUTPUT_DIR",
        tmp_path / "forecasts",
    )
    monthly = pd.DataFrame(
        {
            "product_id": [1, 1, 2],
            "month": pd.to_datetime(
                ["2020-01-01", "2020-03-01", "2020-02-01"]
            ),
            "demand_units": [10, 30, 20],
            "transaction_rows": [1, 1, 1],
        }
    )

    summary = build_monthly_demand.build_quality_artifacts(monthly)
    panel = pd.read_csv(
        tmp_path / "quality" / "product_month_quality_panel.csv"
    )
    gap = pd.read_csv(
        tmp_path / "quality" / "internal_gaps_imputed_review.csv"
    )
    labelled = pd.read_csv(
        tmp_path
        / "quality"
        / "monthly_demand_observed_plus_labelled_imputation.csv"
    )

    assert summary["expected_full_grid_product_months"] == 6
    assert summary["observed_product_months"] == 3
    assert summary["absent_product_months"] == 3
    assert summary["internal_missing_months"] == 1
    assert summary["products_with_internal_gaps"] == 1
    assert set(panel["record_status"]) >= {
        "observed",
        "before_first_observation_not_assumed_zero",
        "after_last_observation_not_assumed_zero",
        "missing_inside_observed_lifecycle_review_required",
    }

    internal_gap = panel.loc[
        panel["record_status"]
        == "missing_inside_observed_lifecycle_review_required"
    ].iloc[0]
    assert pd.isna(internal_gap["demand_units"])
    assert internal_gap["demand_units_imputed"] == 20
    assert gap.iloc[0]["imputation_method"] == (
        "linear_interpolation_synthetic_not_observed"
    )
    assert labelled.loc[
        labelled["record_status"] == "observed", "demand_units"
    ].notna().all()
    synthetic = labelled.loc[
        labelled["record_status"] == "synthetic_internal_gap"
    ]
    assert synthetic["demand_units"].isna().all()
    assert synthetic["demand_units_imputed"].iloc[0] == 20
