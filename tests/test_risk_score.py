import pandas as pd
import pytest

from src.risk.risk_score import (
    build_demand_risk_assessment,
    summarize_risk_indicators,
)


def _inputs():
    months = pd.date_range("2016-01-01", periods=12, freq="MS")
    demand = pd.DataFrame(
        [
            {"product_id": 1, "month": month, "demand_units": 10 + index}
            for index, month in enumerate(months)
        ]
        + [
            {"product_id": 2, "month": pd.Timestamp("2017-01-01"), "demand_units": 0},
            {"product_id": 2, "month": pd.Timestamp("2017-03-01"), "demand_units": 4},
            {"product_id": 2, "month": pd.Timestamp("2017-08-01"), "demand_units": 8},
        ]
    )
    forecasts = pd.DataFrame(
        {
            "product_id": [1, 2],
            "forecast_month": ["2017-09-01", "2017-09-01"],
            "forecast_demand": [20, 4],
            "forecast_method": [
                "random_forest_recursive",
                "moving_average_baseline",
            ],
            "forecast_origin_month": ["2017-08-01", "2017-08-01"],
            "model_name": ["Random Forest", "Moving Average Baseline"],
        }
    )
    warnings = pd.DataFrame(
        {
            "product_id": [1, 2],
            "warning_category": ["STALE_HISTORY", "BASELINE_METHOD"],
            "warning_message": ["Old history", "Fallback"],
        }
    )
    evaluation = pd.DataFrame(
        [
            {
                "product_id": 1,
                "period": "First period",
                "model": model,
                "mae": mae,
                "rmse": mae,
                "wape_pct": mae * 2,
            }
            for model, mae in [
                ("Random Forest", 2),
                ("Moving Average", 3),
                ("Naive", 4),
            ]
        ]
    )
    return demand, forecasts, warnings, evaluation


def test_risk_assessment_flags_and_evaluation_evidence_are_explainable():
    demand, forecasts, warnings, evaluation = _inputs()

    result = build_demand_risk_assessment(
        demand, forecasts, warnings, evaluation
    ).set_index("product_id")

    assert bool(result.loc[1, "stale_history"])
    assert not bool(result.loc[1, "short_history"])
    assert bool(result.loc[2, "short_history"])
    assert bool(result.loc[2, "baseline_fallback"])
    assert result.loc[2, "internal_missing_months"] == 5
    assert result.loc[1, "baseline_comparison"].startswith(
        "Selected method beat"
    )
    assert result.loc[2, "evaluation_period"] is None or pd.isna(
        result.loc[2, "evaluation_period"]
    )
    assert "Collect additional observed months" in result.loc[
        2, "recommended_action"
    ]
    assert result.loc[2, "demand_coefficient_of_variation"] == pytest.approx(1)


def test_indicator_summary_counts_independent_categories_without_score():
    demand, forecasts, warnings, evaluation = _inputs()
    result = build_demand_risk_assessment(
        demand, forecasts, warnings, evaluation
    )

    summary = summarize_risk_indicators(result).set_index("indicator")

    assert summary.loc["Stale product history", "affected_products"] == 1
    assert summary.loc["Short history (<12 observed months)", "affected_products"] == 1
    assert summary.loc["Baseline forecast method", "affected_products"] == 1
    assert "score" not in " ".join(result.columns).lower()


def test_risk_assessment_rejects_invalid_or_duplicate_demand():
    demand, forecasts, warnings, evaluation = _inputs()
    demand.loc[0, "demand_units"] = -1

    with pytest.raises(ValueError, match="non-negative"):
        build_demand_risk_assessment(demand, forecasts, warnings, evaluation)

    demand, forecasts, warnings, evaluation = _inputs()
    demand = pd.concat([demand, demand.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_demand_risk_assessment(demand, forecasts, warnings, evaluation)


def test_risk_assessment_handles_empty_evaluation_with_explicit_unavailable():
    demand, forecasts, warnings, _ = _inputs()
    empty = pd.DataFrame(
        columns=["product_id", "period", "model", "mae", "rmse", "wape_pct"]
    )

    result = build_demand_risk_assessment(
        demand, forecasts, warnings, empty
    )

    assert result["evaluation_period"].isna().all()
    assert result["baseline_comparison"].eq("Unavailable").all()
