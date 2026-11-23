import os

import pandas as pd

from dashboard.data import loader
from dashboard.styles import theme
from src.analysis import analyze_demand_shift
from src.forecasting import evaluate_later_period


def _write_monthly_demand(path, end="2017-08-01"):
    months = pd.date_range("2015-01-01", end, freq="MS")
    pd.DataFrame(
        {
            "month": months,
            "product_id": 1,
            "demand_units": range(1, len(months) + 1),
        }
    ).to_csv(path, index=False)


def test_later_period_evaluation_ends_at_last_available_month(
    tmp_path, monkeypatch
):
    source = tmp_path / "monthly_demand.csv"
    _write_monthly_demand(source)
    monkeypatch.setattr(evaluate_later_period, "DATA_FILE", source)

    monthly = evaluate_later_period.load_data()
    eligible = evaluate_later_period.select_eligible_products(monthly)

    assert monthly["month"].max() == pd.Timestamp("2017-08-01")
    assert eligible["month"].max() == pd.Timestamp("2017-08-01")
    assert evaluate_later_period.TEST_START == pd.Timestamp("2017-05-01")


def test_demand_shift_analysis_uses_available_cutoff(tmp_path, monkeypatch):
    source = tmp_path / "monthly_demand.csv"
    _write_monthly_demand(source)
    monkeypatch.setattr(analyze_demand_shift, "DATA_FILE", source)

    monthly = analyze_demand_shift.load_data()

    assert monthly["month"].max() == pd.Timestamp("2017-08-01")
    assert monthly["month"].max() < pd.Timestamp("2017-09-01")


def test_dashboard_loader_reads_current_outputs_and_invalidates_cache(
    tmp_path, monkeypatch
):
    loader._read_csv_cached.clear()
    monkeypatch.setattr(loader, "_FORECAST_DIR", tmp_path)
    monkeypatch.setattr(loader, "_PROCESSED_DIR", tmp_path)

    monthly_path = tmp_path / "monthly_demand_observed.csv"
    pd.DataFrame(
        {
            "month": ["2017-08-01"],
            "product_id": [1],
            "demand_units": [10],
        }
    ).to_csv(monthly_path, index=False)
    assert loader.load_monthly_demand()["demand_units"].iloc[0] == 10

    old_stat = monthly_path.stat()
    pd.DataFrame(
        {
            "month": ["2017-08-01"],
            "product_id": [1],
            "demand_units": [20],
        }
    ).to_csv(monthly_path, index=False)
    os.utime(
        monthly_path,
        ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000),
    )
    assert loader.load_monthly_demand()["demand_units"].iloc[0] == 20

    pd.DataFrame({"model": ["New evaluation"], "mae": [1.0]}).to_csv(
        tmp_path / "later_period_model_comparison.csv",
        index=False,
    )
    pd.DataFrame({"model": ["Stale evaluation"], "mae": [99.0]}).to_csv(
        tmp_path / "later_period_error_model_summary.csv",
        index=False,
    )
    assert (
        loader.load_later_period_model_summary()["model"].iloc[0]
        == "New evaluation"
    )

    loader._read_csv_cached.clear()


def test_theme_tokens_fall_back_when_streamlit_has_no_active_theme(monkeypatch):
    monkeypatch.setattr(theme, "get_current_theme", lambda: None)

    assert theme.get_theme_tokens()["name"] == "light"
    assert theme.get_theme_tokens("dark")["name"] == "dark"
