"""
Centralised data loader for the Supply Chain Dashboard.

All paths are resolved relative to the project root so the dashboard
can be launched from any working directory.  Every public function uses
@st.cache_data so that CSV files are read from disk only once per
Streamlit session.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# dashboard/data/loader.py → go up two levels to reach the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FORECAST_DIR = _PROJECT_ROOT / "outputs" / "forecasts"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def _read(relative_path: str) -> pd.DataFrame:
    """Read a CSV from the forecast outputs directory."""
    return pd.read_csv(_FORECAST_DIR / relative_path)


# ---------------------------------------------------------------------------
# Public loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_monthly_demand() -> pd.DataFrame:
    """Monthly observed demand: month, product_id, demand_units."""
    df = pd.read_csv(
        _PROCESSED_DIR / "monthly_demand_observed.csv",
        parse_dates=["month"],
    )
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values(["product_id", "month"]).reset_index(drop=True)


@st.cache_data
def load_future_forecasts() -> pd.DataFrame:
    """3-month forward demand forecasts for all products."""
    df = _read("future_demand_forecasts.csv")
    df["forecast_month"] = pd.to_datetime(df["forecast_month"])
    return df.sort_values(["product_id", "forecast_month"]).reset_index(drop=True)


@st.cache_data
def load_forecast_error_summary() -> pd.DataFrame:
    """Aggregate model error metrics (MAE/RMSE/WAPE) for both periods."""
    return _read("forecast_error_summary.csv")


@st.cache_data
def load_forecast_error_by_product() -> pd.DataFrame:
    """Per-product model error metrics for all three models."""
    return _read("forecast_error_by_product.csv")


@st.cache_data
def load_rf_per_product_metrics() -> pd.DataFrame:
    """Random-Forest per-product metrics vs baselines."""
    return _read("random_forest_per_product_metrics.csv")


@st.cache_data
def load_rf_feature_importance() -> pd.DataFrame:
    """Random Forest feature importances."""
    return _read("random_forest_feature_importance.csv")


@st.cache_data
def load_rf_model_comparison() -> pd.DataFrame:
    """Overall RF model comparison with hyperparameters."""
    return _read("random_forest_model_comparison.csv")


@st.cache_data
def load_common_cutoff_comparison() -> pd.DataFrame:
    """Per-product common-cutoff comparison (Naive vs Moving Average)."""
    return _read("common_cutoff_model_comparison.csv")


@st.cache_data
def load_demand_shift_analysis() -> pd.DataFrame:
    """Train vs test demand shift analysis per product."""
    return _read("demand_shift_analysis.csv")


@st.cache_data
def load_forecast_method_summary() -> pd.DataFrame:
    """Breakdown of forecast methods and confidence levels."""
    return _read("forecast_method_summary.csv")


@st.cache_data
def load_forecast_warning_summary() -> pd.DataFrame:
    """Structured warning counts by category."""
    return _read("forecast_warning_summary.csv")


@st.cache_data
def load_forecast_eligibility() -> pd.DataFrame:
    """Product eligibility status for forecasting."""
    return _read("forecast_eligibility.csv")


@st.cache_data
def load_data_quality_summary() -> pd.DataFrame:
    """Dataset-level quality metrics."""
    return _read("data_quality_summary.csv")


@st.cache_data
def load_forecast_data_warnings() -> pd.DataFrame:
    """Per-product structured data-quality warnings."""
    return _read("forecast_data_warnings.csv")


@st.cache_data
def load_multi_product_comparison() -> pd.DataFrame:
    """Multi-product continuous-history model comparison."""
    return _read("multi_product_model_comparison.csv")


@st.cache_data
def load_later_period_model_summary() -> pd.DataFrame:
    """Later-period RF depth-comparison summary."""
    return _read("later_period_error_model_summary.csv")


@st.cache_data
def load_later_period_per_product() -> pd.DataFrame:
    """Later-period per-product metrics."""
    return _read("later_period_per_product_metrics.csv")


@st.cache_data
def load_skipped_products() -> pd.DataFrame:
    """Products that were skipped during forecasting."""
    return _read("skipped_products.csv")
