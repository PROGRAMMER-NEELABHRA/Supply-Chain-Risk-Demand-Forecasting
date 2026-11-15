"""Model Performance — Benchmark evaluation, error distributions, and feature importance."""

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    error_distribution_histogram,
    feature_importance_bar,
    model_comparison_bar,
    per_product_error_scatter,
)
from dashboard.components.kpi_cards import kpi_row, page_header
from dashboard.components.sidebar import metric_selector, model_filter
from dashboard.data.loader import (
    load_forecast_error_by_product,
    load_forecast_error_summary,
    load_later_period_model_summary,
    load_later_period_per_product,
    load_rf_feature_importance,
    load_rf_model_comparison,
    load_rf_per_product_metrics,
    load_monthly_demand,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

error_summary = load_forecast_error_summary()
error_by_product = load_forecast_error_by_product()
rf_metrics = load_rf_per_product_metrics()
feature_importance = load_rf_feature_importance()
rf_comparison = load_rf_model_comparison()
later_summary = load_later_period_model_summary()
later_per_product = load_later_period_per_product()
demand = load_monthly_demand()
test_end = demand["month"].max()
test_start = pd.Timestamp("2017-05-01")
later_product_count = int(later_per_product["product_id"].nunique())
portfolio_product_count = int(demand["product_id"].nunique())

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Benchmark Controls")

selected_metric = metric_selector(key="mp_metric")
selected_model = model_filter(key="mp_model")

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    title="Model Performance",
    subtitle="Comparative benchmark evaluation across Random Forest, Moving Average, and Naive baselines",
    category="DEMAND INTELLIGENCE",
    badge_text="Validated Metrics",
    badge_variant="blue",
)

# ---------------------------------------------------------------------------
# Primary Accuracy KPIs (First Period Benchmarks)
# ---------------------------------------------------------------------------

first_period = error_summary[error_summary["period"] == "First period"]


def _get(period_df: pd.DataFrame, model: str, metric: str) -> float:
    row = period_df[period_df["model"] == model]
    return round(float(row[metric].values[0]), 2) if not row.empty else float("nan")


rf_mae = _get(first_period, "Random Forest", "mae")
ma_mae = _get(first_period, "Moving Average", "mae")
naive_mae = _get(first_period, "Naive", "mae")
rf_wape = _get(first_period, "Random Forest", "wape_pct")

kpi_row([
    {
        "label": "RF Mean Absolute Error",
        "value": f"{rf_mae:.2f}",
        "delta": f"{rf_mae - naive_mae:+.2f} vs Naive",
        "delta_color": "inverse" if (rf_mae - naive_mae) > 0 else "normal",
        "help": "Mean Absolute Error on validation test set.",
        "provenance": "forecast_error_summary.csv",
    },
    {
        "label": "RF WAPE (Weighted Error)",
        "value": f"{rf_wape:.1f}%",
        "help": "Weighted Absolute Percentage Error on validation test set.",
        "provenance": "forecast_error_summary.csv",
    },
    {
        "label": "Moving Average MAE",
        "value": f"{ma_mae:.2f}",
        "help": "Baseline rolling 3-month average benchmark MAE.",
        "provenance": "forecast_error_summary.csv",
    },
    {
        "label": "Naive Baseline MAE",
        "value": f"{naive_mae:.2f}",
        "help": "Naive last-observation benchmark MAE.",
        "provenance": "forecast_error_summary.csv",
    },
])

st.divider()

# ---------------------------------------------------------------------------
# Tabs for Analytical Views
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Benchmark Comparison",
    "Feature Importance",
    "Per-Product Error Distribution",
    f"Rolling Evaluation ({test_start:%b}–{test_end:%b %Y})",
])

# ── Tab 1: Benchmark comparison ─────────────────────────────────────────────

with tab1:
    col_a, col_b = st.columns([1.6, 1], gap="medium")

    with col_a:
        fig_comp = model_comparison_bar(error_summary, metric=selected_metric)
        st.plotly_chart(fig_comp, width="stretch")

    with col_b:
        st.markdown("### Benchmark Summary Table")
        display_cols = ["period", "model", "mae", "rmse", "wape_pct"]
        disp = error_summary[display_cols].copy()
        disp.columns = ["Evaluation Horizon", "Model", "MAE", "RMSE", "WAPE (%)"]
        st.dataframe(
            disp.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "WAPE (%)": "{:.1f}"}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("<div style='margin-bottom:0.75rem;'></div>", unsafe_allow_html=True)

        if not rf_comparison.empty:
            rf_row = rf_comparison[rf_comparison["model"] == "Random Forest"]
            if not rf_row.empty:
                r = rf_row.iloc[0]
                st.markdown(
                    f"""
                    <div class="ent-card">
                        <div class="ent-card-header">
                            <span>Random Forest Model Configuration</span>
                        </div>
                        <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.6;">
                            <strong style="color:var(--text-primary);">Estimators:</strong> {int(r['random_forest_n_estimators'])} trees<br>
                            <strong style="color:var(--text-primary);">Maximum Depth:</strong> {int(r['random_forest_max_depth'])}<br>
                            <strong style="color:var(--text-primary);">Forecast Strategy:</strong> {r['forecast_strategy']}<br>
                            <strong style="color:var(--text-primary);">Training Period:</strong> {r['training_start']} → {r['training_end']}<br>
                            <strong style="color:var(--text-primary);">Validation Period:</strong> {r['test_start']} → {r['test_end']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ── Tab 2: Feature importance ────────────────────────────────────────────────

with tab2:
    col_a, col_b = st.columns([1.6, 1], gap="medium")

    with col_a:
        fig_feat = feature_importance_bar(feature_importance)
        st.plotly_chart(fig_feat, width="stretch")

    with col_b:
        st.markdown("### Attribution Weights")
        st.dataframe(
            feature_importance.style.format({
                "importance": "{:.4f}",
                "importance_pct": "{:.1f}%",
            }),
            width="stretch",
            hide_index=True,
        )
        st.info(
            "Feature weights reflect Gini impurity reductions in the Random Forest ensemble. "
            "`lag_1` (immediate preceding month) and rolling quarterly averages dominate the prediction signal.",
            icon="ℹ️",
        )

# ── Tab 3: Per-product errors ────────────────────────────────────────────────

with tab3:
    col_a, col_b = st.columns([1.6, 1], gap="medium")

    with col_a:
        first_period_product = error_by_product[error_by_product["period"] == "First period"]
        fig_scatter = per_product_error_scatter(
            first_period_product, metric=selected_metric, model=selected_model
        )
        st.plotly_chart(fig_scatter, width="stretch")

    with col_b:
        st.markdown(f"### {selected_model} — Error Distribution")
        fig_hist = error_distribution_histogram(first_period_product, model=selected_model)
        st.plotly_chart(fig_hist, width="stretch")

    with st.expander("Per-Product Metric Lookup (First Period)", expanded=False):
        disp_prod = first_period_product.copy()
        disp_prod.columns = [c.replace("_", " ").title() for c in disp_prod.columns]
        st.dataframe(
            disp_prod.style.format({
                "Mae": "{:.2f}",
                "Rmse": "{:.2f}",
                "Wape Pct": "{:.1f}",
            }),
            width="stretch",
            height=360,
            hide_index=True,
        )

# ── Tab 4: Later period regime ───────────────────────────────────────────────

with tab4:
    st.info(
        f"One-step-ahead rolling evaluation: {test_start:%b %Y}–"
        f"{test_end:%b %Y}. The Random Forest was trained through Apr 2017; "
        "actual demand from completed test months is available at each "
        f"subsequent forecast step. The complete-history cohort includes "
        f"{later_product_count} of {portfolio_product_count} products, so "
        "these results are not portfolio-wide; no "
        "observations after the source cutoff are included."
    )

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown("### Later-Period Model Summary")
        disp_later = later_summary.copy()
        disp_later.columns = [c.replace("_", " ").title() for c in disp_later.columns]
        float_cols = [c for c in disp_later.columns if disp_later[c].dtype == "float64"]
        st.dataframe(
            disp_later.style.format({col: "{:.2f}" for col in float_cols}),
            width="stretch",
            hide_index=True,
        )

    with col_b:
        st.markdown("### Per-Product Metric Summary (Later Period)")
        if not later_per_product.empty:
            st.dataframe(
                later_per_product.style.format({
                    "mae": "{:.2f}",
                    "rmse": "{:.2f}",
                    "wape_pct": "{:.1f}",
                }),
                width="stretch",
                height=300,
                hide_index=True,
            )
        else:
            st.markdown("No per-product later-period records available.")
