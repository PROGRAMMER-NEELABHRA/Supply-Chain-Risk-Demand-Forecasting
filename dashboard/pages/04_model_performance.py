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
selected_metric_label = {
    "mae": "MAE (units)",
    "rmse": "RMSE (units)",
    "wape_pct": "WAPE (%)",
}.get(selected_metric, selected_metric.upper())

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


def _is_available(value) -> bool:
    if pd.isna(value):
        return False
    try:
        return float(value) == float(value) and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return True


def _get(period_df: pd.DataFrame, model: str, metric: str) -> float | None:
    row = period_df[period_df["model"] == model]
    if row.empty:
        return None
    value = row[metric].iloc[0]
    return round(float(value), 2) if _is_available(value) else None


def _replace_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    display = df.copy()
    for column in columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "Unavailable" if not _is_available(value) else value
            )
    return display


def _format_metric(value, decimals: int) -> str:
    if isinstance(value, str):
        return value
    if not _is_available(value):
        return "Unavailable"
    return f"{float(value):.{decimals}f}"


rf_mae = _get(first_period, "Random Forest", "mae")
ma_mae = _get(first_period, "Moving Average", "mae")
naive_mae = _get(first_period, "Naive", "mae")
rf_wape = _get(first_period, "Random Forest", "wape_pct")

rf_mae_kpi = {
    "label": "RF Mean Absolute Error",
    "value": "Unavailable" if rf_mae is None else f"{rf_mae:.2f}",
    "help": "Mean Absolute Error on validation test set.",
    "provenance": "forecast_error_summary.csv",
}
if rf_mae is not None and naive_mae is not None:
    rf_mae_kpi["delta"] = f"{rf_mae - naive_mae:+.2f} vs Naive"
    rf_mae_kpi["delta_color"] = "inverse" if (rf_mae - naive_mae) > 0 else "normal"

kpi_row([
    rf_mae_kpi,
    {
        "label": "RF WAPE (Weighted Error)",
        "value": "Unavailable" if rf_wape is None else f"{rf_wape:.1f}%",
        "help": "Weighted Absolute Percentage Error on validation test set.",
        "provenance": "forecast_error_summary.csv",
    },
    {
        "label": "Moving Average MAE",
        "value": "Unavailable" if ma_mae is None else f"{ma_mae:.2f}",
        "help": "Baseline rolling 3-month average benchmark MAE.",
        "provenance": "forecast_error_summary.csv",
    },
    {
        "label": "Naive Baseline MAE",
        "value": "Unavailable" if naive_mae is None else f"{naive_mae:.2f}",
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
        comparison_results = error_summary[
            error_summary[selected_metric].map(_is_available)
        ] if selected_metric in error_summary.columns else pd.DataFrame()
        if comparison_results.empty:
            st.info(
                f"No valid {selected_metric_label} evaluation results are available "
                "for the benchmark comparison.",
                icon="ℹ️",
            )
        else:
            fig_comp = model_comparison_bar(comparison_results, metric=selected_metric)
            st.plotly_chart(
                fig_comp,
                width="stretch",
                alt=(
                    f"Comparison of Random Forest, Moving Average, and Naive "
                    f"forecast errors using {selected_metric_label}. The horizontal axis "
                    "is evaluation horizon and the vertical axis is the selected "
                    "error metric; grouped bars compare models across periods."
                ),
            )

    with col_b:
        st.markdown("### Benchmark Summary Table")
        display_cols = ["period", "model", "mae", "rmse", "wape_pct"]
        disp = _replace_missing(
            error_summary[display_cols],
            ["mae", "rmse", "wape_pct"],
        )
        disp.columns = ["Evaluation Horizon", "Model", "MAE", "RMSE", "WAPE (%)"]
        st.dataframe(
            disp.style.format({
                "MAE": lambda value: _format_metric(value, 2),
                "RMSE": lambda value: _format_metric(value, 2),
                "WAPE (%)": lambda value: _format_metric(value, 1),
            }),
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
        valid_feature_importance = feature_importance[
            feature_importance["importance_pct"].map(_is_available)
        ] if "importance_pct" in feature_importance.columns else pd.DataFrame()
        if valid_feature_importance.empty:
            st.info(
                "No valid Random Forest feature-importance results are available.",
                icon="ℹ️",
            )
        else:
            fig_feat = feature_importance_bar(valid_feature_importance)
            st.plotly_chart(
                fig_feat,
                width="stretch",
                alt=(
                    "Random Forest feature importance percentages. The vertical "
                    "axis lists features and the horizontal axis shows relative "
                    "importance; longer bars indicate features with greater model "
                    "importance."
                ),
            )

    with col_b:
        st.markdown("### Attribution Weights")
        display_feature_importance = _replace_missing(
            feature_importance,
            ["importance", "importance_pct"],
        )
        st.dataframe(
            display_feature_importance.style.format({
                "importance": lambda value: _format_metric(value, 4),
                "importance_pct": lambda value: (
                    f"{_format_metric(value, 1)}%"
                    if _is_available(value)
                    else "Unavailable"
                ),
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
        model_results = first_period_product[
            (first_period_product["model"] == selected_model)
            & (first_period_product[selected_metric].map(_is_available))
        ] if selected_metric in first_period_product.columns else pd.DataFrame()
        if model_results.empty:
            st.info(
                f"No valid {selected_metric_label} results are available for "
                f"{selected_model} in the first evaluation period.",
                icon="ℹ️",
            )
        else:
            fig_scatter = per_product_error_scatter(
                model_results, metric=selected_metric, model=selected_model
            )
            st.plotly_chart(
                fig_scatter,
                width="stretch",
                alt=(
                    f"Per-product {selected_metric_label} for {selected_model} in the "
                    "first evaluation period. The horizontal axis is product ID, "
                    "the vertical axis is the selected error metric, and marker "
                    "color also indicates its magnitude."
                ),
            )

    with col_b:
        st.markdown(f"### {selected_model} — Error Distribution")
        model_mae_results = first_period_product[
            (first_period_product["model"] == selected_model)
            & (first_period_product["mae"].map(_is_available))
        ] if "mae" in first_period_product.columns else pd.DataFrame()
        if model_mae_results.empty:
            st.info(
                f"No valid MAE distribution results are available for "
                f"{selected_model} in the first evaluation period.",
                icon="ℹ️",
            )
        else:
            fig_hist = error_distribution_histogram(
                model_mae_results, model=selected_model
            )
            st.plotly_chart(
                fig_hist,
                width="stretch",
                alt=(
                    f"Distribution of per-product MAE values for {selected_model} "
                    "in the first evaluation period. The horizontal axis is MAE "
                    "and the vertical axis is the number of products in each bin."
                ),
            )

    with st.expander("Per-Product Metric Lookup (First Period)", expanded=False):
        disp_prod = first_period_product.copy()
        disp_prod.columns = [c.replace("_", " ").title() for c in disp_prod.columns]
        if disp_prod.empty:
            st.info("No per-product first-period evaluation records are available.", icon="ℹ️")
        else:
            disp_prod = _replace_missing(disp_prod, ["Mae", "Rmse", "Wape Pct"])
            st.dataframe(
                disp_prod.style.format({
                    "Mae": lambda value: _format_metric(value, 2),
                    "Rmse": lambda value: _format_metric(value, 2),
                    "Wape Pct": lambda value: _format_metric(value, 1),
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
        if later_summary.empty:
            st.info("No later-period model summary records are available.", icon="ℹ️")
        else:
            disp_later = later_summary.copy()
            numeric_later_cols = [
                column for column in disp_later.columns
                if pd.api.types.is_numeric_dtype(disp_later[column])
            ]
            disp_later = _replace_missing(disp_later, numeric_later_cols)
            disp_later.columns = [
                c.replace("_", " ").title() for c in disp_later.columns
            ]
            renamed_numeric_cols = [
                c.replace("_", " ").title() for c in numeric_later_cols
            ]
            st.dataframe(
                disp_later.style.format({
                    col: lambda value: _format_metric(value, 2)
                    for col in renamed_numeric_cols
                }),
                width="stretch",
                hide_index=True,
            )

    with col_b:
        st.markdown("### Per-Product Metric Summary (Later Period)")
        valid_later_product = later_per_product[
            pd.Series(
                [
                    any(_is_available(row[column]) for column in ("mae", "rmse", "wape_pct")
                        if column in later_per_product.columns)
                    for _, row in later_per_product.iterrows()
                ],
                index=later_per_product.index,
            )
        ] if not later_per_product.empty else pd.DataFrame()
        if not valid_later_product.empty:
            disp_later_product = _replace_missing(
                valid_later_product,
                ["mae", "rmse", "wape_pct"],
            )
            st.dataframe(
                disp_later_product.style.format({
                    "mae": lambda value: _format_metric(value, 2),
                    "rmse": lambda value: _format_metric(value, 2),
                    "wape_pct": lambda value: _format_metric(value, 1),
                }),
                width="stretch",
                height=300,
                hide_index=True,
            )
        else:
            st.markdown("No per-product later-period records available.")
