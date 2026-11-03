"""Scenario Simulator — Exploratory multiplier what-if prototype."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.kpi_cards import kpi_row, page_header, under_development_banner
from dashboard.data.loader import load_future_forecasts, load_monthly_demand
from dashboard.styles.theme import get_plotly_layout, get_theme_tokens

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

forecasts = load_future_forecasts()
demand = load_monthly_demand()
t = get_theme_tokens()

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

page_header(
    title="Scenario Simulator",
    subtitle="Interactive prototype for what-if demand scaling and portfolio sensitivity analysis",
    category="OPERATIONS",
    badge_text="Prototype Tool",
    badge_variant="blue",
)

# ---------------------------------------------------------------------------
# Development Notice
# ---------------------------------------------------------------------------

under_development_banner(
    module_title="Stochastic Simulation Engine",
    target_module="src/scenarios/simulation.py",
    description=(
        "The production simulation module will feature parameterized Monte Carlo simulations, "
        "supplier lead-time shock propagation, and econometric elasticity models."
    ),
    disclaimer=(
        "The interface below is a prototype that applies a deterministic multiplier to existing "
        "forecast outputs. It is intended for exploratory sensitivity analysis only and does NOT "
        "represent a validated stochastic risk simulation."
    ),
)

st.divider()

# ---------------------------------------------------------------------------
# Scenario Controls in Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Simulation Parameters")

all_products = sorted(forecasts["product_id"].unique().tolist())
selected_product = st.sidebar.selectbox(
    "Target SKU",
    all_products,
    format_func=lambda pid: f"Product {pid}",
    key="sim_product",
)

demand_multiplier = st.sidebar.slider(
    "Demand Multiplier",
    min_value=0.5,
    max_value=2.5,
    value=1.0,
    step=0.05,
    key="sim_multiplier",
    help="Scale factor applied directly to baseline forecast demand (1.0 = Baseline).",
)

scenario_label_map = {
    (0.5, 0.79): "Contraction Scenario (-20% to -50%)",
    (0.8, 0.99): "Mild Decline (-1% to -20%)",
    (1.0, 1.0): "Baseline Model Output (1.0x)",
    (1.01, 1.25): "Moderate Growth (+1% to +25%)",
    (1.26, 2.5): "Surge / Upside (+26% to +150%)",
}

scenario_name = "Baseline Model Output (1.0x)"
for (lo, hi), label in scenario_label_map.items():
    if lo <= demand_multiplier <= hi:
        scenario_name = label
        break

st.sidebar.markdown(
    f"""
    <div style="padding:0.6rem 0.8rem;background:var(--bg-card);border:1px solid var(--border-color);border-radius:6px;font-size:0.78rem;">
        <span style="color:var(--text-muted);font-weight:600;text-transform:uppercase;">Scenario Classification</span><br>
        <span style="color:var(--text-primary);font-weight:600;">{scenario_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Scenario Calculation & KPIs
# ---------------------------------------------------------------------------

prod_fc = forecasts[forecasts["product_id"] == selected_product].copy()
prod_hist = demand[demand["product_id"] == selected_product].copy()

if not prod_fc.empty:
    base_total = float(prod_fc["forecast_demand"].sum())
    scenario_total = base_total * demand_multiplier
    delta_units = scenario_total - base_total
    delta_pct = (demand_multiplier - 1.0) * 100

    kpi_row([
        {
            "label": "Multiplier Setting",
            "value": f"×{demand_multiplier:.2f}",
            "help": "Direct scaling multiplier selected in the sidebar.",
            "provenance": "User Parameter",
        },
        {
            "label": "Baseline 3M Projection",
            "value": f"{base_total:,.0f} units",
            "help": "Original unscaled Random Forest / Baseline forecast.",
            "provenance": "future_demand_forecasts.csv",
        },
        {
            "label": "Simulated 3M Demand",
            "value": f"{scenario_total:,.0f} units",
            "delta": f"{delta_units:+,.0f} ({delta_pct:+.1f}%)" if delta_units != 0 else "Baseline",
            "delta_color": "normal" if delta_units >= 0 else "inverse",
            "help": "Adjusted forward demand volume under active multiplier.",
            "provenance": "Scaled output",
        },
    ])

    st.divider()

    # ---------------------------------------------------------------------------
    # Scenario Chart
    # ---------------------------------------------------------------------------

    fig = go.Figure()

    # Observed History
    fig.add_trace(
        go.Scatter(
            x=prod_hist["month"],
            y=prod_hist["demand_units"],
            name="Observed Demand",
            mode="lines+markers",
            line=dict(color=t["accent_blue"], width=2),
            marker=dict(size=4),
        )
    )

    # Baseline Forecast
    fig.add_trace(
        go.Scatter(
            x=prod_fc["forecast_month"],
            y=prod_fc["forecast_demand"],
            name="Baseline Forecast",
            mode="lines+markers",
            line=dict(color=t["accent_green"], width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        )
    )

    # Simulated Forecast
    if demand_multiplier != 1.0:
        fig.add_trace(
            go.Scatter(
                x=prod_fc["forecast_month"],
                y=prod_fc["forecast_demand"] * demand_multiplier,
                name=f"Simulated (×{demand_multiplier:.2f})",
                mode="lines+markers",
                line=dict(color=t["accent_orange"], width=2.5, dash="dot"),
                marker=dict(size=7, symbol="circle"),
            )
        )

    # Forecast start vertical cutoff
    if not prod_hist.empty:
        cutoff = prod_hist["month"].max()
        fig.add_vline(
            x=cutoff,
            line=dict(color=t["accent_orange"], width=1, dash="dot"),
            annotation_text="Cutoff",
            annotation_font=dict(color=t["accent_orange"], size=10),
            annotation_position="top right",
        )

    layout = dict(**get_plotly_layout())
    layout["title"] = dict(
        text=f"Product {selected_product} — Scenario Projection Comparison",
        font=dict(size=13, color=t["text_primary"]),
        x=0.01,
    )
    layout["xaxis_title"] = "Month"
    layout["yaxis_title"] = "Units"
    fig.update_layout(**layout)

    st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------------------
    # Month-by-Month Scenario Table
    # ---------------------------------------------------------------------------

    col_m, col_p = st.columns([1.2, 1], gap="medium")

    with col_m:
        st.markdown("### Monthly Projection Impact")
        breakdown = prod_fc[["forecast_month", "forecast_demand", "forecast_confidence"]].copy()
        breakdown["scenario_demand"] = (breakdown["forecast_demand"] * demand_multiplier).round(1)
        breakdown["delta"] = (breakdown["scenario_demand"] - breakdown["forecast_demand"]).round(1)
        breakdown["forecast_month"] = breakdown["forecast_month"].dt.strftime("%b %Y")
        breakdown.columns = [
            "Month",
            "Base Forecast",
            "Confidence",
            "Simulated Demand",
            "Variance (Δ)",
        ]

        st.dataframe(
            breakdown.style.format({
                "Base Forecast": "{:,.1f}",
                "Simulated Demand": "{:,.1f}",
                "Variance (Δ)": "{:+,.1f}",
            }),
            width="stretch",
            hide_index=True,
        )

    with col_p:
        st.markdown("### Portfolio Aggregate Impact")
        agg_base = forecasts.groupby("forecast_month")["forecast_demand"].sum().reset_index()
        agg_base["scenario_demand"] = agg_base["forecast_demand"] * demand_multiplier
        agg_base["delta"] = agg_base["scenario_demand"] - agg_base["forecast_demand"]
        agg_base["forecast_month"] = agg_base["forecast_month"].dt.strftime("%b %Y")
        agg_base.columns = ["Month", "Portfolio Base", "Portfolio Simulated", "Variance (Δ)"]

        st.dataframe(
            agg_base.style.format({
                "Portfolio Base": "{:,.0f}",
                "Portfolio Simulated": "{:,.0f}",
                "Variance (Δ)": "{:+,.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

else:
    st.error(f"No forecast records available for Product {selected_product}.")
