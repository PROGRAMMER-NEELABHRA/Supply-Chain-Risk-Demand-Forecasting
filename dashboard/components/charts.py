"""Reusable Plotly chart builders for the Supply Chain Platform with dynamic Light/Dark theme support."""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.styles.theme import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    CONFIDENCE_COLOURS,
    DANGER_RED,
    MODEL_COLOURS,
    get_plotly_layout,
    get_theme_tokens,
)


def _apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    """Apply the dynamic theme-aware layout to a Plotly figure."""
    t = get_theme_tokens()
    layout = get_plotly_layout()
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(
                size=13,
                color=t["text_primary"],
                family="Inter, -apple-system, sans-serif",
            ),
            x=0.01,
            xanchor="left",
            pad=dict(l=4, t=8),
        )
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Demand charts
# ---------------------------------------------------------------------------


def demand_time_series(
    df: pd.DataFrame,
    product_id: Optional[int] = None,
    title: str = "Monthly Demand Trend",
) -> go.Figure:
    """Line chart of actual demand over time with subtle area fill."""
    t = get_theme_tokens()
    data = df.copy()
    fill_color = "rgba(59, 130, 246, 0.08)" if t["name"] == "dark" else "rgba(37, 99, 235, 0.08)"

    if product_id is not None:
        data = data[data["product_id"] == product_id]
        agg = data.rename(columns={"month": "Month", "demand_units": "Demand"})
        fig = px.line(
            agg,
            x="Month",
            y="Demand",
            color_discrete_sequence=[t["accent_blue"]],
            markers=True,
        )
        fig.update_traces(line=dict(width=2), marker=dict(size=5))
    else:
        agg = (
            data.groupby("month")["demand_units"]
            .sum()
            .reset_index()
            .rename(columns={"month": "Month", "demand_units": "Demand"})
        )
        fig = px.line(
            agg,
            x="Month",
            y="Demand",
            color_discrete_sequence=[t["accent_blue"]],
            markers=True,
        )
        fig.update_traces(
            line=dict(width=2.5),
            fill="tozeroy",
            fillcolor=fill_color,
            marker=dict(size=4),
        )

    return _apply_layout(fig, title)


def demand_forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    product_id: int,
) -> go.Figure:
    """Overlaid actual history + 3-month forecast for a specific product."""
    t = get_theme_tokens()
    hist = history[history["product_id"] == product_id].copy()
    fore = forecast[forecast["product_id"] == product_id].copy()

    fig = go.Figure()

    # Actual history trace
    fig.add_trace(
        go.Scatter(
            x=hist["month"],
            y=hist["demand_units"],
            name="Actual Demand",
            mode="lines+markers",
            line=dict(color=t["accent_blue"], width=2.2),
            marker=dict(size=5),
        )
    )

    # Bridge connection between last observation and forecast
    if not hist.empty and not fore.empty:
        bridge_x = [hist["month"].iloc[-1], fore["forecast_month"].iloc[0]]
        bridge_y = [hist["demand_units"].iloc[-1], fore["forecast_demand"].iloc[0]]
        fig.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines",
                line=dict(color=t["accent_green"], width=1.5, dash="dot"),
                showlegend=False,
            )
        )

    # Forecast trace
    fig.add_trace(
        go.Scatter(
            x=fore["forecast_month"],
            y=fore["forecast_demand"],
            name="Forward Forecast",
            mode="lines+markers",
            line=dict(color=t["accent_green"], width=2.5, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        )
    )

    # Cutoff indicator
    if not hist.empty:
        cutoff = hist["month"].max()
        fig.add_vline(
            x=cutoff,
            line=dict(color=t["accent_orange"], width=1.2, dash="dot"),
            annotation_text="Forecast Cutoff",
            annotation_font=dict(color=t["accent_orange"], size=10),
            annotation_position="top right",
        )

    return _apply_layout(
        fig,
        f"Product {product_id} — Observed History & 3-Month Forecast",
        xaxis_title="Month",
        yaxis_title="Units",
    )


def demand_heatmap(df: pd.DataFrame) -> go.Figure:
    """Product x Month demand matrix heatmap."""
    t = get_theme_tokens()
    pivot = df.pivot_table(
        index="product_id", columns="month", values="demand_units", aggfunc="sum"
    )
    pivot.columns = [str(c)[:7] for c in pivot.columns]

    if t["name"] == "light":
        colorscale = [
            [0, "#F1F5F9"],
            [0.5, "#93C5FD"],
            [1, "#059669"],
        ]
    else:
        colorscale = [
            [0, "rgba(17,30,53,0.5)"],
            [0.5, "rgba(59,130,246,0.6)"],
            [1, "#10B981"],
        ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(tickfont=dict(color=t["text_muted"], size=10)),
        )
    )
    return _apply_layout(
        fig,
        "Demand Heatmap — Product × Month",
        xaxis=dict(tickangle=-45, gridcolor=t["grid_color"]),
        yaxis=dict(gridcolor=t["grid_color"], autorange="reversed"),
        height=480,
    )


# ---------------------------------------------------------------------------
# Model performance charts
# ---------------------------------------------------------------------------


def model_comparison_bar(df: pd.DataFrame, metric: str = "mae") -> go.Figure:
    """Grouped bar chart comparing baseline models vs Random Forest."""
    t = get_theme_tokens()
    label_map = {"mae": "MAE", "rmse": "RMSE", "wape_pct": "WAPE (%)"}
    metric_label = label_map.get(metric, metric.upper())

    model_colors = {
        "Random Forest": t["accent_green"],
        "Moving Average": t["accent_blue"],
        "Naive": t["accent_orange"],
    }

    fig = go.Figure()
    for model, colour in model_colors.items():
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                name=model,
                x=sub["period"],
                y=sub[metric],
                marker_color=colour,
                text=sub[metric].round(1),
                textposition="outside",
                textfont=dict(size=11, color=t["text_primary"]),
            )
        )

    fig.update_layout(barmode="group")
    return _apply_layout(
        fig,
        f"Benchmark Evaluation — {metric_label}",
        yaxis_title=metric_label,
        xaxis_title="Evaluation Horizon",
    )


def feature_importance_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for Random Forest feature importances."""
    t = get_theme_tokens()
    sorted_df = df.sort_values("importance_pct")
    fig = go.Figure(
        go.Bar(
            x=sorted_df["importance_pct"],
            y=sorted_df["feature"],
            orientation="h",
            marker=dict(
                color=sorted_df["importance_pct"],
                colorscale=[[0, t["accent_purple"]], [1, t["accent_green"]]],
                showscale=False,
            ),
            text=sorted_df["importance_pct"].round(1).astype(str) + "%",
            textposition="outside",
            textfont=dict(size=10, color=t["text_primary"]),
        )
    )
    return _apply_layout(
        fig,
        "Random Forest — Feature Importance Distribution",
        xaxis_title="Relative Importance (%)",
        height=340,
    )


def per_product_error_scatter(
    df: pd.DataFrame, metric: str = "mae", model: str = "Random Forest"
) -> go.Figure:
    """Scatter plot of per-product error metric."""
    t = get_theme_tokens()
    label_map = {"mae": "MAE", "rmse": "RMSE", "wape_pct": "WAPE (%)"}
    sub = df[df["model"] == model].copy()
    fig = px.scatter(
        sub,
        x="product_id",
        y=metric,
        color=metric,
        color_continuous_scale=[[0, t["accent_green"]], [0.5, t["accent_orange"]], [1, t["danger_red"]]],
        labels={"product_id": "Product ID", metric: label_map.get(metric, metric)},
        hover_data=["product_id", metric],
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_coloraxes(showscale=False)
    return _apply_layout(
        fig,
        f"Per-Product {label_map.get(metric, metric)} Distribution — {model}",
        xaxis_title="Product ID",
        yaxis_title=label_map.get(metric, metric),
    )


# ---------------------------------------------------------------------------
# Distribution / summary charts
# ---------------------------------------------------------------------------


def confidence_donut(method_df: pd.DataFrame) -> go.Figure:
    """Donut chart of forecast confidence rating distribution."""
    t = get_theme_tokens()
    conf_df = method_df[method_df["summary_type"] == "forecast_confidence"].copy()
    confidence_colors = {
        "HIGH": t["accent_green"],
        "MEDIUM": t["accent_orange"],
        "LOW": t["danger_red"],
    }
    colours = [confidence_colors.get(c, t["accent_blue"]) for c in conf_df["category"]]
    fig = go.Figure(
        go.Pie(
            labels=conf_df["category"],
            values=conf_df["product_count"],
            hole=0.62,
            marker=dict(colors=colours, line=dict(color=t["bg_card"], width=2)),
            textinfo="label+percent",
            textfont=dict(color=t["text_primary"], size=11),
        )
    )
    return _apply_layout(fig, "Forecast Confidence Distribution", showlegend=False)


def method_donut(method_df: pd.DataFrame) -> go.Figure:
    """Donut chart of forecast method assignment."""
    t = get_theme_tokens()
    meth_df = method_df[method_df["summary_type"] == "forecast_method"].copy()
    label_map = {
        "random_forest_recursive": "Random Forest",
        "moving_average_baseline": "Moving Average",
        "naive_last_observation_baseline": "Naive Baseline",
    }
    meth_df["label"] = meth_df["category"].map(label_map).fillna(meth_df["category"])
    colours = [t["accent_green"], t["accent_blue"], t["accent_orange"]]
    fig = go.Figure(
        go.Pie(
            labels=meth_df["label"],
            values=meth_df["product_count"],
            hole=0.62,
            marker=dict(colors=colours, line=dict(color=t["bg_card"], width=2)),
            textinfo="label+percent",
            textfont=dict(color=t["text_primary"], size=11),
        )
    )
    return _apply_layout(fig, "Forecast Method Distribution", showlegend=False)


def warning_bar(warning_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for data quality warning frequencies."""
    t = get_theme_tokens()
    label_map = {
        "STALE_HISTORY": "Stale History",
        "BASELINE_METHOD": "Baseline Method Only",
        "INTERNAL_MISSING_MONTHS": "Internal Gaps in Data",
    }
    colour_map = {
        "STALE_HISTORY": t["danger_red"],
        "BASELINE_METHOD": t["accent_orange"],
        "INTERNAL_MISSING_MONTHS": t["accent_blue"],
    }
    df = warning_df.copy()
    df["label"] = df["warning_category"].map(label_map).fillna(df["warning_category"])
    df["colour"] = df["warning_category"].map(colour_map).fillna(t["accent_blue"])
    fig = go.Figure(
        go.Bar(
            x=df["product_count"],
            y=df["label"],
            orientation="h",
            marker_color=df["colour"].tolist(),
            text=df["product_count"],
            textposition="outside",
            textfont=dict(size=11, color=t["text_primary"]),
        )
    )
    return _apply_layout(
        fig,
        "Data Quality Warnings by Category",
        xaxis_title="Affected Products",
        height=240,
    )


def demand_shift_bar(shift_df: pd.DataFrame) -> go.Figure:
    """Bar chart of demand-shift category counts between train and test periods."""
    t = get_theme_tokens()
    counts = shift_df["demand_shift"].value_counts().reset_index()
    counts.columns = ["shift", "count"]
    fig = px.bar(
        counts,
        x="shift",
        y="count",
        color="count",
        color_continuous_scale=[[0, t["accent_blue"]], [1, t["accent_green"]]],
        labels={"shift": "Demand Shift", "count": "Products"},
        text="count",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color=t["text_primary"]),
    )
    fig.update_coloraxes(showscale=False)
    return _apply_layout(
        fig,
        "Demand Shift Classification (Train vs Test)",
        xaxis_title="",
        yaxis_title="Product Count",
        xaxis_tickangle=-15,
    )


def top_products_bar(
    df: pd.DataFrame,
    n: int = 10,
    title: str = "Top Products by Total Historical Volume",
) -> go.Figure:
    """Horizontal bar chart showing highest-volume products."""
    t = get_theme_tokens()
    top = (
        df.groupby("product_id")["demand_units"]
        .sum()
        .nlargest(n)
        .reset_index()
        .sort_values("demand_units")
    )
    fig = go.Figure(
        go.Bar(
            x=top["demand_units"],
            y=top["product_id"].astype(str),
            orientation="h",
            marker=dict(
                color=top["demand_units"],
                colorscale=[[0, t["accent_blue"]], [1, t["accent_green"]]],
                showscale=False,
            ),
            text=top["demand_units"].apply(lambda v: f"{v:,.0f}"),
            textposition="outside",
            textfont=dict(size=10, color=t["text_primary"]),
        )
    )
    return _apply_layout(
        fig,
        title,
        xaxis_title="Total Units Ordered",
        yaxis_title="Product ID",
        height=380,
    )


def seasonality_line(df: pd.DataFrame) -> go.Figure:
    """Seasonality profile line chart showing average demand across calendar months."""
    t = get_theme_tokens()
    df = df.copy()
    df["month_num"] = df["month"].dt.month
    seasonal = (
        df.groupby("month_num")["demand_units"]
        .mean()
        .reset_index()
    )
    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    seasonal["month_name"] = seasonal["month_num"].map(lambda m: month_names[m])

    fill_color = "rgba(139, 92, 246, 0.08)" if t["name"] == "dark" else "rgba(124, 58, 237, 0.08)"

    fig = go.Figure(
        go.Scatter(
            x=seasonal["month_name"],
            y=seasonal["demand_units"],
            mode="lines+markers",
            line=dict(color=t["accent_purple"], width=2.2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor=fill_color,
        )
    )
    return _apply_layout(
        fig,
        "Portfolio Aggregate Demand Seasonality Profile",
        xaxis_title="Calendar Month",
        yaxis_title="Mean Monthly Units",
    )


def error_distribution_histogram(df: pd.DataFrame, model: str = "Random Forest") -> go.Figure:
    """Histogram of per-product MAE distribution."""
    t = get_theme_tokens()
    sub = df[df["model"] == model]["mae"].dropna()
    fig = px.histogram(
        sub,
        nbins=18,
        color_discrete_sequence=[t["accent_blue"]],
        labels={"value": "MAE"},
    )
    fig.update_layout(showlegend=False)
    return _apply_layout(
        fig,
        f"Per-Product Error Distribution (MAE) — {model}",
        xaxis_title="Mean Absolute Error",
        yaxis_title="Product Count",
    )
