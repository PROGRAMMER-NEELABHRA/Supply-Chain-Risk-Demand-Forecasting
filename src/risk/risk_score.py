"""Explainable demand and forecast data-risk indicators (no composite score)."""

import numpy as np
import pandas as pd
from src.forecasting.user_demand_backtesting import RollingOriginBacktest
from src.forecasting.user_demand_diagnostics import DemandPatternDiagnostics
from src.preprocessing.user_demand_preparation import ForecastReadyDemandData


def build_demand_risk_assessment(
    monthly_demand: pd.DataFrame,
    forecasts: pd.DataFrame,
    warnings: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine observed data-quality flags and available backtest evidence by product."""
    for frame, required, label in [
        (
            monthly_demand,
            {"product_id", "month", "demand_units"},
            "Monthly demand",
        ),
        (
            forecasts,
            {"product_id", "forecast_month", "forecast_demand", "forecast_method"},
            "Forecasts",
        ),
        (
            warnings,
            {"product_id", "warning_category", "warning_message"},
            "Warnings",
        ),
        (
            evaluation_metrics,
            {"product_id", "period", "model", "mae", "rmse", "wape_pct"},
            "Evaluation metrics",
        ),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    demand = monthly_demand.copy()
    demand["month"] = pd.to_datetime(demand["month"], errors="coerce")
    demand["demand_units"] = pd.to_numeric(
        demand["demand_units"], errors="coerce"
    )
    forecast = forecasts.copy()
    forecast["forecast_month"] = pd.to_datetime(
        forecast["forecast_month"], errors="coerce"
    )
    forecast["forecast_demand"] = pd.to_numeric(
        forecast["forecast_demand"], errors="coerce"
    )
    if demand[["month", "product_id", "demand_units"]].isna().any().any():
        raise ValueError("Monthly demand contains invalid required values.")
    if forecast[
        ["product_id", "forecast_month", "forecast_demand", "forecast_method"]
    ].isna().any().any():
        raise ValueError("Forecasts contain invalid required values.")
    if not np.isfinite(demand["demand_units"]).all() or (
        demand["demand_units"] < 0
    ).any():
        raise ValueError("Monthly demand must be finite and non-negative.")
    if not np.isfinite(forecast["forecast_demand"]).all() or (
        forecast["forecast_demand"] < 0
    ).any():
        raise ValueError("Forecast demand must be finite and non-negative.")
    if demand.duplicated(["product_id", "month"]).any():
        raise ValueError("Monthly demand has duplicate product-month rows.")
    if forecast.duplicated(["product_id", "forecast_month"]).any():
        raise ValueError("Forecasts have duplicate product-month rows.")

    warning_lookup = {
        product_id: group for product_id, group in warnings.groupby("product_id")
    }
    evaluation = evaluation_metrics.copy()
    evaluation["mae"] = pd.to_numeric(evaluation["mae"], errors="coerce")
    evaluation["rmse"] = pd.to_numeric(evaluation["rmse"], errors="coerce")
    evaluation["wape_pct"] = pd.to_numeric(
        evaluation["wape_pct"], errors="coerce"
    )
    evaluation_order = {"First period": 0, "Later period": 1}
    records = []

    for product_id, product_forecasts in forecast.groupby("product_id", sort=True):
        history = demand.loc[demand["product_id"] == product_id].sort_values(
            "month"
        )
        months = pd.DatetimeIndex(history["month"]).to_period("M")
        gaps = max(0, (months.max().ordinal - months.min().ordinal + 1) - len(months)) if len(months) else 0
        history_count = len(history)
        mean_demand = float(history["demand_units"].mean()) if history_count else np.nan
        std_demand = (
            float(history["demand_units"].std(ddof=1))
            if history_count >= 2
            else np.nan
        )
        cv = std_demand / mean_demand if mean_demand > 0 and np.isfinite(std_demand) else np.nan
        forecast_method = str(product_forecasts["forecast_method"].iloc[0])
        product_warnings = warning_lookup.get(product_id, pd.DataFrame())
        warning_categories = set(product_warnings.get("warning_category", []))
        latest_month = history["month"].max() if history_count else pd.NaT
        origin = pd.to_datetime(
            product_forecasts.get(
                "forecast_origin_month",
                pd.Series([product_forecasts["forecast_month"].min()]),
            ),
            errors="coerce",
        ).max()
        stale = (
            "STALE_HISTORY" in warning_categories
            or (
                pd.notna(origin)
                and pd.notna(latest_month)
                and latest_month.to_period("M") < origin.to_period("M")
            )
        )
        short = history_count < 12
        fallback = not forecast_method.startswith("random_forest")
        internal_gap = gaps > 0 or "INTERNAL_MISSING_MONTHS" in warning_categories

        evaluation_model = (
            "Moving Average"
            if "moving_average" in forecast_method
            else "Naive"
            if "naive" in forecast_method
            else "Random Forest"
        )
        product_evaluation = evaluation.loc[
            evaluation["product_id"] == product_id
        ].copy()
        product_evaluation["_period_order"] = product_evaluation["period"].map(
            evaluation_order
        ).fillna(-1)
        relevant = product_evaluation.loc[
            product_evaluation["model"] == evaluation_model
        ].sort_values("_period_order")
        if relevant.empty:
            evidence = "No matching product/model backtest evidence."
            metric_period = None
            evaluated_mae = np.nan
            evaluated_wape = np.nan
            baseline_comparison = "Unavailable"
        else:
            row = relevant.iloc[-1]
            metric_period = row["period"]
            evaluated_mae = row["mae"]
            evaluated_wape = row["wape_pct"]
            same_period = product_evaluation.loc[
                product_evaluation["period"] == metric_period
            ]
            baseline_mae = same_period.loc[
                same_period["model"].isin(["Naive", "Moving Average"]), "mae"
            ].dropna()
            if pd.isna(evaluated_mae) or baseline_mae.empty:
                baseline_comparison = "Insufficient comparable metrics"
            elif evaluated_mae < baseline_mae.min():
                baseline_comparison = "Selected method beat both available baselines by MAE"
            elif evaluated_mae > baseline_mae.min():
                baseline_comparison = "Selected method did not beat the best available baseline by MAE"
            else:
                baseline_comparison = "Selected method tied the best available baseline by MAE"
            evidence = (
                f"{metric_period}: {evaluation_model} MAE={evaluated_mae:.2f}, "
                f"WAPE={evaluated_wape:.2f}%"
                if pd.notna(evaluated_wape)
                else f"{metric_period}: {evaluation_model} MAE={evaluated_mae:.2f}; WAPE unavailable."
            )

        actions = []
        if stale:
            actions.append("Confirm whether the product is inactive and obtain recent demand before planning.")
        if short:
            actions.append("Collect additional observed months before relying on long-history comparisons.")
        if fallback:
            actions.append("Review the baseline forecast and compare it with suitable historical benchmarks.")
        if internal_gap:
            actions.append("Investigate internal missing product-month records; do not treat them as zero demand.")
        if not actions:
            actions.append("Continue routine monitoring; absence of these flags is not a guarantee of reliability.")
        records.append(
            {
                "product_id": product_id,
                "last_observed_month": latest_month,
                "observed_months": history_count,
                "stale_history": bool(stale),
                "short_history": bool(short),
                "baseline_fallback": bool(fallback),
                "internal_missing_months": int(gaps),
                "mean_observed_monthly_demand": mean_demand,
                "observed_monthly_demand_std": std_demand,
                "demand_coefficient_of_variation": cv,
                "forecast_method": forecast_method,
                "evaluation_period": metric_period,
                "evaluation_model": evaluation_model,
                "evaluation_mae": evaluated_mae,
                "evaluation_wape_pct": evaluated_wape,
                "baseline_comparison": baseline_comparison,
                "evaluation_evidence": evidence,
                "recommended_action": " ".join(actions),
            }
        )
    return pd.DataFrame(records)


def build_uploaded_demand_risk_assessment(
    demand: ForecastReadyDemandData,
    diagnostics: DemandPatternDiagnostics,
    forecasts: pd.DataFrame,
    forecast_warnings: pd.DataFrame,
    backtest: RollingOriginBacktest,
) -> pd.DataFrame:
    """Build per-series, evidence-labeled data-risk indicators for user uploads."""
    if diagnostics.frequency != demand.frequency or backtest.frequency != demand.frequency:
        raise ValueError("Demand, diagnostics, and backtest frequencies must match.")
    forecast_required = {"period", "predicted_demand", "model_name"}
    if not forecast_required.issubset(forecasts.columns) and not forecasts.empty:
        raise ValueError(
            f"Forecasts missing columns: {sorted(forecast_required - set(forecasts.columns))}"
        )
    warning_required = {"warning_message"}
    if not warning_required.issubset(forecast_warnings.columns) and not forecast_warnings.empty:
        raise ValueError(
            "Forecast warnings must contain a warning_message column."
        )

    history = demand.data.copy(deep=True)
    if "product_id" not in history.columns:
        history["product_id"] = None
    records: list[dict[str, object]] = []
    selection_map = {
        selection.product_id: selection
        for selection in backtest.selections
    }
    summary_map = {
        (summary.product_id, summary.model_name): summary
        for summary in backtest.model_summaries
    }
    for item in diagnostics.series:
        product_id = item.product_id
        product_history = (
            history.loc[history["product_id"].isna()]
            if product_id is None
            else history.loc[history["product_id"] == product_id]
        )
        known_history = product_history.loc[
            product_history["training_eligible"]
        ].sort_values("period")
        latest_period = (
            pd.Timestamp(product_history["period"].max())
            if not product_history.empty
            else pd.NaT
        )
        last_known_period = (
            pd.Timestamp(known_history["period"].max())
            if not known_history.empty
            else pd.NaT
        )
        recent_change_units = None
        recent_change_pct = None
        if (
            len(known_history) >= 2
            and pd.notna(latest_period)
            and last_known_period == latest_period
        ):
            previous = known_history.iloc[-2]
            latest = known_history.iloc[-1]
            period_step = {
                "daily": pd.Timedelta(days=1),
                "weekly": pd.Timedelta(days=7),
                "monthly": pd.offsets.MonthBegin(1),
            }[demand.frequency]
            if pd.Timestamp(previous["period"] + period_step) == last_known_period:
                previous_demand = float(previous["demand_quantity"])
                latest_demand = float(latest["demand_quantity"])
                recent_change_units = latest_demand - previous_demand
                if previous_demand != 0:
                    recent_change_pct = (
                        recent_change_units / previous_demand * 100
                    )
        stale = bool(
            pd.notna(latest_period)
            and pd.notna(last_known_period)
            and latest_period > last_known_period
        )
        product_forecasts = _rows_for_series(forecasts, product_id)
        product_warnings = _rows_for_series(forecast_warnings, product_id)
        selection = selection_map.get(product_id)
        selected_model = selection.selected_model if selection else None
        model_summary = (
            summary_map.get((product_id, selected_model))
            if selected_model is not None
            else None
        )
        enough_origins = bool(
            model_summary is not None
            and model_summary.comparable_origins
            >= backtest.config.min_comparable_origins
            and selected_model
            in {
                name
                for name, _ in (selection.candidate_scores if selection else ())
            }
        )
        metrics = (
            model_summary.comparable_metrics
            if enough_origins and model_summary is not None
            else None
        )
        candidate_scores = dict(selection.candidate_scores) if selection else {}
        selected_score = candidate_scores.get(selected_model)
        baseline_scores = {
            name: score
            for name, score in candidate_scores.items()
            if name in {"Naive", "Moving Average"}
        }
        if metrics is None:
            evidence = "Unavailable: no sufficiently representative comparable backtest."
            baseline_comparison = "Unavailable"
        elif not baseline_scores or selected_score is None:
            evidence = (
                f"{model_summary.comparable_origins} comparable origins; "
                f"{selection.metric.upper()}={selected_score!s}."
            )
            baseline_comparison = "No comparable simple baseline was selected."
        else:
            best_baseline_name, best_baseline_score = min(
                baseline_scores.items(), key=lambda entry: entry[1]
            )
            evidence = (
                f"{model_summary.comparable_origins} comparable origins; "
                f"{selection.metric.upper()}={selected_score:.3f}."
            )
            baseline_comparison = (
                f"Selected model beat {best_baseline_name}."
                if selected_score < best_baseline_score
                else f"Selected model did not beat {best_baseline_name}."
                if selected_score > best_baseline_score
                else f"Selected model tied {best_baseline_name}."
            )

        model_names = (
            sorted(product_forecasts["model_name"].dropna().astype(str).unique())
            if "model_name" in product_forecasts.columns
            else []
        )
        fallback = bool(
            selected_model in {"Naive", "Moving Average"}
            or any(name in {"Naive", "Moving Average"} for name in model_names)
        )
        classification_set = set(item.classifications)
        irregular = bool(
            classification_set.intersection({"intermittent", "erratic", "lumpy"})
        )
        short_history = item.eligible_periods < diagnostics.thresholds.min_history_periods
        actions: list[str] = []
        if stale:
            actions.append(
                "Confirm whether the latest unknown period reflects missing records or a stopped series."
            )
        if short_history:
            actions.append("Collect additional eligible periods before relying on model comparisons.")
        if item.unknown_periods:
            actions.append(
                "Resolve unknown periods; they are not confirmed zeros and were excluded from model fitting."
            )
        if irregular:
            actions.append(
                "Review intermittent or irregular demand patterns and the underlying period coverage."
            )
        if item.trend_evidence:
            actions.append(
                f"Investigate the fitted {item.trend_direction} demand trend and "
                "confirm it is not caused by missing coverage or a source/process change."
            )
        if fallback:
            actions.append(
                "Review the selected simple baseline and compare it with other eligible methods."
            )
        if selected_model is None:
            actions.append(
                "No model was selected as eligible by the configured rolling-origin comparison."
            )
        if metrics is None:
            actions.append(
                "Treat backtest performance as unavailable until enough comparable origins are present."
            )
        if not actions:
            actions.append(
                "Continue routine data-quality review; no listed indicator guarantees forecast reliability."
            )
        warning_text = "; ".join(
            product_warnings["warning_message"].dropna().astype(str)
        ) if "warning_message" in product_warnings.columns else ""
        records.append(
            {
                "product_id": product_id,
                "series_label": (
                    f"Product {product_id}" if product_id is not None else "Single series"
                ),
                "frequency": demand.frequency,
                "first_period": (
                    pd.Timestamp(product_history["period"].min())
                    if not product_history.empty
                    else pd.NaT
                ),
                "last_known_period": last_known_period,
                "history_periods": item.history_length,
                "eligible_periods": item.eligible_periods,
                "unknown_periods": item.unknown_periods,
                "confirmed_zero_periods": item.confirmed_zero_periods,
                "observed_zero_periods": item.observed_zero_periods,
                "coverage_ratio": item.coverage_ratio,
                "stale_history": stale,
                "short_history": bool(short_history),
                "internal_missing_months": item.missing_internal_periods,
                "intermittent_or_irregular": irregular,
                "trend_evidence": bool(item.trend_evidence),
                "trend_direction": item.trend_direction,
                "trend_slope_per_period": item.trend_slope_per_period,
                "trend_r_squared": item.trend_r_squared,
                "demand_pattern": "; ".join(item.classifications),
                "mean_demand_per_period": item.mean_demand,
                "demand_cv": item.coefficient_of_variation,
                "recent_change_units": recent_change_units,
                "recent_change_pct": recent_change_pct,
                "model_eligible": bool(not product_forecasts.empty),
                "selected_model": selected_model,
                "baseline_fallback": fallback,
                "forecast_warning": warning_text or None,
                "backtest_origins": (
                    model_summary.comparable_origins
                    if enough_origins and model_summary is not None
                    else 0
                ),
                "backtest_metric": selection.metric if enough_origins and selection else None,
                "backtest_mae": metrics.mae if metrics is not None else None,
                "backtest_wape": metrics.wape if metrics is not None else None,
                "baseline_comparison": baseline_comparison,
                "evaluation_evidence": evidence,
                "recommended_action": " ".join(actions),
            }
        )
    return pd.DataFrame(records)


def _rows_for_series(frame: pd.DataFrame, product_id: object | None) -> pd.DataFrame:
    if frame.empty or "product_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    if product_id is None:
        return frame.loc[frame["product_id"].isna()].copy()
    return frame.loc[frame["product_id"] == product_id].copy()


def summarize_risk_indicators(assessment: pd.DataFrame) -> pd.DataFrame:
    """Count products affected by each independent indicator; no total score."""
    required = {
        "stale_history",
        "short_history",
        "baseline_fallback",
        "internal_missing_months",
    }
    missing = required - set(assessment.columns)
    if missing:
        raise ValueError(f"Risk assessment missing columns: {sorted(missing)}")
    rows = []
    for label, column in [
        ("Stale product history", "stale_history"),
        ("Short history (<12 observed months)", "short_history"),
        ("Baseline forecast method", "baseline_fallback"),
    ]:
        rows.append(
            {
                "indicator": label,
                "affected_products": int(assessment[column].sum()),
            }
        )
    rows.append(
        {
            "indicator": "Internal missing months",
            "affected_products": int(
                assessment["internal_missing_months"].gt(0).sum()
            ),
        }
    )
    if "trend_evidence" in assessment.columns:
        rows.append(
            {
                "indicator": "Trend screening signal",
                "affected_products": int(assessment["trend_evidence"].sum()),
            }
        )
    return pd.DataFrame(rows)