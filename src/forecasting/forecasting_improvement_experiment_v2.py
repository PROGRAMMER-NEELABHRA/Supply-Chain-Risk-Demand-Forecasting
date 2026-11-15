"""Leakage-controlled rolling-origin comparison of forecasting methods.

Products are selected independently at each origin using only observations
available by that origin. Eligibility requires at least 13 consecutive
observed months ending at the origin: 12 months to form lag features and at
least one supervised Random Forest training row. Missing product-month rows
remain unknown, are never filled with zero, and are excluded from scored
rows when the corresponding future actual is unavailable.

The oracle is an explicitly non-deployable diagnostic that uses all observed
data through STABLE_END. It is emitted separately and is excluded from
deployable-method summaries and bootstrap comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "quality_review"
    / "monthly_demand_observed_rebuilt.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

START_MONTH = pd.Timestamp("2015-01-01")
STABLE_END = pd.Timestamp("2017-03-01")
HORIZON = 3
MIN_HISTORY_MONTHS = 13
HIGH_VOLUME_THRESHOLD = 100.0
RF_MIN_HISTORY = 12
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 5
RF_RANDOM_STATE = 42
BOOTSTRAP_BLOCK_LENGTH = HORIZON
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 0

METHODS = (
    "naive",
    "moving_average_3",
    "random_forest_per_product",
    "shrunk_daily_rate",
)
ORACLE_METHOD = "oracle_future_shrunk_daily_rate_NON_DEPLOYABLE"
METHOD_STATUS = {
    **{method: "deployable_candidate" for method in METHODS},
    ORACLE_METHOD: "non_deployable_oracle_uses_future_data",
}
RF_FEATURES = (
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "month_number",
    "quarter",
)


def load_observed_demand(file_path: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load observed monthly rows without imputing absent product-months."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Observed monthly demand not found: {path}")

    data = pd.read_csv(path, parse_dates=["month"])
    required = {"month", "product_id", "demand_units"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing demand columns: {sorted(missing)}")
    data = data.loc[:, ["month", "product_id", "demand_units"]].copy()
    data["month"] = pd.to_datetime(data["month"], errors="raise")
    data["demand_units"] = pd.to_numeric(data["demand_units"], errors="raise")
    if data[list(required)].isna().any().any():
        raise ValueError("Observed monthly demand contains missing values.")
    if (data["month"].dt.day != 1).any():
        raise ValueError("Monthly dates must be month starts.")
    if data.duplicated(["product_id", "month"]).any():
        raise ValueError("Duplicate observed product-month rows found.")
    if (data["demand_units"] < 0).any():
        raise ValueError("Observed demand cannot be negative.")
    return data.sort_values(["product_id", "month"]).reset_index(drop=True)


def _latest_contiguous_run(product: pd.DataFrame) -> pd.DataFrame:
    """Return only the final uninterrupted run of observed calendar months."""
    ordered = product.sort_values("month").reset_index(drop=True)
    month_ordinal = ordered["month"].dt.year * 12 + ordered["month"].dt.month
    run_id = month_ordinal.diff().ne(1).cumsum()
    latest_run_id = run_id.iloc[-1]
    return ordered.loc[run_id.eq(latest_run_id)].reset_index(drop=True)


def eligible_histories(
    data: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    minimum_history_months: int = MIN_HISTORY_MONTHS,
) -> dict[int, pd.DataFrame]:
    """Select products using only their contiguous history through origin."""
    if minimum_history_months < RF_MIN_HISTORY + 1:
        raise ValueError(
            "Minimum history must allow 12 lag months and a training target."
        )
    origin = pd.Timestamp(origin).to_period("M").to_timestamp()
    available = data.loc[data["month"] <= origin]
    histories: dict[int, pd.DataFrame] = {}
    for product_id, group in available.groupby("product_id", sort=True):
        run = _latest_contiguous_run(group)
        if (
            len(run) >= minimum_history_months
            and run["month"].iloc[-1] == origin
        ):
            histories[int(product_id)] = run
    return histories


def _history_band(month_count: int) -> str:
    if month_count < 18:
        return "13-17"
    if month_count < 24:
        return "18-23"
    return "24+"


def _volume_band(history: pd.DataFrame) -> str:
    return (
        "high"
        if float(history["demand_units"].mean()) >= HIGH_VOLUME_THRESHOLD
        else "low"
    )


def _rf_row(history: list[float], month: pd.Timestamp) -> list[float]:
    return [
        history[-1],
        history[-2],
        history[-3],
        history[-6],
        history[-12],
        sum(history[-3:]) / 3,
        sum(history[-6:]) / 6,
        month.month,
        month.quarter,
    ]


def _random_forest_forecast(
    history_data: pd.DataFrame,
    target_months: pd.DatetimeIndex,
) -> np.ndarray:
    """Fit on past-only lag rows and recursively predict the horizon."""
    values = history_data["demand_units"].astype(float).tolist()
    months = pd.DatetimeIndex(history_data["month"])
    feature_rows = []
    targets = []
    for index in range(RF_MIN_HISTORY, len(values)):
        feature_rows.append(_rf_row(values[:index], months[index]))
        targets.append(values[index])

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=1,
    ).fit(pd.DataFrame(feature_rows, columns=RF_FEATURES), targets)

    running = values.copy()
    forecasts = []
    for month in target_months:
        prediction = max(
            0.0,
            float(
                model.predict(
                    pd.DataFrame([_rf_row(running, month)], columns=RF_FEATURES)
                )[0]
            ),
        )
        forecasts.append(prediction)
        running.append(prediction)
    return np.asarray(forecasts)


def _shrunk_daily_rates(
    histories: Mapping[int, pd.DataFrame],
    *,
    threshold: float = HIGH_VOLUME_THRESHOLD,
) -> dict[int, float]:
    """Estimate empirical-Bayes-shrunk daily demand rates from histories."""
    own_rate: dict[int, float] = {}
    sampling_variance: dict[int, float] = {}
    high_volume: dict[int, bool] = {}
    for product_id, history in histories.items():
        days = history["month"].dt.days_in_month.to_numpy(dtype=float)
        rates = history["demand_units"].to_numpy(dtype=float) / days
        own_rate[product_id] = float(rates.mean())
        sampling_variance[product_id] = float(rates.var(ddof=1) / len(rates))
        high_volume[product_id] = (
            float(history["demand_units"].mean()) >= threshold
        )

    shrunk = own_rate.copy()
    for is_high in (True, False):
        product_ids = [
            product_id
            for product_id in histories
            if high_volume[product_id] is is_high
        ]
        if len(product_ids) < 3:
            continue
        group_rates = np.asarray([own_rate[pid] for pid in product_ids])
        group_sampling = np.asarray(
            [sampling_variance[pid] for pid in product_ids]
        )
        between_variance = max(
            float(group_rates.var(ddof=1) - group_sampling.mean()),
            0.0,
        )
        group_mean = float(group_rates.mean())
        for product_id in product_ids:
            denominator = between_variance + sampling_variance[product_id]
            weight = (
                between_variance / denominator if denominator > 0 else 0.0
            )
            shrunk[product_id] = group_mean + weight * (
                own_rate[product_id] - group_mean
            )
    return shrunk


def _forecast_history(
    histories: Mapping[int, pd.DataFrame],
    origin: pd.Timestamp,
) -> dict[str, dict[int, np.ndarray]]:
    """Build four method forecasts from the same origin-specific cohort."""
    origin = pd.Timestamp(origin).to_period("M").to_timestamp()
    target_months = pd.date_range(
        origin + pd.offsets.MonthBegin(1),
        periods=HORIZON,
        freq="MS",
    )
    forecasts: dict[str, dict[int, np.ndarray]] = {
        method: {} for method in METHODS
    }

    shrunk_rates = _shrunk_daily_rates(histories)
    for product_id, history_data in histories.items():
        values = history_data["demand_units"].astype(float).to_numpy()
        forecasts["naive"][product_id] = np.repeat(values[-1], HORIZON)

        moving_history = values[-3:].tolist()
        moving_forecasts = []
        for _ in range(HORIZON):
            prediction = float(np.mean(moving_history[-3:]))
            moving_forecasts.append(prediction)
            moving_history.append(prediction)
        forecasts["moving_average_3"][product_id] = np.asarray(
            moving_forecasts
        )
        forecasts["random_forest_per_product"][product_id] = (
            _random_forest_forecast(history_data, target_months)
        )
        forecasts["shrunk_daily_rate"][product_id] = (
            shrunk_rates[product_id]
            * target_months.days_in_month.to_numpy(dtype=float)
        )
    return forecasts


def _future_actuals(data: pd.DataFrame) -> dict[tuple[int, pd.Timestamp], float]:
    return {
        (int(row.product_id), pd.Timestamp(row.month)): float(row.demand_units)
        for row in data.itertuples(index=False)
    }


def predict_at_origin(
    data: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    include_oracle: bool = False,
) -> pd.DataFrame:
    """Forecast one origin; future rows are read only as scoring labels."""
    origin = pd.Timestamp(origin).to_period("M").to_timestamp()
    histories = eligible_histories(data, origin)
    if not histories:
        return pd.DataFrame(columns=_PREDICTION_COLUMNS)

    method_forecasts = _forecast_history(histories, origin)
    target_months = pd.date_range(
        origin + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS"
    )
    actuals = _future_actuals(data)
    rows = []
    for method in METHODS:
        for product_id, history in histories.items():
            for horizon, target_month in enumerate(target_months, start=1):
                actual = actuals.get((product_id, target_month))
                rows.append(
                    {
                        "method": method,
                        "method_status": METHOD_STATUS[method],
                        "origin_month": origin,
                        "history_months": len(history),
                        "history_band": _history_band(len(history)),
                        "volume_band": _volume_band(history),
                        "eligible_products": len(histories),
                        "product_id": product_id,
                        "horizon": horizon,
                        "target_month": target_month,
                        "actual": actual,
                        "actual_observed": actual is not None,
                        "forecast": float(
                            method_forecasts[method][product_id][horizon - 1]
                        ),
                    }
                )

    if include_oracle:
        full_histories = {
            int(product_id): _latest_contiguous_run(group)
            for product_id, group in data.loc[
                data["month"] <= STABLE_END
            ].groupby("product_id", sort=True)
        }
        oracle_rates = _shrunk_daily_rates(full_histories)
        for product_id, history in histories.items():
            if product_id not in oracle_rates:
                continue
            for horizon, target_month in enumerate(target_months, start=1):
                actual = actuals.get((product_id, target_month))
                rows.append(
                    {
                        "method": ORACLE_METHOD,
                        "method_status": METHOD_STATUS[ORACLE_METHOD],
                        "origin_month": origin,
                        "history_months": len(history),
                        "history_band": _history_band(len(history)),
                        "volume_band": _volume_band(history),
                        "eligible_products": len(histories),
                        "product_id": product_id,
                        "horizon": horizon,
                        "target_month": target_month,
                        "actual": actual,
                        "actual_observed": actual is not None,
                        "forecast": float(
                            oracle_rates[product_id]
                            * target_month.days_in_month
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=_PREDICTION_COLUMNS)


_PREDICTION_COLUMNS = [
    "method",
    "method_status",
    "origin_month",
    "history_months",
    "history_band",
    "volume_band",
    "eligible_products",
    "product_id",
    "horizon",
    "target_month",
    "actual",
    "actual_observed",
    "forecast",
]


def _metric_values(group: pd.DataFrame) -> dict[str, float | int]:
    scored = group.loc[group["actual"].notna()]
    if scored.empty:
        return {
            "scored_rows": 0,
            "scored_products": 0,
            "MAE": np.nan,
            "pooled_WAPE_%": np.nan,
            "per_product_WAPE_%": np.nan,
            "bias_%": np.nan,
        }
    error = scored["forecast"] - scored["actual"]
    actual_total = float(scored["actual"].sum())
    if actual_total <= 0:
        raise ValueError("WAPE is undefined when total actual demand is zero.")
    product_totals = scored.assign(abs_error=error.abs()).groupby(
        "product_id"
    ).agg(
        absolute_error=("abs_error", "sum"),
        actual=("actual", "sum"),
    )
    eligible_product_totals = product_totals.loc[product_totals["actual"] > 0]
    return {
        "scored_rows": len(scored),
        "scored_products": int(scored["product_id"].nunique()),
        "MAE": float(error.abs().mean()),
        "pooled_WAPE_%": float(error.abs().sum() / actual_total * 100),
        "per_product_WAPE_%": float(
            (
                eligible_product_totals["absolute_error"]
                / eligible_product_totals["actual"]
            ).mean()
            * 100
        ),
        "bias_%": float(error.sum() / actual_total * 100),
    }


def build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize deployable methods overall and by observed-history band."""
    deployable = predictions.loc[predictions["method"].isin(METHODS)]
    rows = []
    for method, method_data in deployable.groupby("method", sort=False):
        for scope, band, group in [
            ("overall", "all", method_data),
            *[
                ("history_band", history_band, band_data)
                for history_band, band_data in method_data.groupby(
                    "history_band", sort=True
                )
            ],
            *[
                ("volume_band", volume_band, band_data)
                for volume_band, band_data in method_data.groupby(
                    "volume_band", sort=True
                )
            ],
        ]:
            metrics = _metric_values(group)
            rows.append(
                {
                    "method": method,
                    "method_status": METHOD_STATUS[method],
                    "scope": scope,
                    "cohort": band,
                    "origins": group["origin_month"].nunique(),
                    "eligible_product_origin_pairs": group[
                        ["origin_month", "product_id"]
                    ].drop_duplicates().shape[0],
                    **metrics,
                    "paired_wape_difference_vs_random_forest_pp": np.nan,
                    "paired_block_bootstrap_95ci_low_pp": np.nan,
                    "paired_block_bootstrap_median_pp": np.nan,
                    "paired_block_bootstrap_95ci_high_pp": np.nan,
                    "bootstrap_block_origins": np.nan,
                    "bootstrap_replicates": np.nan,
                    "bootstrap_scheme": "",
                    "bootstrap_limitation": "",
                }
            )

    ci = paired_block_bootstrap(predictions)
    candidate_row = next(
        row
        for row in rows
        if row["method"] == "shrunk_daily_rate" and row["scope"] == "overall"
    )
    candidate_row[
        "paired_wape_difference_vs_random_forest_pp"
    ] = candidate_row["pooled_WAPE_%"] - next(
        row["pooled_WAPE_%"]
        for row in rows
        if row["method"] == "random_forest_per_product"
        and row["scope"] == "overall"
    )
    candidate_row["paired_block_bootstrap_95ci_low_pp"] = ci[0]
    candidate_row["paired_block_bootstrap_median_pp"] = ci[1]
    candidate_row["paired_block_bootstrap_95ci_high_pp"] = ci[2]
    candidate_row["bootstrap_block_origins"] = BOOTSTRAP_BLOCK_LENGTH
    candidate_row["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    candidate_row["bootstrap_scheme"] = (
        "Paired moving blocks of three adjacent forecast origins; "
        "all products and horizons remain grouped."
    )
    candidate_row["bootstrap_limitation"] = (
        "Only 12 origins are available; percentile intervals may be "
        "unstable, and dependence can remain across block boundaries."
    )
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap paired pooled-WAPE differences using contiguous origin blocks.

    Each resampled block contains three adjacent origins. All products and
    horizons from each selected origin are kept together for both methods.
    This preserves pairing and much of the dependence from overlapping
    horizons; the small number of origins still limits interval reliability.
    """
    challenger = predictions.loc[
        predictions["method"].eq("shrunk_daily_rate")
    ].sort_values(["origin_month", "product_id", "horizon"])
    reference = predictions.loc[
        predictions["method"].eq("random_forest_per_product")
    ].sort_values(["origin_month", "product_id", "horizon"])
    keys = ["origin_month", "product_id", "horizon", "target_month"]
    if not challenger[keys + ["actual"]].reset_index(drop=True).equals(
        reference[keys + ["actual"]].reset_index(drop=True)
    ):
        raise ValueError("Paired methods do not share identical scored rows.")
    origins = sorted(challenger["origin_month"].unique())
    if len(origins) < BOOTSTRAP_BLOCK_LENGTH:
        raise ValueError("Not enough origins for a horizon-sized block bootstrap.")

    paired = predictions.loc[
        predictions["method"].isin(
            ("shrunk_daily_rate", "random_forest_per_product")
        )
        & predictions["actual"].notna()
    ].assign(
        abs_error=lambda frame: (
            frame["forecast"] - frame["actual"]
        ).abs()
    )
    origin_totals = paired.groupby(
        ["origin_month", "method"], sort=True
    ).agg(
        abs_error=("abs_error", "sum"),
        actual=("actual", "sum"),
    )
    sums = origin_totals.unstack("method").reindex(origins)
    challenger_error = sums["abs_error"]["shrunk_daily_rate"].to_numpy()
    reference_error = sums["abs_error"][
        "random_forest_per_product"
    ].to_numpy()
    actual_total = sums["actual"]["shrunk_daily_rate"].to_numpy()
    rng = np.random.default_rng(seed)
    possible_starts = np.arange(len(origins) - BOOTSTRAP_BLOCK_LENGTH + 1)
    blocks_needed = int(np.ceil(len(origins) / BOOTSTRAP_BLOCK_LENGTH))
    differences = np.empty(n_bootstrap, dtype=float)
    for replicate in range(n_bootstrap):
        starts = rng.choice(
            possible_starts, size=blocks_needed, replace=True
        )
        sampled_indices = np.concatenate(
            [
                np.arange(start, start + BOOTSTRAP_BLOCK_LENGTH)
                for start in starts
            ]
        )[: len(origins)]
        denominator = actual_total[sampled_indices].sum()
        differences[replicate] = (
            100
            * (
                challenger_error[sampled_indices].sum()
                - reference_error[sampled_indices].sum()
            )
            / denominator
        )
    return tuple(np.percentile(differences, [2.5, 50.0, 97.5]).tolist())


def build_origin_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report metrics by origin, horizon, and available-history band."""
    rows = []
    deployable = predictions.loc[predictions["method"].isin(METHODS)]
    group_specs: list[tuple[str, list[str]]] = [
        ("origin", ["origin_month", "method"]),
        ("origin_horizon", ["origin_month", "horizon", "method"]),
        ("history_band", ["history_band", "method"]),
        ("volume_band", ["volume_band", "method"]),
    ]
    for metric_level, group_columns in group_specs:
        for keys, group in deployable.groupby(group_columns, sort=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            metadata = dict(zip(group_columns, key_values, strict=True))
            metrics = _metric_values(group)
            rows.append(
                {
                    "metric_level": metric_level,
                    **metadata,
                    "eligible_products": int(group["eligible_products"].max()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_experiment(data: pd.DataFrame | None = None) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Run 12 rolling origins and return outputs without writing files."""
    if data is None:
        data = load_observed_demand()
    data = data.loc[data["month"] <= STABLE_END].copy()
    prediction_frames = []
    for history_months in range(MIN_HISTORY_MONTHS, 25):
        origin = START_MONTH + pd.DateOffset(months=history_months - 1)
        prediction_frames.append(
            predict_at_origin(data, pd.Timestamp(origin), include_oracle=True)
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = build_summary(predictions)
    origin_metrics = build_origin_metrics(predictions)
    return predictions, summary, origin_metrics


def main() -> None:
    """Run and save only the separate V2 experiment outputs."""
    predictions, summary, origin_metrics = run_experiment()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        OUTPUT_DIR / "improvement_experiment_v2_predictions.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "improvement_experiment_v2_summary.csv",
        index=False,
    )
    origin_metrics.to_csv(
        OUTPUT_DIR / "improvement_experiment_v2_origin_metrics.csv",
        index=False,
    )
    print("\nV2 origin-specific evaluation summary")
    print(
        summary.loc[summary["scope"].eq("overall"), [
            "method",
            "MAE",
            "pooled_WAPE_%",
            "per_product_WAPE_%",
            "bias_%",
        ]].to_string(index=False)
    )
    print("\nDirect paired comparison: shrunk daily rate minus per-product RF")
    candidate = summary.loc[
        summary["method"].eq("shrunk_daily_rate")
        & summary["scope"].eq("overall")
    ].iloc[0]
    print(
        f"Pooled WAPE difference: "
        f"{candidate['paired_wape_difference_vs_random_forest_pp']:.3f} pp; "
        f"three-origin-block bootstrap 95% CI: "
        f"[{candidate['paired_block_bootstrap_95ci_low_pp']:.3f}, "
        f"{candidate['paired_block_bootstrap_95ci_high_pp']:.3f}] pp"
    )
    print(f"\nSaved V2-only outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
