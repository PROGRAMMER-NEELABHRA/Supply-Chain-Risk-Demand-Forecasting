"""Rolling-origin backtest: current forecasters vs. improvement candidates.

Why this exists
---------------
The existing evaluation scores one origin (train to Oct 2016, predict
Nov 2016 - Jan 2017) on 162 product-month rows. That is too little data to
tell a real improvement from luck. This script re-scores every method over
many forecast origins inside the stable data regime (Jan 2015 - Mar 2017,
54 products observed every month), using the same 3-month horizon.

Methods compared
----------------
Current:    naive, moving_average_3, random_forest_per_product (as shipped)
Candidates: longer-window means, days-in-month-adjusted level models,
            partial pooling (shrinkage), a global gradient-boosting model
            trained across all products, and simple ensembles.

Run from the repository root:
    python -m src.forecasting.forecasting_improvement_experiment

Only numpy, pandas, and scikit-learn are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "quality_review" / "monthly_demand_observed_rebuilt.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

STABLE_END = pd.Timestamp("2017-03-01")  # last month before the coverage change
HORIZON = 3
MIN_HISTORY = 12  # lag-12 feature needs a year of history
HIGH_VOLUME_THRESHOLD = 100  # units/month; separates the two product tiers
RF_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "month_number", "quarter",
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_panel() -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Return demand matrix (months x products), month index, product ids.

    Only products observed in every month of the stable regime are kept.
    """
    df = pd.read_csv(DATA_FILE, parse_dates=["month"])
    df = df[df["month"] <= STABLE_END]
    wide = df.pivot(index="month", columns="product_id", values="demand_units")
    wide = wide.dropna(axis=1).sort_index()
    return wide.to_numpy(dtype=float), wide.index, wide.columns.to_numpy()


def days_in_month(months: pd.DatetimeIndex) -> np.ndarray:
    return months.days_in_month.to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Forecasters. Each receives history (L x N) plus calendar info and returns
# a (HORIZON x N) array of forecasts for the next HORIZON months.
# --------------------------------------------------------------------------
def naive(hist, hist_days, tgt_days, tgt_months):
    return np.tile(hist[-1], (HORIZON, 1))


def moving_average_3(hist, hist_days, tgt_days, tgt_months):
    """Recursive 3-month moving average, exactly as in the shipped evaluation."""
    running = [row for row in hist[-3:]]
    out = []
    for _ in range(HORIZON):
        pred = np.mean(running[-3:], axis=0)
        out.append(pred)
        running.append(pred)
    return np.stack(out)


def mean_12(hist, hist_days, tgt_days, tgt_months):
    return np.tile(hist[-12:].mean(0), (HORIZON, 1))


def mean_all(hist, hist_days, tgt_days, tgt_months):
    return np.tile(hist.mean(0), (HORIZON, 1))


def seasonal_naive(hist, hist_days, tgt_days, tgt_months):
    L = len(hist)
    return np.stack([hist[L + h - 12] for h in range(HORIZON)])


def daily_rate_mean_12(hist, hist_days, tgt_days, tgt_months):
    """Average demand per day over 12 months x days in each target month."""
    rate = (hist / hist_days[:, None])[-12:].mean(0)
    return rate[None, :] * tgt_days[:, None]


def daily_rate_mean_all(hist, hist_days, tgt_days, tgt_months):
    rate = (hist / hist_days[:, None]).mean(0)
    return rate[None, :] * tgt_days[:, None]


def _shrunk_rate(hist, hist_days, threshold=HIGH_VOLUME_THRESHOLD):
    """Empirical-Bayes partial pooling of per-day rates within volume tiers.

    Each product's mean rate is pulled toward its tier mean. The pull is
    strong when products in a tier are statistically indistinguishable
    (between-product variance <= sampling noise) and negligible when they
    genuinely differ. Uses history only.
    """
    L = len(hist)
    rate = hist / hist_days[:, None]
    own = rate.mean(0)
    sampling_var = rate.var(0, ddof=1) / L
    tier_high = hist.mean(0) >= threshold
    out = own.copy()
    for mask in (tier_high, ~tier_high):
        if mask.sum() < 3:
            continue
        group_mean = own[mask].mean()
        tau2 = max(own[mask].var(ddof=1) - sampling_var[mask].mean(), 0.0)
        weight = tau2 / (tau2 + sampling_var[mask])
        out[mask] = group_mean + weight * (own[mask] - group_mean)
    return out


def shrunk_daily_rate(hist, hist_days, tgt_days, tgt_months):
    return _shrunk_rate(hist, hist_days)[None, :] * tgt_days[:, None]


_RF_CACHE: dict[int, np.ndarray] = {}


def random_forest_per_product(hist, hist_days, tgt_days, tgt_months):
    """The shipped approach: one RF per product, recursive 3-step forecast."""
    L, N = hist.shape
    if L in _RF_CACHE:
        return _RF_CACHE[L]
    hist_months = tgt_months[0] - pd.offsets.MonthBegin(1) * np.arange(L, 0, -1)
    out = np.zeros((HORIZON, N))
    for j in range(N):
        series = hist[:, j].tolist()
        rows, targets = [], []
        for i in range(MIN_HISTORY, L):
            rows.append(_rf_row(series[:i], hist_months[i]))
            targets.append(series[i])
        model = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
        ).fit(pd.DataFrame(rows, columns=RF_FEATURES), targets)
        running = series.copy()
        for h in range(HORIZON):
            x = pd.DataFrame([_rf_row(running, tgt_months[h])], columns=RF_FEATURES)
            pred = max(0.0, float(model.predict(x)[0]))
            out[h, j] = pred
            running.append(pred)
    _RF_CACHE[L] = out
    return out


def _rf_row(history, month):
    return [
        history[-1], history[-2], history[-3], history[-6], history[-12],
        sum(history[-3:]) / 3, sum(history[-6:]) / 6,
        month.month, month.quarter,
    ]


def global_gbm(hist, hist_days, tgt_days, tgt_months):
    """One model across ALL products, on scale-free inputs.

    Target and features are expressed relative to each product's own
    per-day level, so a 30-unit product and a 2,000-unit product teach the
    same model. log-level is kept as a feature so it can learn how much to
    shrink noisy low-volume levels. Direct strategy: horizon is a feature
    (no recursive error accumulation).
    """
    L, N = hist.shape
    rate = hist / hist_days[:, None]
    X, y = [], []
    for o in range(5, L - 1):  # origin o = last observed index
        level = rate[: o + 1].mean(0)
        recent3 = rate[max(0, o - 2): o + 1].mean(0)
        for h in range(1, HORIZON + 1):
            t = o + h
            if t > L - 1:
                break
            X.append(_gbm_features(level, recent3, rate[o], h, o))
            y.append(rate[t] / level)
    X = np.vstack(X)
    y = np.concatenate(y)

    model = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=150,
        min_samples_leaf=30, random_state=42,
    ).fit(X, y)

    o = L - 1
    level = rate.mean(0)
    recent3 = rate[-3:].mean(0)
    out = np.zeros((HORIZON, N))
    for h in range(1, HORIZON + 1):
        ratio = model.predict(_gbm_features(level, recent3, rate[o], h, o))
        out[h - 1] = ratio * level * tgt_days[h - 1]
    return out


def _gbm_features(level, recent3, last, h, o):
    n = len(level)
    return np.column_stack([
        np.log(level), recent3 / level, last / level,
        np.full(n, h), np.full(n, o),
    ])


def ensemble_shrunk_gbm(hist, hist_days, tgt_days, tgt_months):
    return 0.5 * shrunk_daily_rate(hist, hist_days, tgt_days, tgt_months) + 0.5 * (
        global_gbm(hist, hist_days, tgt_days, tgt_months)
    )


def ensemble_rf_ma3(hist, hist_days, tgt_days, tgt_months):
    return 0.5 * random_forest_per_product(
        hist, hist_days, tgt_days, tgt_months
    ) + 0.5 * moving_average_3(hist, hist_days, tgt_days, tgt_months)


# --------------------------------------------------------------------------
# Backtest and metrics
# --------------------------------------------------------------------------
def backtest(methods, X, months, products, history_lengths):
    days = days_in_month(months)
    records = []
    for L in history_lengths:
        hist, hist_days = X[:L], days[:L]
        tgt_idx = np.arange(L, L + HORIZON)
        actual = X[tgt_idx]
        for name, fn in methods.items():
            pred = fn(hist, hist_days, days[tgt_idx], months[tgt_idx])
            for h in range(HORIZON):
                records.append(pd.DataFrame({
                    "method": name, "origin_len": L, "horizon": h + 1,
                    "target_month": months[tgt_idx[h]], "product_id": products,
                    "actual": actual[h], "forecast": pred[h],
                }))
    return pd.concat(records, ignore_index=True)


def summarise(results: pd.DataFrame, product_mean: pd.Series) -> pd.DataFrame:
    results = results.assign(
        abs_err=(results.forecast - results.actual).abs(),
        err=results.forecast - results.actual,
        high_volume=results.product_id.map(product_mean) >= HIGH_VOLUME_THRESHOLD,
    )

    def block(g):
        per_prod = g.groupby("product_id").apply(
            lambda p: p.abs_err.sum() / p.actual.sum(), include_groups=False
        )
        return pd.Series({
            "MAE": g.abs_err.mean(),
            "pooled_WAPE_%": 100 * g.abs_err.sum() / g.actual.sum(),
            "per_product_WAPE_%": 100 * per_prod.mean(),
            "bias_%": 100 * g.err.sum() / g.actual.sum(),
        })

    overall = results.groupby("method").apply(block, include_groups=False)
    low = results[~results.high_volume].groupby("method").apply(block, include_groups=False)
    high = results[results.high_volume].groupby("method").apply(block, include_groups=False)
    table = overall.join(
        low[["per_product_WAPE_%"]].rename(columns={"per_product_WAPE_%": "low_vol_WAPE_%"})
    ).join(
        high[["pooled_WAPE_%"]].rename(columns={"pooled_WAPE_%": "high_vol_WAPE_%"})
    )
    return table.round(2)


def paired_bootstrap(results, challenger, reference, n_boot=2000, seed=0):
    """CI on (challenger - reference) pooled WAPE, resampling target months.

    Months are resampled (not rows) because every product shares the same
    calendar shocks, so rows within a month are not independent.
    """
    rng = np.random.default_rng(seed)
    piv = {}
    for name in (challenger, reference):
        g = results[results.method == name]
        piv[name] = g.assign(ae=(g.forecast - g.actual).abs()).groupby(
            "target_month"
        ).agg(ae=("ae", "sum"), act=("actual", "sum"))
    months = piv[challenger].index
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(months, size=len(months), replace=True)
        c, r = piv[challenger].loc[pick], piv[reference].loc[pick]
        diffs.append(100 * (c.ae.sum() / c.act.sum() - r.ae.sum() / r.act.sum()))
    return np.percentile(diffs, [2.5, 50, 97.5])


# --------------------------------------------------------------------------
def main() -> None:
    X, months, products = load_panel()
    product_mean = pd.Series(X.mean(0), index=products)
    print(f"Panel: {X.shape[0]} months x {X.shape[1]} products "
          f"({months[0]:%b %Y} - {months[-1]:%b %Y})")

    # 1) Sanity check: reproduce the published single-origin result.
    published = {"naive": naive, "moving_average_3": moving_average_3,
                 "random_forest_per_product": random_forest_per_product}
    check = backtest(published, X, months, products, history_lengths=[22])
    print("\n[Check] Published origin (train to Oct 2016, test Nov 2016-Jan 2017)."
          "\nExpected: Naive MAE 23.43 / WAPE 10.89, MA 17.29 / 8.03, RF 15.55 / 7.22")
    print(summarise(check, product_mean)[["MAE", "pooled_WAPE_%"]])

    # 2) Rolling-origin comparison. History length 18..24 gives the per-product
    #    RF at least 6 training rows, and 7 origins x 3 horizons x 54 products.
    methods = {
        "naive": naive,
        "moving_average_3": moving_average_3,
        "random_forest_per_product": random_forest_per_product,
        "mean_12": mean_12,
        "mean_all": mean_all,
        "seasonal_naive": seasonal_naive,
        "daily_rate_mean_12": daily_rate_mean_12,
        "daily_rate_mean_all": daily_rate_mean_all,
        "shrunk_daily_rate": shrunk_daily_rate,
        "global_gbm": global_gbm,
        "ensemble_shrunk_gbm": ensemble_shrunk_gbm,
        "ensemble_rf_ma3": ensemble_rf_ma3,
    }
    # Reference only: the same level model fitted on ALL 27 months, including
    # the months being predicted. It uses the future, so it cannot be achieved;
    # it estimates the noise floor for any level + calendar model.
    all_days = days_in_month(months)
    methods["oracle_floor (uses future)"] = (
        lambda hist, hd, td, tm: _shrunk_rate(X, all_days)[None, :] * td[:, None]
    )

    results = backtest(methods, X, months, products, range(18, 25))
    table = summarise(results, product_mean).sort_values("pooled_WAPE_%")
    print("\n[Rolling origin] 7 origins x 3 horizons x 54 products "
          f"= {len(results) // len(methods)} forecasts per method")
    print(table.to_string())

    print("\nPaired bootstrap of pooled-WAPE difference vs moving_average_3 "
          "(negative = better; 95% CI):")
    for name in table.index:
        if name in ("moving_average_3", "oracle_floor (uses future)"):
            continue
        lo, mid, hi = paired_bootstrap(results, name, "moving_average_3")
        print(f"  {name:28s} {mid:6.2f}  [{lo:6.2f}, {hi:6.2f}]")

    # Consistency: in how many of the 7 origins does the challenger beat the RF?
    per_origin = results.assign(ae=(results.forecast - results.actual).abs())
    per_origin = per_origin.groupby(["origin_len", "method"]).agg(
        ae=("ae", "sum"), act=("actual", "sum"))
    per_origin["wape"] = 100 * per_origin.ae / per_origin.act
    wape = per_origin["wape"].unstack("method")
    print("\nOrigins (of 7) where each method beats the shipped per-product RF:")
    for name in ["shrunk_daily_rate", "daily_rate_mean_all", "mean_all",
                 "global_gbm", "moving_average_3"]:
        wins = int((wape[name] < wape["random_forest_per_product"]).sum())
        print(f"  {name:28s} {wins}/7")

    # Robustness: cheap methods over a longer set of origins (12..24 months of
    # history). The per-product RF cannot run with so little history.
    cheap = {k: v for k, v in methods.items()
             if k in ("naive", "moving_average_3", "mean_12", "mean_all",
                      "daily_rate_mean_all", "shrunk_daily_rate", "global_gbm")}
    wide = backtest(cheap, X, months, products, range(12, 25))
    print("\n[Robustness] 13 origins (12..24 months of history), cheap methods only")
    print(summarise(wide, product_mean).sort_values("pooled_WAPE_%")
          [["MAE", "pooled_WAPE_%", "per_product_WAPE_%"]].to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "improvement_experiment_predictions.csv", index=False)
    table.to_csv(OUTPUT_DIR / "improvement_experiment_summary.csv")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
