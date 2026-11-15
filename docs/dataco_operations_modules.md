# DataCo Operations Dashboard Modules

These pages default to the configured DataCo history through August 2017 and
saved forecasts for September–November 2017. Users can select a prepared
user-uploaded dataset in any of the three operations pages; that data and its
generated forecasts are read from the existing User Demand Forecasting session
state and are never written to DataCo files. These are exploratory
decision-support tools, not validated operational policies.

## Inventory planning

`src/inventory/inventory_analysis.py` computes:

- Observed and forecast averages per selected frequency period. Unknown periods
  are excluded from averages and are not imputed as zero.
- Lead-time demand by distributing each forecast period's demand uniformly
  across its calendar days. If lead time exceeds the forecast horizon,
  the observed average daily demand is used for the remainder and identified
  as an extrapolation.
- Reorder point as lead-time demand plus safety stock when safety stock is
  supportable; otherwise it is the lead-time demand estimate without a safety
  buffer.
- Safety stock only with at least 12 contiguous observed periods and a
  user-entered lead-time standard deviation. It uses
  `z(service_level) * sqrt(L * sigma_d^2 + mean_d^2 * sigma_L^2)`. Period
  means are converted to daily means using the period's average day count;
  period-level standard deviations are divided by the square root of the
  average period length, assuming independent within-period demand. Demand
  and lead-time variation are treated as normally approximated.
- EOQ as `sqrt(2 * annualized observed demand * ordering cost / annual holding
  cost per unit)`. Costs are user inputs; annual demand is annualized from the
  selected daily, weekly, or monthly observed frequency.
- Stockout dates from user-entered stock and the selected forecast profile,
  assuming demand is uniform within each forecast period. The available
  forecast horizon is not extrapolated for stockout dates.

Current stock, supplier lead time and its variability, service target, and
cost inputs are not in the DataCo source. The UI leaves product inputs blank
and shows why unsupported calculations are unavailable. It does not infer
inventory balances or supplier behavior.

## Demand and forecast data risk

`src/risk/risk_score.py` reports independent, explainable indicators:

- Stale history: last observed product month precedes the forecast origin.
- Short history: fewer than 12 observed product-month rows.
- Baseline fallback: the saved method is not recursive Random Forest.
- Internal missing months: absent calendar months between a product's first
  and last observations; these remain unknown, not zero demand.
- Demand mean, sample standard deviation, and coefficient of variation are
  descriptive values from recorded rows only; no arbitrary risk band or
  composite score is assigned.
- The latest matching saved historical product/method evaluation is shown
  where available, including MAE/WAPE and MAE comparison with simple
  baselines. It is evidence from the named evaluation window, not a future
  accuracy guarantee.

The DataCo files do not establish supplier reliability, geopolitical exposure,
logistics bottlenecks, single-source dependency, or actual disruption, and the
page does not claim to measure them.

For uploads, the risk register uses the existing demand diagnostics and
rolling-origin model comparison. It separately lists unknown periods,
confirmed-zero periods, short or stale history, irregular-pattern screening,
forecast eligibility, selected simple baselines, and comparable backtest
evidence. Backtest metrics are shown only when the selected method meets the
comparison's minimum shared-origin requirement; otherwise they remain
unavailable. The page labels a single-series upload as a series rather than
inventing a product identifier.

## Deterministic scenarios

`src/scenarios/simulation.py` multiplies each saved forecast by user-adjusted
optimistic, pessimistic, and custom factors. Unit and percentage deltas are
relative to the saved forecast; percentage deltas are unavailable when the
baseline is zero. Portfolio percentages are calculated after summing product
demand by the selected daily, weekly, or monthly period.

Saved evaluation residuals are not used to simulate uncertainty: the available
evidence consists of a limited three-month recursive holdout and a separate
four-month rolling evaluation for only nine complete-history products. Their
cohort sizes and forecast protocols differ, so pooling them would not provide
a defensible portfolio residual distribution. The simulator therefore remains
deterministic and makes no Monte Carlo, supplier-shock, or elasticity claim.
