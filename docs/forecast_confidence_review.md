# DataCo Forecast Confidence Review

## Decision

Do not replace the current confidence calculation with new HIGH/MEDIUM/LOW
rules yet. A fresh, in-memory run of the existing leakage-controlled
rolling-origin evaluator on the configured DataCo source produced comparative
evidence for a limited cohort, but not enough evidence to calibrate reliable
per-product confidence categories or their minimum sample sizes. The proposed
criteria below are a review framework, not implemented rating rules.

## Current calculation and weaknesses

`src/forecasting/generate_future_forecasts.py::forecast_one_product` assigns
confidence alongside method selection:

- Random Forest is assigned HIGH when the latest contiguous observed run has
  more than 12 months.
- Moving-average fallback is assigned MEDIUM for at least two records.
- Naive-last-observation fallback is assigned LOW for one record.
- Any product whose latest record precedes the portfolio cutoff is forced to
  LOW, regardless of history or method.
- Internal missing months downgrade HIGH to MEDIUM; other tiers are retained.

These are history-eligibility and data-quality heuristics, not estimates of
forecast reliability. In particular, RF eligibility does not establish
accuracy, moving-average MEDIUM has no validation basis, and any one-period
staleness causes LOW. The rating does not use evaluated forecast errors,
baseline comparisons, forecast horizon, or an evidence count. Warning text
describes stale history, internal gaps, and baseline use, but there is no
per-forecast explanation that ties a confidence rating to out-of-sample
performance.

## Available backtesting evidence

The existing evaluation utilities calculate MAE, RMSE, and WAPE; WAPE is
undefined when total actual demand is zero. The fixed recursive evaluation
in `run_random_forest_experiment.py` scores three future months for its
complete-history cohort. The V2 rolling-origin evaluator in
`forecasting_improvement_experiment_v2.py` fits using history available at
each origin, compares Random Forest to naive and three-month moving-average
baselines, excludes unobserved future actuals from scoring, and reports paired
portfolio block-bootstrap uncertainty. Its own report cautions that only 12
origins are available and the intervals may be unstable.

For this review, that rolling-origin method was run in memory over the
configured source. The source validated at 166,773 rows and 87 products; the
aggregation contained 1,713 observed product-month rows from January 2015
through August 2017. The V2 stable-period design evaluated 54 products across
12 rolling origins and three horizons (1,944 scored product-month rows per
method):

| Method | MAE | Pooled WAPE |
|---|---:|---:|
| Naive | 22.07 | 10.39% |
| Moving Average (3) | 18.65 | 8.78% |
| Random Forest | 18.03 | 8.48% |
| Shrunk daily rate | 15.12 | 7.12% |

Across those 54 products, Random Forest had lower aggregate absolute error
than naive for 50 products and lower aggregate absolute error than moving
average for 33. RF's pooled WAPE improvement over moving average was about
0.30 percentage points. These are comparative results for this historical
cohort, not calibrated confidence probabilities.

### Coverage limitations

- The 12 evaluated origins run from January through December 2016; the
  associated scored targets end in March 2017. The additional raw coverage
  through August 2017 is not represented in those scores.
- Each eligible product has 12 origin evaluations, but this is a small,
  temporally dependent sample with overlapping forecast horizons. The V2
  evaluator explicitly warns that its 12-origin bootstrap intervals may be
  unstable.
- The rolling experiment requires at least 13 consecutive observations.
  Products that are shorter, stale at the forecast cutoff, or have internal
  gaps are not represented by this complete-history cohort. It therefore
  cannot validate the deployed fallback behavior for those cases.
- Existing saved evaluation and forecast artifacts predate the selected
  source and are not used as evidence here.
- Aggregate error differences do not by themselves calibrate a per-product
  confidence label or prove future accuracy.

## Proposed criteria (not implemented)

Keep **method suitability** separate from **confidence**. Continue to report
the selected forecast method (Random Forest, moving average, or naive) as a
method label, not as a confidence tier.

A future evidence-based rule should use paired, leakage-safe rolling-origin
predictions for the same product, target periods, and horizons, compared with
appropriate simple baselines:

- For Random Forest, compare against both naive and three-month moving
  average.
- For moving average, compare against naive.
- For naive, report its measured performance against moving average where
  available; do not treat the method as its own evidence.
- Compare paired error differences, and require the upper bound of a
  predeclared 95% block-bootstrap interval for WAPE difference versus the
  relevant baseline(s) to be below zero before describing a method as
  demonstrably better.
- A middle evidence category should require adequate observed backtest
  coverage and performance that is not demonstrably worse than the relevant
  baseline. A low category should cover supported evidence of worse
  performance. Insufficient evidence must remain explicitly insufficient,
  not be promoted to medium or high.

No defensible numeric minimum number of origins, acceptable WAPE, or
per-product interval width can be selected from the current results. The
available 12-origin design is explicitly flagged as unstable, and it has no
held-out calibration showing that a particular sample count or error
difference predicts future reliability. Therefore no numeric category
thresholds or new labels are activated in this phase.

## Edge-case treatment for a future redesign

- **Short history:** mark backtest evidence insufficient; method selection may
  still use the existing fallback, but fallback eligibility is not
  confidence evidence.
- **Stale history:** show staleness separately and prevent a stale series
  from receiving a strong evidence rating until a recency policy is validated.
- **Internal unknown periods:** never fill absent months with zero. Evaluate
  only forecast targets with observed actuals and preserve observed-run
  boundaries; record excluded unknown targets.
- **Insufficient evaluation data:** expose evidence count and assign an
  insufficient-evidence status. Do not substitute portfolio averages for
  missing per-product outcomes without a separately validated pooling method.
- **Undefined WAPE:** retain MAE/RMSE where defined, mark WAPE unavailable
  when the scored actual total is zero, and do not silently substitute zero.
- **Fallback methods:** evaluate each fallback independently against its
  appropriate simple alternative. Do not infer confidence from the fact that
  a fallback successfully returned a forecast.
- **Reasons:** persist machine-readable evidence count, baseline comparison,
  uncertainty bounds, data-quality caps, and a human-readable explanation
  that matches the assigned status.

## Current dashboard wording

The DataCo forecasting page now states that the current rating is a
data-quality heuristic based on history length, freshness, and gaps, is not
calibrated to backtest performance, and does not guarantee accuracy. The
rating logic and thresholds remain unchanged pending stronger evidence.
