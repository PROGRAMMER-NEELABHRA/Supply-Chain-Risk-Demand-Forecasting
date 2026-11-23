# Forecasting Results and Dashboard Handoff

## Overview

> **Historical artifact notice:** This report describes outputs from the
> previous source ending in January 2018. The configured DataCo input now ends
> in August 2017, and the pipeline has not yet been rerun. These results are
> not results for the currently selected source.

The pipeline rebuilds monthly demand from the raw transaction CSV, evaluates
three models with a recursive three-month backtest, and generates historical
three-month forecasts. In the previous run, the source ended in January 2018, so future forecast
months are February–April 2018. They are demonstration artifacts, not current
operational predictions.

## Evaluation

The evaluation uses only complete observed product histories. Training ends
in October 2016; the test period is November 2016–January 2017, with 54
products and 162 test product-month rows. For each model, predictions are
recursive: test actuals are recorded for scoring but never fed into subsequent
test predictions.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 23.43 | 46.15 | 10.89% |
| Moving Average | 17.29 | 31.74 | 8.03% |
| Random Forest | 15.55 | 27.17 | 7.22% |

These are pooled metrics across test rows. MAE and RMSE are in demand units;
WAPE is expressed as a percentage. Both evaluation and production forecasting
use the same nine features and Random Forest parameters: 200 estimators,
maximum depth 5, and random state 42.

This test period predates the major source-coverage and quantity-distribution
regime change in late 2017. It does not show that these models are accurate
under that later data regime, and the results should not be described as
operational validation.

## Tiered forecast coverage and quality handling

The prior January 2018 cutoff produced three forecast rows per product with
at least one observed record: 354 rows for the previous 118-product source.
Products with no observed demand were not forecast and were documented in
`outputs/forecasts/forecast_products_skipped.csv` when present in the known
product set.

In that prior run, 54 products used Random Forest, 55 used Moving Average,
and 9 used Naive. Confidence was 0 high, 30 medium, and 324 low forecast rows.
The low-confidence count includes all 108 products whose last actual
observation precedes the January 2018 cutoff; it is not an accuracy estimate.
All 118 products had actual observations in the previous source, so the skipped
products file is empty.

The generator selects one of three clearly labelled methods:

1. **Random Forest (`random_forest_recursive`)** requires a latest run of at
   least 13 consecutive actual monthly observations. This supports the
   existing lag-12 feature and at least one supervised training example. The
   nine existing features and shared configuration (200 estimators, depth 5,
   random state 42) are retained. The three forecast steps are recursive.
2. **Moving Average baseline (`moving_average_baseline`)** is used when the
   RF history requirement is not met but there are at least two observed
   records. It repeats the mean of up to the latest three actual records.
   Missing months are not added to that mean.
3. **Naive baseline (`naive_last_observation_baseline`)** is used for a single
   observed record. It repeats that actual value and is marked low confidence.

These fallback estimates are simple baselines, not Random Forest predictions,
and are not validated by the Random Forest historical backtest.

Every product's `internal_missing_months` counts absent calendar months
strictly between its first and last actual observation. Random Forest training
uses only uninterrupted observed runs; baselines use available actual values
without inserting unknown values. Staleness does not exclude a product:
`months_since_last_record` records the age at the cutoff, stale predictions
are marked `STALE_HISTORY_LOW_CONFIDENCE`, and the warning report explains
that they should not be treated as reliable. Internal gaps and baseline
methods also generate explicit warnings. These outputs are historical
demonstrations, not inventory or purchasing recommendations.

## Data-quality findings

The prior raw dataset had 180,519 transactions, 118 products, and 37 months.
Its monthly rebuild contained 1,818 observed product-months. Of 4,366
combinations in its 118-product by 37-month grid, 2,548 were absent and must
not be assumed to represent zero demand. These counts do not describe the
currently configured source.

The prior report observed changes in monthly product activity and transaction
quantity distributions late in 2017. The currently configured source ends in
August 2017, so this report's later-period finding cannot be verified against
that source. See
`data/quality_review/DATA_QUALITY_REPORT.md` and
`outputs/forecasts/data_quality_summary.csv`.

## Dashboard files

The reproducible pipeline writes these files to `outputs/forecasts/`:

| File | Purpose |
|---|---|
| `future_demand_forecasts.csv` | Product/month forecast values, origin, horizon, model metadata, and history quality |
| `forecast_products_skipped.csv` | Product identifiers and skip reasons |
| `forecast_data_warnings.csv` | Structured warning rows with category, message, last-observed month, staleness, forecast method, and confidence |
| `forecast_warning_summary.csv` | Warning and affected-product counts grouped by warning category |
| `forecast_method_summary.csv` | Product counts by method and forecast-row counts for high/medium/low confidence, including zero-count categories |
| `random_forest_model_comparison.csv` | Pooled model metrics and historical evaluation period |
| `random_forest_predictions.csv` | Test actuals and recursive model predictions |
| `random_forest_per_product_metrics.csv` | Metrics by product and model |
| `data_quality_summary.csv` | Machine-readable source and missingness summary |
| `forecast_demonstration.png` | Observed history and forecasts |
| `model_comparison.png` | Backtest model comparison |

The warning CSV contains one row per warned product. The primary warning
category is mutually exclusive and prioritized as stale history, internal
missing months, then baseline method. `warning_message` retains every
applicable warning detail, including baseline-method information for stale
fallback forecasts. The current counts are 108 stale, 6 baseline-only, and 4
internal-gap-only products. `forecast_warning_summary.csv` groups product and
warning counts by the stable categories `STALE_HISTORY`, `BASELINE_METHOD`,
and `INTERNAL_MISSING_MONTHS`. Forecast consumers can join on `product_id` and
`forecast_month`. Show forecast origin and historical dates, visually
distinguish model predictions from fallback estimates, and keep warnings and
skipped products available to dashboard users. These data cover
February–April 2018, not the current period.

## Reproduce

Install dependencies with `pip install -r requirements.txt`, then run:

```powershell
python run_pipeline.py
```

The pipeline rebuilds the observed input and outputs, runs the evaluation and
forecast stages, generates charts, and stops on a failed stage.
