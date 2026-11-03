# Supply Chain Risk & Demand Forecasting

An end-to-end analytics demonstration that validates integrated order data, builds observed monthly product demand, evaluates recursive forecasts, and exports quality-aware forecasts and charts for dashboard integration.

> **Historical demonstration only:** The source data ends in January 2018. Generated forecasts cover February–April 2018 and are not present-day operational predictions.

## Architecture

```text
data/raw/integrated_supply_chain.csv
  -> validate raw records and aggregate by product/month
  -> data/processed/monthly_demand_observed.csv
  -> data/quality_review/monthly_demand_observed_rebuilt.csv
     + classified product-month panel and labelled imputation review
  -> recursive backtest (Naive, Moving Average, Random Forest)
  -> three-month future forecast, skip/warning reports, charts
```

The workflow is implemented in `src/data/`, `src/analysis/`,
`src/preprocessing/`, `src/forecasting/`, and `src/visualization/`.
`run_pipeline.py` runs the stages in order, stops on the first failure, and
returns a nonzero exit code when a stage fails.

## Dataset and data quality

The input is `data/raw/integrated_supply_chain.csv`, with order date, product
ID, quantity, and transaction identifiers. The validated source contains
180,519 transaction rows, 118 products, and 37 months (January 2015–January
2018). The reproducible aggregation contains 1,818 observed product-month
rows.

The product-by-month grid has 4,366 combinations: 1,818 observed and 2,548
absent. An absent row is not evidence of zero demand. Ten missing months are
inside a product's first-to-last observed span; they are separately
interpolated and labelled synthetic for review. Forecasting and evaluation use
only `monthly_demand_observed_rebuilt.csv`, never those synthetic values.

The source has a major coverage and transaction-quantity regime change in
late 2017. This may indicate a source-recording/coverage change; it does not
prove that demand truly collapsed. See
[`data/quality_review/DATA_QUALITY_REPORT.md`](./data/quality_review/DATA_QUALITY_REPORT.md)
and `outputs/forecasts/data_quality_summary.csv`.

## Install

Use Python 3.11 or later. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the tests:

```powershell
python -m pytest -q
```

## Run the complete workflow

From the repository root:

```powershell
python run_pipeline.py
```

Pipeline stages:

1. Validate source rows and rebuild observed monthly demand and data-quality
   artifacts. Exact duplicate rows, duplicate order-item IDs, invalid dates,
   nonnumeric IDs/quantities, and nonpositive transaction quantities fail
   validation.
2. Evaluate Naive, Moving Average, and Random Forest models.
3. Generate forecasts, skipped-product reasons, and stale-history warnings.
4. Create forecast and model-comparison visualizations.

Individual stages may also be run with
`python -m src.analysis.build_monthly_demand`,
`python -m src.forecasting.run_random_forest_experiment`,
`python -m src.forecasting.generate_future_forecasts`, and
`python -m src.visualization.forecast_demo`, from the repository root.

## Methodology and evaluation

The forecast horizon is three months. Features are the existing lag (1, 2, 3,
6, 12), rolling-mean (3, 6), month-number, and quarter features. The Random
Forest configuration is shared by evaluation and forecasting: 200 trees,
maximum depth 5, random state 42. Forecasts are recursive: each prediction,
not the eventual actual, becomes the next step's input. Negative predictions
are clipped at zero.

The backtest trains through October 2016 and recursively predicts November
2016–January 2017 for 54 products with complete observed history (162 test
rows). Metrics below are pooled across those test rows:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 23.43 | 46.15 | 10.89% |
| Moving Average | 17.29 | 31.74 | 8.03% |
| Random Forest | 15.55 | 27.17 | 7.22% |

MAE and RMSE are in demand units; WAPE is a percentage. These historical
results do not prove accuracy after the late-2017 data regime change and
should not be presented as validation of operational performance.

## Forecast results and limitations

The current reproducible run uses January 2018 as the forecast origin and
and produces three forecast rows for each product with at least one actual
observation (354 rows for the current 118-product source). Forecasting has
three explicit tiers:

1. **Random Forest:** used only when a product has at least 13 consecutive
   actual monthly observations to support the lag-12 features and a supervised
   training row. It uses the existing nine features, 200 trees, depth 5, and
   random state 42, then forecasts recursively for three months.
2. **Moving Average baseline:** products without a qualifying recent
   consecutive run but with at least two observed records use the mean of up
   to their latest three actual records. This is a baseline estimate, not a
   Random Forest prediction.
3. **Naive baseline:** a product with one observed record repeats its last
   actual value. It is explicitly marked low confidence.

All fallback methods use observed records only; missing months are not
inserted into their histories. `internal_missing_months` records gaps within
each product's first-to-last observed span. A stale product is not excluded:
`months_since_last_record` is retained, confidence is lowered, and the status
is `STALE_HISTORY_LOW_CONFIDENCE`. Stale and internal-gap warnings are written
to the warnings file. Stale predictions should not be considered reliable.
Products with no observed records cannot be forecast and are reported in the
skip file when they can be identified from the source product set.

The current data run uses Random Forest for 54 products, moving-average
baselines for 55, and naive baselines for 9. There are 0 high-, 30 medium-,
and 324 low-confidence forecast rows: the most recent observations are stale
for 108 products, so those products are explicitly downgraded. The method
summary CSV includes zero-count confidence categories as well as actual
product and forecast-row counts.

The warning report assigns one primary category per product to make category
counts mutually exclusive: current counts are 108 `STALE_HISTORY`, 6
`BASELINE_METHOD`, and 4 `INTERNAL_MISSING_MONTHS`. Where warnings overlap,
`warning_message` retains all applicable details, while `warning_category`
uses the priority stale history, then internal gaps, then baseline method.

The output files are under `outputs/forecasts/`:

| File | Contents |
|---|---|
| `future_demand_forecasts.csv` | Product/month forecasts, method, confidence, quality status, observed-history metadata, origin, and horizon |
| `forecast_products_skipped.csv` | Products without observed history and reason |
| `forecast_data_warnings.csv` | One structured row per warned product with primary category, all applicable warning details, and product forecast metadata |
| `forecast_warning_summary.csv` | Mutually exclusive warned-product counts grouped by primary warning category |
| `forecast_method_summary.csv` | Product counts by method and forecast-row counts for high/medium/low confidence |
| `random_forest_model_comparison.csv` | Pooled MAE, RMSE, WAPE, and evaluation-period metadata for all three models |
| `random_forest_predictions.csv` | Actuals and recursive test predictions |
| `random_forest_per_product_metrics.csv` | Metrics by product and model |
| `data_quality_summary.csv` | Source, coverage, gap, imputation, and regime-change summary |
| `forecast_demonstration.png` | Observed history and forecasts for selected products |
| `model_comparison.png` | Backtest metric comparison |

Dashboard consumers should retain skip and warning files alongside forecasts,
show the forecast origin/period, and visually distinguish Random Forest model
predictions from moving-average and naive baseline estimates. Fallback
estimates are not validated by the Random Forest backtest. The data does not
establish whether absent months represent zero demand, product lifecycle
changes, or incomplete recording.

## Dashboard and other project resources

The CSV outputs use one row per product/month for forecasts and predictions;
the comparison and quality-summary files are long-form/tabular reports. They
are generated directly from the pipeline and require no dashboard-specific
transformation. The dashboard can join forecast rows on `product_id` and
`forecast_month`, and surface `data_quality_status`, the skip report, and
warning report.

Additional analysis scripts are in `src/analysis/`; SQL investigations and
queries are in `sql/` and `docs/`. A SQLite database also exists under
`dataset/processed/processed/dataco/`; confirm its relationship to the
integrated raw CSV before using it as a pipeline source.

## Limitations

- The source ends in January 2018; forecast dates are historical.
- The evaluation period predates the late-2017 source-data regime change.
- Most product-month combinations are absent and cannot safely be assumed to
  be zero.
- Random Forest coverage is limited by observed consecutive history; the
  lower-history tiers are explicit baselines rather than trained models.
- Stale predictions have low confidence; they are generated for coverage,
  not claimed as reliable forecasts.
- Validate source completeness and forecast suitability before inventory,
  purchasing, or other operational decisions.
