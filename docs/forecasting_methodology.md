# Demand Forecasting Methodology

## 1. Objective

Establish a reproducible baseline for monthly product-demand forecasting and use it as a benchmark for future forecasting models.

## 2. Data Source and Preparation

The forecasting dataset was derived from the cleaned DataCo sales data. Demand was aggregated to the product-month level.

The dataset used for this evaluation covers January 2015 through September 2017. Products were retained according to the existing forecasting-data preparation criteria.

A source-data structure change was identified beginning around October 2017. The period from October 2017 onward was excluded from this initial evaluation because its substantially different order and product patterns could distort model evaluation.

## 3. Forecasting Methods

Three baseline forecasting methods were evaluated:

1. **Previous-month naive:** Forecast demand using the previous month's observed demand.
2. **Three-month moving average:** Forecast demand using the average demand from the previous three months.
3. **Seasonal naive:** Forecast demand using the observed demand from the same month one year earlier.

## 4. Evaluation Design

A chronological rolling one-month-ahead evaluation was conducted over January–September 2017, using the earlier observations as historical data.

Each forecast uses only demand observations available before its target month. Actual demand from a completed month can be used when forecasting the following month.

The evaluation contains 54 product records per month and 486 evaluated product-month records per model.

## 5. Evaluation Metrics

- **MAE:** Mean absolute error, measuring the average absolute difference between actual and predicted demand.
- **RMSE:** Root mean squared error, which penalizes larger errors more heavily.
- **WAPE:** Weighted absolute percentage error, calculated as total absolute error divided by total actual demand, multiplied by 100.

The reported rolling-backtest summary uses the mean of the nine monthly metric values. It is not the same as calculating a single pooled metric over all nine months.

## 6. Rolling Backtest Results

| Model | Mean monthly MAE | Mean monthly RMSE | Mean monthly WAPE |
|---|---:|---:|---:|
| Previous-month naive | 18.45 | 43.70 | 9.35% |
| Three-month moving average | 18.59 | 35.43 | 9.57% |
| Seasonal naive | 29.67 | 42.70 | 15.49% |

## 7. Interpretation and Baseline Selection

The previous-month naive model achieved the lowest mean monthly MAE and WAPE. It is selected as the provisional baseline for comparison with more advanced models.

The three-month moving average achieved the lowest mean monthly RMSE, indicating better performance on that particular aggregate metric. It should therefore remain a comparison model rather than being discarded.

The seasonal naive method performed worse overall in this evaluation.

## 8. Data Quality Limitations

A substantial change in source-data structure was observed from October 2017. Monthly demand and active-product counts declined sharply, while many records became single-item orders with quantity one. The underlying cause has not been established.

Consequently, these initial results should not be interpreted as evidence of forecasting performance under all future demand conditions. The evaluation applies to the selected historical period and product population.

## 9. Reproducibility

The rolling backtest can be reproduced from the project root with:

`python -m src.analysis.rolling_backtest`

The script reads `data/processed/dataco/forecasting_dataset.csv` and writes the monthly evaluation metrics to:

`data/processed/dataco/rolling_backtest_metrics.csv`

## 10. Next Steps

1. Build an advanced forecasting model and compare it against the baseline methods.
2. Evaluate performance using the same chronological test period and clearly defined metrics.
3. Investigate product-level forecast errors and low-volume products.
4. Extend the project to supplier risk, inventory analysis, and scenario simulation using additional appropriate data sources.



## Model Comparison and Findings

### Evaluation setup

Five forecasting models were evaluated on the same 324 product-month observations covering April 2017 through September 2017. The models were trained using data available through March 2017.

The evaluation metrics are:

- **MAE (Mean Absolute Error):** average absolute difference between actual and forecast demand. Lower is better.
- **RMSE (Root Mean Squared Error):** penalizes large forecast errors more heavily. Lower is better.
- **WAPE (Weighted Absolute Percentage Error):** total absolute error divided by total actual demand, expressed as a percentage. Lower is better.

### Results

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive — previous month | 16.80 | 46.23 | 8.96% |
| Moving average — 3 months | 18.55 | 37.53 | 9.89% |
| Random Forest | 31.73 | 41.50 | 16.92% |
| Random Forest with product calendar features | 31.94 | 41.47 | 17.03% |
| Seasonal naive — previous year | 33.85 | 44.16 | 18.05% |

### Interpretation

The naive previous-month model achieved the lowest MAE and WAPE. The three-month moving-average model achieved the lowest RMSE. Neither Random Forest model outperformed these simpler baselines across all evaluation metrics.

These results demonstrate why machine-learning models must be compared with simple baselines rather than assumed to be better because they are more complex. The naive model is a strong candidate when average absolute and demand-weighted percentage errors are the main priorities. The moving-average model is worth considering when reducing large errors is more important.

### Data quality and limitations

An investigation of the DataCo source data found a substantial change in the observed order and product patterns beginning around October 2017. The number of active products and monthly demand decreased sharply. The cause of this change has not been established, so it should not automatically be interpreted as a genuine business-demand decline.

The forecasting dataset therefore uses January 2015 through September 2017 for model development and evaluation. The period after September 2017 should not be treated as directly comparable without further investigation.

The current DataCo dataset does not provide all the information needed for a complete operational supply-chain risk system, such as reliable supplier lead times, procurement records, inventory on hand, reorder points, and external risk indicators. These areas require additional validated data or clearly labelled synthetic scenarios.

### Reproducibility

Forecast predictions and evaluation outputs are stored in `data/processed/dataco/`. The `model_comparison.csv` file contains the comparable model metrics, and `supply_chain_data_workbook.xlsx` consolidates the processed tables and analysis outputs.

Model performance should be re-evaluated when the source data, forecast horizon, or test period changes.
