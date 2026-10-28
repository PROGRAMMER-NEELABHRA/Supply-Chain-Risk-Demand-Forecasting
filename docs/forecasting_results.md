# Forecasting Results and Dashboard Handoff

## 1. Overview

The demand-forecasting workflow aggregates order-level data into monthly product demand, evaluates baseline forecasting methods against a Random Forest model, and generates three-month historical forecasts.

The dataset used by the current forecasting pipeline covers **January 2015 through January 2018**. Therefore, the generated forecast period is **February–April 2018**. These are historical forecasts, not current operational predictions.

## 2. Model evaluation

The existing model-comparison report contains the following results:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 21.71 | 25.65 | 33.94% |
| Moving Average | 19.31 | 22.18 | 30.44% |
| Random Forest (existing comparison) | 17.06 | 20.74 | 25.77% |

Lower values indicate lower error for these metrics. In this evaluation, Random Forest performed better than Naive and Moving Average on MAE, RMSE, and WAPE.

A separate tuning comparison identified Random Forest with `max_depth=5` as a provisional configuration. The future-forecast script was then updated to use `max_depth=5` and generated the outputs below. **The table above is from the existing model-comparison report; it should not be presented as the confirmed evaluation of the regenerated depth-5 model unless that exact configuration is evaluated on the same held-out test data.**

- **MAE (Mean Absolute Error):** average absolute prediction error, in demand units.
- **RMSE (Root Mean Squared Error):** error metric that penalizes larger errors more heavily.
- **WAPE (Weighted Absolute Percentage Error):** total absolute error divided by total actual demand, expressed as a percentage.

## 3. Forecast coverage

The latest forecast-generation run used Random Forest with `max_depth=5`, with a common cutoff of January 2018.

| Item | Result |
|---|---:|
| Products in the processed dataset | 118 |
| Products forecasted | 54 |
| Products skipped | 64 |
| Forecast rows generated | 162 |
| Forecast horizon | 3 months per product |
| Forecast months | February–April 2018 |
| Products with data warnings | 54 |

Each forecasted product has three forecast rows, one for each forecast month. The 64 skipped products did not meet the script's minimum requirement of 12 calendar months of history. Their individual skip reasons are recorded in `outputs/forecasts/forecast_products_skipped.csv`.

## 4. Data-quality warning and missing-month assumption

The monthly-demand dataset contains 1,818 recorded product-month rows for 118 products and has no explicit zero-demand rows. The forecast-generation script fills missing product-month records with zero before training the model.

The source documentation available during development does **not** confirm that an absent product-month record means zero demand. The meaning of the missing months therefore remains unverified.

All 54 forecasted products have data warnings because their latest recorded demand month is earlier than the January 2018 forecast cutoff and missing product-month records were treated as zero. Checks against the raw order dataset found no raw orders after each product's last recorded demand month, but this does not establish why orders stopped or prove that the products were discontinued.

For these reasons:

1. Treat the forecasts as experimental historical outputs, not current operational recommendations.
2. Do not label products as discontinued based only on the absence of later recorded orders.
3. Keep `forecast_data_warnings.csv` and `forecast_products_skipped.csv` with the forecast file.
4. Display the forecast period and limitations wherever these forecasts are shown.
5. Validate the source data's completeness and the zero-demand assumption before using forecasts for inventory or purchasing decisions.

## 5. Files for dashboard integration

The following files are prepared in `outputs/handoff/`:

| File | Purpose |
|---|---|
| `future_demand_forecasts.csv` | Forecast values, forecast months, model metadata, and history metadata |
| `forecast_products_skipped.csv` | Products not forecasted and the reason for each skip |
| `forecast_data_warnings.csv` | Product-level warnings about stale history and missing months |
| `random_forest_model_comparison.csv` | Existing comparison of Naive, Moving Average, and Random Forest |

The dashboard should label the forecast period as **February–April 2018** and make the historical nature and data limitations clear. Do not present these forecasts as predictions for the present day.

## 6. Recommended follow-up

- Evaluate the `max_depth=5` Random Forest configuration on the same held-out test data used for the baseline comparison.
- Confirm how missing product-months should be interpreted using authoritative source documentation, if available.
- Preserve the original raw data and retain the warning and skipped-product reports for reproducibility.
