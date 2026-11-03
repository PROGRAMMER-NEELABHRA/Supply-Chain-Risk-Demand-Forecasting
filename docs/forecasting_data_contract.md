# Forecasting Data Contract

## Purpose

Define the observed monthly-demand input and output contracts for the
forecasting pipeline.

## Raw source fields

| Field | Description |
|---|---|
| `order_date` | Source transaction date; parsed using the source's day-first format and ISO dates |
| `product_id` | Numeric product identifier |
| `quantity` | Positive order-item quantity |
| `order_item_id` | Optional transaction key; duplicates are rejected when present |

Exact duplicate source rows are rejected. Raw dates and quantities must be
valid and finite. Duplicate product/month rows are expected at transaction
granularity and are aggregated; duplicate product/month rows in the monthly
input are invalid.

## Observed monthly input

The forecasting input has one row per observed product and calendar month:

| Field | Description |
|---|---|
| `month` | Month start (`YYYY-MM-01`) |
| `product_id` | Product identifier |
| `demand_units` | Sum of observed transaction quantities |

The rebuilt input is
`data/quality_review/monthly_demand_observed_rebuilt.csv`. Absent product/month
rows are unknown, not implicit zero demand. Internal gaps are classified and
may be included in separate review files with explicit synthetic labels; they
are excluded from model training and evaluation.

## Forecast output

`outputs/forecasts/future_demand_forecasts.csv` contains one row per forecast
product/month, including `forecast_demand`, `forecast_origin_month`,
`forecast_horizon_month`, model configuration, observed-history metadata, and
`data_quality_status`. Forecast values must be finite and nonnegative.

Every ineligible product appears in `forecast_products_skipped.csv` with a
reason. Data-quality warnings are written separately to
`forecast_data_warnings.csv`; dashboard consumers should surface them rather
than hiding those records.