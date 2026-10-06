# Forecasting Data Contract

## Purpose

Define the data required by the demand forecasting pipeline.

## Required Fields

| Field | Description | Required |
|---|---|---|
| date | Date of demand observation | Yes |
| product_id | Unique product identifier | Yes |
| quantity | Demand quantity | Yes |
| region | Sales region | Preferred |
| revenue | Revenue generated | Preferred |

## Expected Granularity

The forecasting dataset should contain one observation
per product per time period.

## Target Variable

quantity

## Forecasting Frequency

To be determined after dataset inspection.

## Forecast Horizon

To be determined after dataset inspection.

## Required Data Quality

- Date must be valid.
- Product ID must not be null.
- Quantity must be numeric.
- Duplicate observations must be investigated.
- Missing demand values must be handled explicitly.

## Notes

Final fields and granularity will be confirmed after
the source dataset is reviewed.