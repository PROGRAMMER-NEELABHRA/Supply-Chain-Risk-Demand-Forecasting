# Supply Chain Delivery Risk Analysis

## 1. Objective

This analysis evaluates delivery performance and identifies potential areas for further investigation using the DataCo integrated dataset. It focuses on shipping delays, shipping modes, geographic patterns, and product-level delivery outcomes.

## 2. Dataset

- Source: Processed DataCo SMART Supply Chain dataset.
- Integrated dataset: 180,519 order-item records.
- Product catalog: 118 products.
- Main fields: actual shipping days, scheduled shipping days, delivery status, supplied late-delivery risk flag, shipping mode, country, region, product ID, and product name.

The analysis operates at the order-item level. An order containing multiple items can therefore contribute multiple records.

## 3. Method

Calculated shipping delay as:

`shipping_delay_days = actual_shipping_days - scheduled_shipping_days`

An item was classified as actually late when its calculated shipping delay was greater than zero.

The supplied `late_delivery_risk` field was analyzed separately from the calculated late indicator. The two measures are not assumed to be identical.

Product-level rankings were also generated. A separate higher-volume product report requires at least 100 order items. Country and regional comparisons in the consolidated report use a minimum of 100 items to reduce the influence of very small samples.

## 4. Overall Findings

Across 180,519 order items:

- 98,977 items were flagged by the supplied late-delivery risk field (54.83%).
- 103,400 items shipped later than their scheduled shipping time (57.28%).
- Average actual shipping time was 3.50 days.
- Average scheduled shipping time was 2.93 days.
- Average actual-minus-scheduled shipping delay was 0.57 days.

These are descriptive statistics for the dataset, not guarantees of future delivery performance.

## 5. Shipping Mode Findings

| Shipping mode | Order items | Actual late rate |
|---|---:|---:|
| First Class | 27,814 | 100.00% |
| Second Class | 35,216 | 79.73% |
| Same Day | 9,737 | 47.83% |
| Standard Class | 107,752 | 39.77% |

The results warrant checking the relationship between shipping-mode definitions, scheduled targets, and actual shipping times. They do not establish that a shipping mode itself causes delays.

## 6. Geographic and Product Findings

Central Africa had the highest calculated late rate among the five largest-rate regions printed in the initial report, at 60.70% across 1,677 items.

Ecuador had a calculated late rate of 69.39% across 294 items in the consolidated report.

Among products with at least 100 order items, Team Golf St. Louis Cardinals Putter Grip had a calculated late rate of 65.18% across 313 items.

These findings are priorities for further investigation. Route mix, shipping mode, data quality, and other confounding factors may contribute to observed differences.

## 7. Limitations

The available data does not directly provide reliable supplier performance, inventory on hand, procurement lead times, reorder points, warehouse stock levels, or external disruption data. Consequently, this analysis should not be described as a complete supplier-risk or inventory-risk model.

The results are historical and descriptive. They identify associations and potential investigation areas, not proven causes of late delivery.

## 8. Reproducibility

The analysis scripts read the integrated dataset and write derived reports to:

`data/processed/dataco/risk_analysis/`

Generated reports include overall delivery risk, risk by shipping mode, region, country, month, and product, as well as a consolidated risk-priority report.

The raw source data and integrated source dataset are not modified by these scripts.