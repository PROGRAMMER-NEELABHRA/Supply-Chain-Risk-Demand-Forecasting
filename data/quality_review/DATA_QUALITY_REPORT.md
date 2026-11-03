# Data quality analysis and repair report

## Dataset examined
- Raw rows: 180,519
- Columns: 33 original columns (two analysis helper columns were added in memory only)
- Date range: 2015-01-01 00:00:00 to 2018-01-31 23:38:00
- Unique products: 118
- Unique order-item IDs: 180519
- Duplicate full rows: 0
- Missing values in original source columns: 0
- Invalid order dates: 0
- Negative or zero quantities: 0

## Main finding: a major source-data regime change
Monthly activity is stable from Jan 2015 through Mar 2017 at 54 active products per month. It changes sharply from Apr 2017 onward:
- Apr 2017: 87 active products
- May-Aug 2017: 42 active products per month
- Sep 2017: 53
- Oct 2017: 20
- Nov 2017: 8
- Dec 2017: 14
- Jan 2018: 10

The transaction quantity distribution also changes:
- Through Sep 2017, each month has quantity values from 1 to 5.
- Oct 2017: 95.79% of rows have quantity 1.
- Nov 2017-Jan 2018: 100% of rows have quantity 1.

This is evidence of a change in data coverage/recording structure. It is not enough to conclude that demand truly collapsed or that all missing product-months were zero demand.

## Missing-month classification
A full product x month grid contains 4,366 combinations, of which 1,818 are observed product-month combinations. The remaining 2,548 combinations are mostly outside each product's observed lifecycle and must NOT be blindly filled with zero.

Within each product's first-to-last observed month span:
- Internal missing product-months: 10
- Products with at least one internal gap: 10
- Only these 10 enclosed gaps were linearly interpolated in the separate labelled-imputation file.

## What was repaired
1. Rebuilt monthly product demand from the uploaded raw transactions.
2. Created a full product-month panel with missingness explicitly classified.
3. Imputed only enclosed internal gaps using linear interpolation.
4. Preserved original observed demand in a separate column.
5. Labelled every imputed record as synthetic, never observed.

## What was NOT imputed
- Months before a product's first observed transaction.
- Months after a product's last observed transaction.
- The large late-2017/Jan-2018 coverage gaps.
- Products excluded from forecasting for insufficient history, internal gaps, or stale data.

These cannot be truthfully reconstructed from this file alone. They may represent product launch/discontinuation, reporting changes, or missing source records. Confirm against the source system/catalog before treating them as zero demand.

## Recommendation for the model
- Use `monthly_demand_observed_rebuilt.csv` for a real-observation baseline.
- Use the labelled-imputation file only for a clearly marked sensitivity experiment; compare performance with and without those imputed points.
- Do not train or report on the 2,548 full-grid absent combinations as zero demand.
- Do not claim the existing Feb-Apr 2018 forecasts are validated real-world predictions.
- Investigate the Oct 2017 quantity and product-coverage regime change before making operational claims.
- The reproducible workflow now writes `outputs/forecasts/data_quality_summary.csv` and reports all product skip reasons.

## Files
- `monthly_demand_observed_rebuilt.csv`: exact aggregation of source transactions.
- `product_month_quality_panel.csv`: complete grid with explicit missingness status.
- `internal_gaps_imputed_review.csv`: the 10 interpolated internal gaps, labelled synthetic.
- `monthly_demand_observed_plus_labelled_imputation.csv`: observed records plus the labelled internal imputations; includes both original and imputed demand columns.
- `monthly_regime_diagnostics.csv`: monthly data-structure diagnostics.
- `product_lifecycle_summary.csv`: product coverage summary.
