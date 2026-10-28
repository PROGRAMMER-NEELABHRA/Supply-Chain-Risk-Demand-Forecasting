### SQL Data Quality Investigation

SQL analysis of the supply-chain database identified a substantial change in product coverage and order structure during the final months of the dataset. The number of distinct products appearing in monthly sales declined from 53 in September 2017 to 20 in October and 8 in November, compared with 118 products in the catalogue.

The sales records also changed structurally. In September 2017, 5,189 sales rows represented 1,723 unique orders. In November 2017, 2,055 sales rows represented 2,055 unique orders, and every recorded row had a quantity of one. Newer product IDs also became more prominent during this period.

A sales arithmetic check found no rows where recorded sales differed from unit price multiplied by quantity for August 2017 through January 2018. Therefore, the main concern is not the arithmetic of the sales values, but the changing product coverage and order structure.

These findings indicate a potential data consistency or source-coverage issue that should be considered when interpreting late-period demand and evaluating forecasting results. The available SQL checks do not establish the cause of the change, so further source-data verification would be needed before treating it as a real business trend.