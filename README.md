# Supply Chain Risk & Demand Forecasting

An analytics project for exploring supply-chain data, preparing product-level demand histories, evaluating demand-forecasting models, and producing historical forecasts that can support a supply-chain risk dashboard.

> **Important:** The forecasts currently available in this repository cover **February–April 2018** using data ending in **January 2018**. They are historical demonstration forecasts, not current operational predictions.

## Project goals

- Integrate and validate supply-chain data.
- Aggregate order records into monthly product demand.
- Explore demand history and data-quality issues.
- Compare baseline forecasting methods with a Random Forest model.
- Generate three-month forecasts for products with sufficient history.
- Provide documented CSV outputs for dashboard development.
- Develop SQL-based data-quality and demand-analysis queries.

## Current project status

| Area | Status |
|---|---|
| Raw integrated dataset | Available |
| Monthly product-demand dataset | Available |
| Demand feature engineering | Implemented |
| Naive and Moving Average baselines | Implemented |
| Random Forest forecasting and tuning | Implemented |
| Forecast evaluation and analysis scripts | Available |
| Historical forecast outputs | Generated |
| Automated tests | Available |
| SQL reports and data-quality queries | Implemented |
| Dashboard | Maintained by the dashboard teammate |

## Repository structure

```text
Supply-Chain-Risk-Demand-Forecasting/
├── data/
│   ├── raw/
│   │   └── integrated_supply_chain.csv
│   ├── processed/
│   │   ├── monthly_demand_observed.csv
│   │   └── forecast_product_shortlist.csv
│   └── external/
├── dataset/
│   └── processed/
│       └── processed/
│           └── dataco/
│               └── supply_chain.db
├── docs/
│   └── forecasting_data_contract.md
├── outputs/
│   ├── forecasts/
│   └── handoff/
├── reports/
├── sql/
│   ├── schema/
│   ├── queries/
│   └── reports/
├── src/
│   ├── analysis/
│   ├── forecasting/
│   ├── preprocessing/
│   └── ...
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

The repository also contains additional analysis scripts and generated CSVs under `src/analysis/` and `outputs/forecasts/`.

**Note on the database:** A SQLite database is present under `dataset/processed/processed/dataco/`. Its relationship to the current `data/raw/integrated_supply_chain.csv` and monthly-demand pipeline should be checked before treating it as the project's primary database. The `sql/` directory contains SQL reports and queries for demand trends, delivery performance, data quality, and product coverage. The findings from the data-quality investigation are documented in `docs/sql_data_quality_investigation.md`.

## Data overview

### Raw input

`data/raw/integrated_supply_chain.csv`

The integrated dataset used by the current demand pipeline contains order-level records. Relevant fields include:

- `order_date`
- `product_id`
- `quantity`

The source data spans **January 2015 through January 2018**, with **180,519 rows** and **118 unique products** in the checks run during development.

### Processed monthly demand

`data/processed/monthly_demand_observed.csv`

This file is created by aggregating order quantities by product and calendar month. Its columns are:

| Column | Meaning |
|---|---|
| `month` | Month represented as a date |
| `product_id` | Product identifier |
| `demand_units` | Sum of recorded order quantity for that product and month |

The current processed file contains **1,818 recorded product-month rows**, covering January 2015 through January 2018, for 118 products.

The file contains only product-months with recorded orders; it has no explicit zero-demand rows. Therefore, an absent product-month row must **not automatically be interpreted as confirmed zero demand** without verifying the source data's completeness.

## Environment setup

Use Python 3.11 or later.

### 1. Create and activate a virtual environment

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python directly with `.venv\Scripts\python.exe`.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run tests

```powershell
python -m pytest
```

The project had 19 passing tests during the latest reported development check. Run the tests again after changes to confirm the current result.

## Demand forecasting workflow

Run commands from the repository root.

### 1. Build monthly demand

```powershell
python -m src.analysis.build_monthly_demand
```

This reads `data/raw/integrated_supply_chain.csv` and writes `data/processed/monthly_demand_observed.csv`.

### 2. Run forecasting evaluation and experiments

The `src/forecasting/` folder contains scripts for baseline experiments, Random Forest experiments, tuning, and evaluation. Review each script before running it, as some commands regenerate output files.

Examples:

```powershell
python -m src.forecasting.run_baseline_experiment
python -m src.forecasting.run_random_forest_experiment
python -m src.forecasting.tune_random_forest
```

### 3. Generate historical forecasts

```powershell
python -m src.forecasting.generate_future_forecasts
```

This writes:

- `outputs/forecasts/future_demand_forecasts.csv`
- `outputs/forecasts/forecast_products_skipped.csv`
- `outputs/forecasts/forecast_data_warnings.csv`

The current forecast-generation script uses a common cutoff of January 2018 and generates forecasts for February, March, and April 2018. The current model setting is Random Forest with `max_depth=5`.

## Forecasting evaluation

The existing baseline comparison reports the following metrics:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 21.71 | 25.65 | 33.94% |
| Moving Average | 19.31 | 22.18 | 30.44% |
| Random Forest (existing comparison) | 17.06 | 20.74 | 25.77% |

Lower values indicate lower error for these metrics. In this evaluation, Random Forest outperformed Naive and Moving Average.

A separate tuning comparison indicated that a shallower Random Forest (`max_depth=5`) was a provisional candidate. **Do not present the metrics above as the confirmed evaluation of the regenerated depth-5 model unless that exact configuration has been evaluated on the same held-out test data.**

- **MAE:** mean absolute error, in the target's units.
- **RMSE:** root mean squared error, which penalizes larger errors more heavily.
- **WAPE:** weighted absolute percentage error, expressed as a percentage.

## Forecast coverage and limitations

The latest forecast-generation run produced:

- **54 products forecasted**
- **162 forecast rows** (three months per forecasted product)
- **64 products skipped** because they did not meet the minimum 12-calendar-month history requirement
- **54 products with data warnings**

The forecast-generation code fills missing product-months with zero before training. In the current dataset, the monthly-demand file has no explicit zero-demand rows, and the source documentation does not confirm that absent months mean zero demand. All 54 forecasted products also have last recorded demand dates earlier than the January 2018 cutoff.

Consequently:

1. Treat the forecasts as **experimental historical outputs**, not current operational recommendations.
2. Do not infer that a product is discontinued solely because its recorded orders stop.
3. Keep the warning and skipped-product reports with the forecast CSV.
4. Document the zero-demand assumption clearly wherever results are displayed.
5. Validate source-data completeness before using the forecasts for inventory or purchasing decisions.

## Dashboard handoff

The following files are prepared under `outputs/handoff/` for dashboard integration:

- `future_demand_forecasts.csv` — historical forecast values and model metadata.
- `forecast_products_skipped.csv` — products not forecasted and reasons.
- `forecast_data_warnings.csv` — product-level data warnings.
- `random_forest_model_comparison.csv` — baseline model comparison metrics.

The dashboard should show the forecast period and include the data limitations above. These files represent historical results ending in April 2018, not present-day demand forecasts.

## SQL work planned

The `sql/` directory is reserved for SQL deliverables. The SQL implementation is not yet complete. Planned files include:

```text
sql/
├── schema/
│   └── 01_create_tables.sql
├── queries/
│   ├── 01_data_quality_checks.sql
│   └── 02_monthly_demand_analysis.sql
└── reports/
```

Planned SQL tasks:

- Define tables and data types for the selected dataset.
- Check missing values, duplicate keys, and date coverage.
- Summarize monthly demand by product.
- Identify product histories with long gaps or stale last-order dates.
- Document query outputs for reproducibility.

Before creating or loading another database, confirm whether the existing SQLite database can be reused and whether its tables match the current integrated dataset.

## Reproducibility and safe data handling

- Run Python commands from the repository root.
- Keep raw input data unchanged.
- Treat generated CSVs under `outputs/` as reproducible outputs.
- Review differences before overwriting existing reports.
- Do not remove legacy datasets or database files until their use has been confirmed.

## Project disclaimer

This project is an analytical and forecasting prototype. Forecast quality depends on data completeness, evaluation design, and whether historical demand patterns remain relevant. Results should be independently validated before operational use.
