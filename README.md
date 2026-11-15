# Supply Chain Risk & Demand Forecasting

An end-to-end supply chain analytics and demand forecasting platform built with Python, Streamlit, Pandas, Scikit-learn, and Plotly.

The project combines two workflows:

1. **DataCo historical analysis:** Data validation, monthly demand preparation, forecasting experiments, model evaluation, and supply chain analytics using the historical DataCo dataset.
2. **User-provided demand forecasting:** Upload custom CSV datasets, prepare demand time series, inspect data quality, evaluate forecasting models, and generate forecasts for user-provided data.

The application also includes inventory management, supply chain risk analysis, scenario simulation, and interactive visualizations.

---

## Features

### 1. Interactive Streamlit Dashboard

The application contains eight dashboard pages:

| Page | Functionality |
|---|---|
| Executive Overview | Key supply chain metrics and overall performance |
| Demand Analytics | Historical demand trends and product-level analysis |
| Demand Forecasting | Historical demand forecasts and forecasting results |
| Model Performance | Model comparison and evaluation metrics |
| Inventory Management | Inventory analysis and replenishment-related metrics |
| Supply Chain Risk | Supply chain risk indicators and analysis |
| Scenario Simulator | Explore alternative demand and supply scenarios |
| User Demand Forecasting | Upload custom datasets and perform demand forecasting |

### 2. User-Provided Demand Forecasting

The platform includes a separate workflow for forecasting demand using user-uploaded CSV data.

Key capabilities include:

- Upload external demand datasets.
- Prepare and validate demand data.
- Generate product-level demand time series.
- Inspect data quality and missing-demand patterns.
- Examine forecasting eligibility and product-level diagnostics.
- Evaluate forecasting approaches through backtesting.
- Compare model performance.
- Generate forecasts from the prepared demand history.

The user-demand workflow is implemented separately from the original DataCo pipeline so that custom datasets do not need to replace the historical project dataset.

### 3. Demand Forecasting and Model Evaluation

The historical forecasting workflow includes:

- Naive forecasting.
- Moving Average forecasting.
- Random Forest forecasting.
- Recursive multi-step forecasting.
- Rolling and historical backtesting experiments.
- Model comparison using MAE, RMSE, and WAPE.
- Product-level evaluation.
- Forecast confidence and data-quality diagnostics.
- Identification of products with insufficient or stale historical observations.

Forecasting experiments and evaluation utilities are maintained under `src/forecasting/`.

### 4. Inventory Management

The dashboard includes an inventory analysis module for examining demand and inventory-related metrics.

Implementation: `src/inventory/inventory_analysis.py`

### 5. Supply Chain Risk Analysis

The project includes a risk-scoring module and dashboard for analyzing supply chain risk indicators.

Implementation: `src/risk/risk_score.py`

### 6. Scenario Simulation

The scenario module supports analysis of alternative demand and supply assumptions through the dashboard.

Implementation: `src/scenarios/simulation.py`

These analytical modules should be evaluated against their implementation assumptions before being used for real operational decisions.

---

## Project Architecture

```text
Supply-Chain-Risk-Demand-Forecasting/
│
├── dashboard/
│   ├── app.py
│   ├── operational_data.py
│   ├── user_demand_forecasting.py
│   │
│   ├── components/
│   │   ├── charts.py
│   │   ├── kpi_cards.py
│   │   └── navigation_drawer.py
│   │
│   ├── data/
│   │   └── loader.py
│   │
│   ├── pages/
│   │   ├── 01_executive_overview.py
│   │   ├── 02_demand_analytics.py
│   │   ├── 03_demand_forecasting.py
│   │   ├── 04_model_performance.py
│   │   ├── 05_inventory_management.py
│   │   ├── 06_supply_chain_risk.py
│   │   ├── 07_scenario_simulator.py
│   │   └── 08_user_demand_forecasting.py
│   │
│   └── styles/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── quality_review/
│
├── dataset/
│   └── processed/
│       └── processed/
│           └── dataco/
│
├── docs/
│
├── outputs/
│   └── forecasts/
│
├── sql/
│   ├── queries.sql
│   └── reports/
│
├── src/
│   ├── analysis/
│   ├── data/
│   ├── database/
│   ├── forecasting/
│   ├── inventory/
│   ├── preprocessing/
│   ├── risk/
│   ├── scenarios/
│   └── visualization/
│
├── tests/
│
├── Supply_Chain_PowerBI/
│
├── config.py
├── main.py
├── run_pipeline.py
├── requirements.txt
└── pyproject.toml
```

---

## Dataset and Data Quality

The original DataCo analysis uses an integrated supply chain dataset containing order, product, sales, and logistics information.

The configured forecasting source is:

```text
data/raw/integrated_supply_chain_until_aug2017.csv
```

The configured source contains:

- 166,773 transaction rows.
- 87 products.
- Coverage from January 1, 2015 through August 31, 2017.

The original `integrated_supply_chain.csv` is preserved as a backup.

### Data Quality Approach

The project distinguishes observed demand from missing product-month observations.

An absent product-month is not automatically interpreted as zero demand.

The analysis includes:

- Duplicate record checks.
- Order-item identifier validation.
- Product identifier validation.
- Missing-value analysis.
- Internal missing-month detection.
- Product lifecycle analysis.
- Demand coverage analysis.
- Forecast eligibility diagnostics.
- Stale-history identification.

Quality-review datasets and reports are maintained under `data/quality_review/`.

**Important:** Existing generated outputs may describe an earlier source dataset. They must be regenerated and validated before being interpreted as results from the August 2017 source.

---

## Forecasting Methodology

The original DataCo forecasting experiment compares three approaches:

1. Naive forecasting.
2. Moving Average forecasting.
3. Random Forest forecasting.

The historical experiment uses a three-month forecast horizon.

The Random Forest configuration includes:

- 200 trees.
- Maximum depth of 5.
- Random state of 42.
- Lag features: 1, 2, 3, 6, and 12 months.
- Rolling mean features: 3 and 6 months.
- Month and quarter features.

Forecasting is recursive: each predicted value becomes an input for the next forecast step.

Negative predictions are clipped at zero.

### Historical Backtesting Results

The original documented backtest trains through October 2016 and predicts November 2016 through January 2017 for 54 products.

These are historical results from the previous source dataset, not a validation of forecasts generated from the August 2017 replacement source.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 23.43 | 46.15 | 10.89% |
| Moving Average | 17.29 | 31.74 | 8.03% |
| Random Forest | 15.55 | 27.17 | 7.22% |

MAE and RMSE are measured in demand units. WAPE is expressed as a percentage.

### Forecast Confidence

The forecasting workflow distinguishes between trained model predictions and baseline estimates.

- Random Forest is used when sufficient consecutive historical observations are available.
- Moving Average is used as a fallback for products with limited but usable history.
- Naive forecasting is used for products with only one observed record.

Stale histories and internal missing-month patterns are identified through diagnostics.

A forecast being generated does not automatically mean it is reliable. Confidence indicators and data-quality warnings should be considered when interpreting results.

---

## Forecasting Experiments

Additional forecasting experiments are maintained separately from the original pipeline.

Important modules include:

```text
src/forecasting/
├── forecasting_improvement_experiment.py
├── forecasting_improvement_experiment_v2.py
├── user_demand_backtesting.py
├── user_demand_diagnostics.py
├── user_demand_models.py
├── evaluate_later_period.py
├── evaluate_multiple_products.py
├── evaluate_common_cutoff.py
└── run_random_forest_experiment.py
```

These modules support experimental model evaluation, product-level diagnostics, backtesting, and forecasting improvements.

Experimental results should not automatically be treated as production model performance. Model changes should be evaluated using appropriate historical test periods and leakage-free validation.

---

## Installation

### Requirements

- Python 3.11 or later.
- Git.
- Windows, macOS, or Linux.

### 1. Clone the repository

```bash
git clone https://github.com/PROGRAMMER-NEELABHRA/Supply-Chain-Risk-Demand-Forecasting.git

cd Supply-Chain-Risk-Demand-Forecasting
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

## Run the Streamlit Dashboard

From the repository root:

```bash
streamlit run dashboard/app.py
```

The dashboard provides navigation between all eight analytical pages.

---

## Run the Historical Data Pipeline

The original DataCo pipeline can be executed from the repository root:

```bash
python run_pipeline.py
```

The pipeline coordinates the historical data preparation, forecasting, and visualization stages.

Individual modules can also be executed independently where appropriate.

For example:

```bash
python -m src.analysis.build_monthly_demand

python -m src.forecasting.run_random_forest_experiment

python -m src.forecasting.generate_future_forecasts

python -m src.visualization.forecast_demo
```

Run the pipeline using the configured source and review its outputs before interpreting them as current results.

---

## SQL Analysis

The project includes SQL-based analysis of the DataCo supply chain database.

Location:

```text
sql/
├── queries.sql
└── reports/
    ├── catalogue_coverage_by_month.sql
    ├── check_late_period_products.sql
    ├── check_product_sales_values.sql
    ├── delivery_performance_by_shipping_mode.sql
    ├── demand_data_quality_check.sql
    ├── monthly_product_demand.sql
    ├── monthly_sales_trends.sql
    ├── order_structure_check.sql
    ├── product_ids_by_month.sql
    └── verify_sales_calculation.sql
```

The SQL analysis investigates:

- Product catalogue coverage.
- Monthly sales trends.
- Product-level demand.
- Order structure.
- Sales calculations.
- Product-sales relationships.
- Delivery performance.
- Late delivery risk.
- Data quality.

The database and source CSV copies are local data assets and are not included in the Git repository.

Findings from the investigation are documented in:

```text
docs/sql_data_quality_investigation.md
```

---

## Power BI

The repository also includes a Power BI project:

```text
Supply_Chain_PowerBI/
```

The dashboard project file is:

```text
Supply_Chain_PowerBI/Documentation/dashboard.pbix
```

Prepared data files are maintained under:

```text
Supply_Chain_PowerBI/Data/
```

The Power BI deliverable complements the Python and Streamlit analytical workflows.

---

## Testing

The project uses `pytest` for automated testing.

Run the complete test suite:

```bash
python -m pytest -q
```

**Latest verified test result:**

```text
327 passed
```

This result was obtained after the dependency cleanup and the latest project changes.

Tests cover data validation, preprocessing, forecasting, backtesting, dashboard workflows, inventory analysis, risk scoring, scenario simulation, and related functionality.

---

## Documentation

Additional project documentation is maintained under `docs/`.

Important documents include:

- `PROJECT_CLEANUP_AUDIT.md` — project cleanup review.
- `forecasting_results.md` — historical forecasting results.
- `forecast_confidence_review.md` — forecast confidence investigation.
- `dataco_operations_modules.md` — DataCo operational modules.
- `sql_data_quality_investigation.md` — SQL-based data quality findings.

---

## Limitations and Considerations

- The original DataCo forecasting source ends in August 2017. Forecast dates from this dataset are historical, not live demand predictions.
- Existing outputs may describe an earlier dataset until the pipeline and dependent analyses are regenerated.
- Missing product-month observations cannot automatically be interpreted as zero demand.
- Products with insufficient consecutive history may use baseline forecasting methods rather than trained Random Forest models.
- Stale historical observations reduce forecast reliability.
- User-uploaded datasets must be checked for appropriate date, product, and demand information before forecasting.
- Backtesting performance does not guarantee future forecasting accuracy.
- Inventory and supply chain recommendations should be validated against actual operational data before business use.
- Experimental models and scenario results should not be treated as production-validated decisions without further evaluation.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and analytics |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and forecasting experiments |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualizations |
| Matplotlib | Static charts |
| SQLite | Local database analysis |
| SQL | Data investigation and reporting |
| Power BI | Business intelligence dashboard |
| Pytest | Automated testing |

---

## Project Objective

The objective of this project is to demonstrate an integrated supply chain analytics and demand forecasting workflow, combining historical data analysis, forecasting experiments, data-quality diagnostics, inventory analysis, risk assessment, scenario simulation, and user-provided demand forecasting in an interactive application.

The project emphasizes transparent forecasting evaluation, careful treatment of missing demand, reproducible analysis, and clear communication of model limitations.