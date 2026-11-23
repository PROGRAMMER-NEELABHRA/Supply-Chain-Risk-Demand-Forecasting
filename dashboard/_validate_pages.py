"""Quick validation script for all dashboard page logic."""
import sys
sys.path.insert(0, '.')

import pandas as pd

print("Loading data modules...")
from dashboard.data.loader import (
    load_monthly_demand, load_future_forecasts, load_forecast_error_summary,
    load_forecast_method_summary, load_forecast_warning_summary,
    load_demand_shift_analysis, load_data_quality_summary,
    load_forecast_data_warnings, load_rf_feature_importance,
    load_forecast_error_by_product, load_later_period_model_summary,
    load_later_period_per_product, load_forecast_eligibility,
)
from dashboard.components.charts import (
    demand_time_series, confidence_donut, method_donut, warning_bar, demand_shift_bar,
    demand_forecast_chart, demand_heatmap, top_products_bar, seasonality_line,
    model_comparison_bar, feature_importance_bar, per_product_error_scatter,
    error_distribution_histogram,
)

print("Loading all datasets...")
demand = load_monthly_demand()
forecasts = load_future_forecasts()
method_summary = load_forecast_method_summary()
warning_summary = load_forecast_warning_summary()
shift_analysis = load_demand_shift_analysis()
quality_summary = load_data_quality_summary()
error_summary = load_forecast_error_summary()
err_by_prod = load_forecast_error_by_product()
feat_imp = load_rf_feature_importance()
later_summary = load_later_period_model_summary()
later_per_prod = load_later_period_per_product()
warnings_df = load_forecast_data_warnings()
eligibility = load_forecast_eligibility()

print("Validating Executive Overview KPIs...")
total_products = demand['product_id'].nunique()
rf_count = int(method_summary.loc[method_summary['category'] == 'random_forest_recursive', 'product_count'].sum())
stale_count = int(warning_summary.loc[warning_summary['warning_category'] == 'STALE_HISTORY', 'product_count'].sum())
quality_score = round((1 - stale_count / total_products) * 100, 1)
total_forecast_demand = forecasts['forecast_demand'].sum()
qs_val = quality_summary.set_index('metric')['value'].to_dict()
raw_tx = int(float(qs_val.get('raw_transaction_records', 0)))
print(f"  products={total_products}, rf={rf_count}, stale={stale_count}, quality={quality_score}%, raw_tx={raw_tx:,}")

print("Generating all charts...")
fig1 = demand_time_series(demand)
fig2 = confidence_donut(method_summary)
fig3 = method_donut(method_summary)
fig4 = warning_bar(warning_summary)
fig5 = demand_shift_bar(shift_analysis)

prod_id = int(forecasts['product_id'].iloc[0])
fig6 = demand_forecast_chart(demand, forecasts, prod_id)
fig7 = top_products_bar(demand)
fig8 = seasonality_line(demand)
fig9 = model_comparison_bar(error_summary, metric='mae')
fig10 = feature_importance_bar(feat_imp)
first_period = err_by_prod[err_by_prod['period'] == 'First period']
fig11 = per_product_error_scatter(first_period)
fig12 = error_distribution_histogram(first_period)
print("  All 12 charts generated OK")

print("Validating Inventory proxy view...")
product_summary = forecasts.groupby('product_id').agg(
    total_forecast_demand=('forecast_demand', 'sum'),
    avg_monthly_forecast=('forecast_demand', 'mean'),
    confidence=('forecast_confidence', 'first'),
    model=('model_name', 'first'),
).reset_index()
last_obs = demand.sort_values('month').groupby('product_id').last()['demand_units'].reset_index()
last_obs.columns = ['product_id', 'last_observed_demand']
product_summary = product_summary.merge(last_obs, on='product_id', how='left')
print(f"  {len(product_summary)} products in proxy view")

print("Validating Risk page...")
total_warned = warnings_df['product_id'].nunique()
risk_pct = round(total_warned / total_products * 100, 1)
print(f"  {total_warned} warned products, risk_pct={risk_pct}%")

print("Validating Scenario simulator...")
agg_base = forecasts.groupby('forecast_month')['forecast_demand'].sum().reset_index()
scenario = agg_base.copy()
scenario['forecast_demand'] = scenario['forecast_demand'] * 1.25
print(f"  Scenario rows: {len(scenario)}")

print("\n=== ALL VALIDATIONS PASSED ===")
