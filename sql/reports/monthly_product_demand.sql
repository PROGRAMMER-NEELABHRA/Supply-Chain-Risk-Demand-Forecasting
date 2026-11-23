
-- Report 4: Monthly Demand Summary

SELECT
    strftime('%Y-%m', order_date) AS demand_month,
    COUNT(DISTINCT product_id) AS products_with_demand,
    COUNT(*) AS total_order_items,
    SUM(quantity) AS total_units_demanded,
    ROUND(SUM(sales), 2) AS total_sales
FROM sales
GROUP BY strftime('%Y-%m', order_date)
ORDER BY demand_month;
