
-- Report 3: Monthly Sales Trends

SELECT
    strftime('%Y-%m', order_date) AS sales_month,
    COUNT(*) AS total_order_items,
    COUNT(DISTINCT order_id) AS unique_orders,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(order_profit), 2) AS total_profit

FROM sales

GROUP BY strftime('%Y-%m', order_date)

ORDER BY sales_month;
