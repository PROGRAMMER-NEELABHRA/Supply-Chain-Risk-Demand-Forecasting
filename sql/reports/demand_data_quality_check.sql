-- Report 5: Demand Data Quality Check
-- Focus on the period when the demand pattern changes.

SELECT
    strftime('%Y-%m', order_date) AS demand_month,
    COUNT(*) AS total_order_items,
    COUNT(DISTINCT product_id) AS products_with_demand,
    COUNT(DISTINCT order_id) AS unique_orders,

    SUM(quantity) AS total_units,
    SUM(CASE WHEN quantity = 1 THEN 1 ELSE 0 END)
        AS items_with_quantity_one,

    MIN(quantity) AS minimum_quantity,
    MAX(quantity) AS maximum_quantity,
    ROUND(AVG(quantity), 2) AS average_quantity,

    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(AVG(sales), 2) AS average_sales_per_item

FROM sales
WHERE order_date >= '2017-03-01'
GROUP BY strftime('%Y-%m', order_date)
ORDER BY demand_month;