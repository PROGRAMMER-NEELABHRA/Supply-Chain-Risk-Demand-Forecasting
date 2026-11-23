-- Report 6: Check Order Structure Before and After October 2017

SELECT
    strftime('%Y-%m', order_date) AS order_month,
    COUNT(*) AS total_order_items,
    COUNT(DISTINCT order_id) AS unique_orders,

    COUNT(*) - COUNT(DISTINCT order_id)
        AS extra_items_beyond_one_per_order,

    SUM(CASE WHEN quantity > 1 THEN 1 ELSE 0 END)
        AS rows_with_multiple_units,

    COUNT(DISTINCT product_id) AS distinct_products,

    COUNT(DISTINCT order_item_id) AS distinct_item_ids

FROM sales
WHERE order_date >= '2017-08-01'
GROUP BY strftime('%Y-%m', order_date)
ORDER BY order_month;