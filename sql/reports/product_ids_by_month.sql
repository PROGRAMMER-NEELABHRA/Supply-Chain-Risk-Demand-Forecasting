-- Report 7: Product IDs Recorded Each Month

SELECT
    strftime('%Y-%m', order_date) AS demand_month,
    COUNT(DISTINCT product_id) AS product_count,
    GROUP_CONCAT(DISTINCT product_id) AS product_ids
FROM sales
WHERE order_date >= '2017-08-01'
GROUP BY strftime('%Y-%m', order_date)
ORDER BY demand_month;