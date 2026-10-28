-- Report 8: Check Products in the Catalogue
-- Focus on product IDs recorded from October 2017 onward.

SELECT
    s.product_id,
    p.product_name,
    p.product_category_id,
    p.product_status,

    MIN(date(s.order_date)) AS first_recorded_order,
    MAX(date(s.order_date)) AS last_recorded_order,

    COUNT(*) AS order_item_rows,
    SUM(s.quantity) AS total_units,
    ROUND(SUM(s.sales), 2) AS total_sales

FROM sales AS s
LEFT JOIN products AS p
    ON s.product_id = p.product_id

WHERE s.order_date >= '2017-10-01'

GROUP BY
    s.product_id,
    p.product_name,
    p.product_category_id,
    p.product_status

ORDER BY s.product_id;
