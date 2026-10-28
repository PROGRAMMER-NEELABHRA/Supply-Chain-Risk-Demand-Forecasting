-- Report 9: Compare Catalogue Prices with Recorded Sales

SELECT
    s.product_id,
    p.product_name,
    ROUND(p.product_price, 2) AS catalogue_price,

    COUNT(*) AS order_item_rows,
    MIN(s.quantity) AS min_quantity,
    MAX(s.quantity) AS max_quantity,

    ROUND(MIN(s.unit_price), 2) AS min_unit_price,
    ROUND(MAX(s.unit_price), 2) AS max_unit_price,
    ROUND(AVG(s.unit_price), 2) AS avg_unit_price,

    ROUND(SUM(s.sales), 2) AS total_sales,
    ROUND(AVG(s.sales), 2) AS avg_sales_per_row

FROM sales AS s
LEFT JOIN products AS p
    ON s.product_id = p.product_id

WHERE s.order_date >= '2017-10-01'

GROUP BY
    s.product_id,
    p.product_name,
    p.product_price

ORDER BY s.product_id;
