-- Report 11: Product Catalogue Coverage by Month

SELECT
    strftime('%Y-%m', s.order_date) AS sales_month,
    COUNT(DISTINCT s.product_id) AS products_with_sales,
    (SELECT COUNT(DISTINCT product_id) FROM products)
        AS total_catalogue_products,
    COUNT(*) AS sales_rows,
    COUNT(DISTINCT s.order_id) AS unique_orders
FROM sales AS s
WHERE s.order_date >= '2017-08-01'
GROUP BY strftime('%Y-%m', s.order_date)
ORDER BY sales_month;