
-- 1. Check total rows in each table
SELECT 'products' AS table_name, COUNT(*) AS total_rows
FROM products

UNION ALL

SELECT 'sales', COUNT(*)
FROM sales

UNION ALL

SELECT 'logistics', COUNT(*)
FROM logistics;


-- 2. Check whether sales records have matching products
SELECT
    COUNT(*) AS total_sales_rows,
    COUNT(p.product_id) AS matched_product_rows,
    COUNT(*) - COUNT(p.product_id) AS unmatched_product_rows
FROM sales s
LEFT JOIN products p
    ON s.product_id = p.product_id;


-- 3. Check whether sales records have matching logistics records
SELECT
    COUNT(*) AS total_sales_rows,
    COUNT(l.order_id) AS matched_logistics_rows,
    COUNT(*) - COUNT(l.order_id) AS unmatched_sales_rows
FROM sales s
LEFT JOIN logistics l
    ON s.order_id = l.order_id
    AND s.order_item_id = l.order_item_id;


-- Report 1: Sales and profit by product category
SELECT
    p.category_name,
    COUNT(*) AS total_order_items,
    COUNT(DISTINCT s.order_id) AS unique_orders,
    SUM(s.quantity) AS total_units_sold,
    ROUND(SUM(s.sales), 2) AS total_sales,
    ROUND(SUM(s.order_profit), 2) AS total_profit,
    ROUND(AVG(s.order_profit), 2) AS avg_profit_per_item
FROM sales s
JOIN products p
    ON s.product_id = p.product_id
GROUP BY p.category_name
ORDER BY total_sales DESC;


-- Report 2: Delivery performance by shipping mode
SELECT
    l.shipping_mode,
    COUNT(*) AS total_order_items,
    SUM(
        CASE WHEN l.late_delivery_risk = 1
             THEN 1 ELSE 0 END
    ) AS high_risk_items,
    ROUND(
        100.0 * SUM(
            CASE WHEN l.late_delivery_risk = 1
                 THEN 1 ELSE 0 END
        ) / COUNT(*), 2
    ) AS high_risk_percentage,
    SUM(
        CASE WHEN LOWER(l.delivery_status)
                  LIKE '%late%'
             THEN 1 ELSE 0 END
    ) AS late_delivery_items,
    ROUND(AVG(l.actual_shipping_days), 2)
        AS avg_actual_shipping_days,
    ROUND(AVG(l.scheduled_shipping_days), 2)
        AS avg_scheduled_shipping_days
FROM logistics l
GROUP BY l.shipping_mode
ORDER BY high_risk_percentage DESC;
