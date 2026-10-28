-- Report 10: Verify Recorded Sales Calculations

SELECT
    strftime('%Y-%m', order_date) AS sales_month,
    COUNT(*) AS total_rows,

    SUM(
        CASE
            WHEN ABS(sales - (unit_price * quantity)) > 0.02
            THEN 1 ELSE 0
        END
    ) AS rows_not_matching_price_times_quantity,

    ROUND(
        AVG(discount),
        4
    ) AS average_discount,

    ROUND(
        AVG(discount_rate),
        4
    ) AS average_discount_rate,

    ROUND(
        SUM(sales),
        2
    ) AS recorded_sales,

    ROUND(
        SUM(unit_price * quantity),
        2
    ) AS calculated_sales_before_discount

FROM sales
WHERE order_date >= '2017-08-01'
GROUP BY strftime('%Y-%m', order_date)
ORDER BY sales_month;
