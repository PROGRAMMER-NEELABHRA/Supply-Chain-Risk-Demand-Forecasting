
-- Report 2: Delivery Performance by Shipping Mode

SELECT
    l.shipping_mode,
    COUNT(*) AS total_order_items,

    SUM(
        CASE
            WHEN l.late_delivery_risk = 1 THEN 1
            ELSE 0
        END
    ) AS high_risk_items,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN l.late_delivery_risk = 1 THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS high_risk_percentage,

    SUM(
        CASE
            WHEN LOWER(l.delivery_status) LIKE '%late%'
            THEN 1
            ELSE 0
        END
    ) AS late_delivery_items,

    ROUND(AVG(l.actual_shipping_days), 2)
        AS avg_actual_shipping_days,

    ROUND(AVG(l.scheduled_shipping_days), 2)
        AS avg_scheduled_shipping_days

FROM logistics AS l

GROUP BY l.shipping_mode

ORDER BY high_risk_percentage DESC;
