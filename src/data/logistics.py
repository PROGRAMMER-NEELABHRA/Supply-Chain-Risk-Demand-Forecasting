
"""
Logistics data preparation utilities for the DataCo dataset.

Extracts shipping, delivery, and destination information without
modifying the original raw dataset.
"""

import pandas as pd


LOGISTICS_COLUMNS = [
    "Order Id",
    "Order Item Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Late_delivery_risk",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Order City",
]


LOGISTICS_COLUMN_MAPPING = {
    "Order Id": "order_id",
    "Order Item Id": "order_item_id",
    "order date (DateOrders)": "order_date",
    "shipping date (DateOrders)": "shipping_date",
    "Days for shipping (real)": "actual_shipping_days",
    "Days for shipment (scheduled)": "scheduled_shipping_days",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Shipping Mode": "shipping_mode",
    "Market": "market",
    "Order Region": "order_region",
    "Order Country": "order_country",
    "Order State": "order_state",
    "Order City": "order_city",
}


def prepare_logistics_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and validate order-item-level logistics data.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw DataCo dataset.

    Returns
    -------
    pandas.DataFrame
        Prepared logistics dataset.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If required columns or essential values are invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The input dataset is empty.")

    missing_columns = [
        column
        for column in LOGISTICS_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required logistics columns are missing: {missing_columns}"
        )

    logistics = data[LOGISTICS_COLUMNS].copy()
    logistics = logistics.rename(columns=LOGISTICS_COLUMN_MAPPING)

    # Convert both dates, accounting for the mixed raw date formats.
    for column in ["order_date", "shipping_date"]:
        logistics[column] = pd.to_datetime(
            logistics[column],
            format="mixed",
            errors="coerce",
        )

        if logistics[column].isna().any():
            invalid_count = int(logistics[column].isna().sum())
            raise ValueError(
                f"{invalid_count} missing or invalid values in {column}."
            )

    # Convert shipping durations to numeric values.
    numeric_columns = [
        "actual_shipping_days",
        "scheduled_shipping_days",
        "late_delivery_risk",
    ]

    for column in numeric_columns:
        logistics[column] = pd.to_numeric(
            logistics[column],
            errors="coerce",
        )

        if logistics[column].isna().any():
            raise ValueError(
                f"Missing or invalid numeric values in {column}."
            )

    # Check essential identifiers and delivery information.
    essential_columns = [
        "order_id",
        "order_item_id",
        "delivery_status",
        "shipping_mode",
    ]

    if logistics[essential_columns].isna().any().any():
        raise ValueError(
            "Missing values found in essential logistics fields."
        )

    if logistics["order_item_id"].duplicated().any():
        raise ValueError("Duplicate order item IDs found in logistics data.")

    if (logistics["actual_shipping_days"] < 0).any():
        raise ValueError("Negative actual shipping durations found.")

    if (logistics["scheduled_shipping_days"] < 0).any():
        raise ValueError("Negative scheduled shipping durations found.")

    return logistics
