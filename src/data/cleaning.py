
"""
Cleaning and validation utilities for the DataCo supply chain dataset.

The raw dataset is never modified. These functions prepare a
separate order-item-level sales dataset for downstream analysis
and demand forecasting.
"""

import pandas as pd


# Raw columns required for the initial sales dataset.
SALES_COLUMNS = [
    "Order Id",
    "Order Item Id",
    "order date (DateOrders)",
    "Order Item Cardprod Id",
    "Order Customer Id",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Order Region",
]


# Mapping from raw column names to standardized column names.
SALES_COLUMN_MAPPING = {
    "Order Id": "order_id",
    "Order Item Id": "order_item_id",
    "order date (DateOrders)": "order_date",
    "Order Item Cardprod Id": "product_id",
    "Order Customer Id": "customer_id",
    "Order Item Quantity": "quantity",
    "Order Item Product Price": "unit_price",
    "Order Item Discount": "discount",
    "Order Item Discount Rate": "discount_rate",
    "Sales": "sales",
    "Order Item Total": "order_item_total",
    "Order Profit Per Order": "order_profit",
    "Order Region": "order_region",
}


def prepare_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare order-item-level sales data for analysis.

    Selects relevant columns, standardizes their names, converts
    order dates to datetime, and checks essential fields.

    The input DataFrame is not modified, and rows are not aggregated.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw DataCo dataset.

    Returns
    -------
    pandas.DataFrame
        Prepared sales dataset.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If required columns are missing or essential values
        are invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The input dataset is empty.")

    missing_columns = [
        column
        for column in SALES_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Select and copy only the columns needed for sales analysis.
    sales = data[SALES_COLUMNS].copy()

    # Standardize column names.
    sales = sales.rename(columns=SALES_COLUMN_MAPPING)

    # Convert dates; invalid dates become NaT and are checked below.
    sales["order_date"] = pd.to_datetime(
        sales["order_date"],
        format="mixed",
        errors="coerce",
    )

    if sales["order_date"].isna().any():
        invalid_count = int(sales["order_date"].isna().sum())
        raise ValueError(
            f"{invalid_count} order dates are missing or invalid."
        )

    # Check required fields for missing values.
    essential_columns = [
        "order_id",
        "order_item_id",
        "product_id",
        "customer_id",
        "quantity",
    ]

    if sales[essential_columns].isna().any().any():
        missing_counts = (
            sales[essential_columns].isna().sum()
        )
        missing_counts = missing_counts[missing_counts > 0].to_dict()

        raise ValueError(
            f"Missing values in essential fields: {missing_counts}"
        )

    # Validate numeric fields before applying business rules.
    numeric_columns = [
        "quantity",
        "unit_price",
        "discount",
        "discount_rate",
        "sales",
        "order_item_total",
        "order_profit",
    ]

    for column in numeric_columns:
        sales[column] = pd.to_numeric(
            sales[column],
            errors="coerce",
        )

    invalid_numeric_counts = (
        sales[numeric_columns].isna().sum()
    )
    invalid_numeric_counts = (
        invalid_numeric_counts[invalid_numeric_counts > 0]
    )

    if not invalid_numeric_counts.empty:
        raise ValueError(
            "Missing or invalid numeric values: "
            f"{invalid_numeric_counts.to_dict()}"
        )

    if (sales["quantity"] <= 0).any():
        raise ValueError(
            "Non-positive quantities found; investigate the source data."
        )

    if (sales["unit_price"] < 0).any():
        raise ValueError("Negative unit prices found.")

    if sales["order_item_id"].duplicated().any():
        raise ValueError("Repeated order_item_id values found.")

    return sales


def validate_sales_data(data: pd.DataFrame) -> None:
    """
    Validate essential business rules for prepared sales data.

    This function checks the dataset without modifying it.

    Parameters
    ----------
    data : pandas.DataFrame
        Prepared sales dataset.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If the dataset is empty or fails validation checks.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Processed sales data is empty.")

    required_columns = [
        "order_id",
        "order_item_id",
        "order_date",
        "product_id",
        "customer_id",
        "quantity",
        "unit_price",
        "sales",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required processed columns are missing: {missing_columns}"
        )

    if data["order_item_id"].isna().any():
        raise ValueError("Missing order item IDs found.")

    if data["order_item_id"].duplicated().any():
        raise ValueError("Duplicate order item IDs found.")

    if data["order_date"].isna().any():
        raise ValueError("Missing or invalid order dates found.")

    if data["product_id"].isna().any():
        raise ValueError("Missing product IDs found.")

    if data["quantity"].isna().any() or (data["quantity"] <= 0).any():
        raise ValueError(
            "Missing or non-positive quantities found."
        )

    if data["unit_price"].isna().any() or (data["unit_price"] < 0).any():
        raise ValueError(
            "Missing or negative unit prices found."
        )

    if data["sales"].isna().any():
        raise ValueError("Missing sales values found.")
