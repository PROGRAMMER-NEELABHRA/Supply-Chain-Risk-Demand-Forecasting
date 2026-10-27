
"""
Product data preparation utilities for the DataCo dataset.

Extracts a product-level dataset from the raw order-item data.
The raw source data is never modified.
"""

import pandas as pd


PRODUCT_COLUMNS = [
    "Product Card Id",
    "Product Category Id",
    "Product Name",
    "Product Price",
    "Product Status",
    "Category Id",
    "Category Name",
    "Department Id",
    "Department Name",
]


def prepare_products_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract one row per product from the raw DataCo dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw DataCo dataset.

    Returns
    -------
    pandas.DataFrame
        Product-level dataset with standardized column names.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If required columns are missing or product identifiers
        have conflicting product attributes.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The input dataset is empty.")

    missing_columns = [
        column for column in PRODUCT_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required product columns are missing: {missing_columns}"
        )

    products = data[PRODUCT_COLUMNS].copy()

    products = products.rename(
        columns={
            "Product Card Id": "product_id",
            "Product Category Id": "product_category_id",
            "Product Name": "product_name",
            "Product Price": "product_price",
            "Product Status": "product_status",
            "Category Id": "category_id",
            "Category Name": "category_name",
            "Department Id": "department_id",
            "Department Name": "department_name",
        }
    )

    if products["product_id"].isna().any():
        raise ValueError("Missing product IDs found.")

    # Verify that each product ID maps to one consistent set
    # of product attributes before removing repeated order rows.
    attribute_columns = [
        column for column in products.columns
        if column != "product_id"
    ]

    for column in attribute_columns:
        distinct_counts = products.groupby(
            "product_id"
        )[column].nunique(dropna=False)

        if (distinct_counts > 1).any():
            conflicting_ids = distinct_counts[
                distinct_counts > 1
            ].index.tolist()

            raise ValueError(
                f"Conflicting product attributes in '{column}' "
                f"for product IDs: {conflicting_ids[:10]}"
            )

    products = products.drop_duplicates(subset=["product_id"])
    products = products.sort_values("product_id").reset_index(drop=True)

    return products
