
"""Machine-learning models for demand forecasting."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


RANDOM_FOREST_N_ESTIMATORS = 200
RANDOM_FOREST_MAX_DEPTH = 5
RANDOM_FOREST_RANDOM_STATE = 42


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = RANDOM_FOREST_N_ESTIMATORS,
    max_depth: int = RANDOM_FOREST_MAX_DEPTH,
    random_state: int = RANDOM_FOREST_RANDOM_STATE,
) -> RandomForestRegressor:
    """Train a Random Forest regression model on demand features."""
    if X_train.empty or y_train.empty:
        raise ValueError("Training features and target cannot be empty.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal lengths.")

    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("Training data contains missing values.")

    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model
