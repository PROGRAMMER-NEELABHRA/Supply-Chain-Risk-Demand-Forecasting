
"""Compare baseline and Random Forest forecasts on matching records."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"

BASELINE_PATH = DATA_DIR / "baseline_predictions.csv"
ADVANCED_PATH = DATA_DIR / "advanced_predictions.csv"
ADVANCED_V2_PATH = DATA_DIR / "advanced_v2_predictions.csv"
OUTPUT_PATH = DATA_DIR / "model_comparison.csv"

KEY_COLUMNS = ["month", "product_id"]
VALUE_COLUMNS = ["actual", "forecast"]


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate a forecast predictions file."""
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    data = pd.read_csv(path)

    required = {"model", *KEY_COLUMNS, *VALUE_COLUMNS}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    data["month"] = data["month"].astype(str)
    data["product_id"] = data["product_id"].astype(str)
    data["actual"] = pd.to_numeric(data["actual"], errors="raise")
    data["forecast"] = pd.to_numeric(data["forecast"], errors="raise")

    if data[list(required)].isna().any().any():
        raise ValueError(f"{path.name} contains missing required values.")

    if (data["actual"] < 0).any() or (data["forecast"] < 0).any():
        raise ValueError(f"{path.name} contains negative demand values.")

    if data.duplicated(["model", *KEY_COLUMNS]).any():
        raise ValueError(
            f"{path.name} contains duplicate model/product/month records."
        )

    return data


def calculate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate MAE, RMSE and WAPE for each model."""
    results = []

    for model_name, group in data.groupby("model"):
        errors = group["actual"] - group["forecast"]
        actual_total = group["actual"].sum()

        results.append(
            {
                "model": model_name,
                "records": len(group),
                "test_months": group["month"].nunique(),
                "MAE": errors.abs().mean(),
                "RMSE": (errors.pow(2).mean()) ** 0.5,
                "WAPE_percent": (
                    errors.abs().sum() / actual_total * 100
                    if actual_total > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(results).sort_values(
        "WAPE_percent"
    ).reset_index(drop=True)


def validate_same_test_records(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_name: str,
) -> None:
    """Ensure predictions use identical product-month keys and actuals."""
    reference_keys = set(
        zip(reference["month"], reference["product_id"])
    )
    candidate_keys = set(
        zip(candidate["month"], candidate["product_id"])
    )

    if reference_keys != candidate_keys:
        raise ValueError(
            f"{candidate_name} does not use the same test records "
            "as the baseline predictions."
        )

    reference_actual = (
        reference.drop_duplicates(KEY_COLUMNS)
        .set_index(KEY_COLUMNS)["actual"]
        .sort_index()
    )
    candidate_actual = (
        candidate.drop_duplicates(KEY_COLUMNS)
        .set_index(KEY_COLUMNS)["actual"]
        .sort_index()
    )

    if not reference_actual.index.equals(candidate_actual.index):
        raise ValueError(
            f"{candidate_name} actual-demand keys do not match."
        )

    if not reference_actual.equals(candidate_actual):
        raise ValueError(
            f"{candidate_name} has different actual-demand values."
        )


def main() -> None:
    """Compare all forecasting models on the same test observations."""
    baseline = load_predictions(BASELINE_PATH)
    advanced = load_predictions(ADVANCED_PATH)
    advanced_v2 = load_predictions(ADVANCED_V2_PATH)

    # Every baseline model must use the same product-month observations.
    baseline_keys = None

    for model_name, group in baseline.groupby("model"):
        keys = set(zip(group["month"], group["product_id"]))

        if baseline_keys is None:
            baseline_keys = keys
        elif keys != baseline_keys:
            raise ValueError(
                f"Baseline model {model_name} uses different test records."
            )

    baseline_reference = baseline[
        ["month", "product_id", "actual"]
    ].drop_duplicates(KEY_COLUMNS)

    validate_same_test_records(
        baseline_reference, advanced, "Original Random Forest"
    )
    validate_same_test_records(
        baseline_reference, advanced_v2, "Enhanced Random Forest"
    )

    combined = pd.concat(
        [
            baseline[["model", *KEY_COLUMNS, *VALUE_COLUMNS]],
            advanced[["model", *KEY_COLUMNS, *VALUE_COLUMNS]],
            advanced_v2[["model", *KEY_COLUMNS, *VALUE_COLUMNS]],
        ],
        ignore_index=True,
    )

    comparison = calculate_metrics(combined)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print("=== FAIR MODEL COMPARISON ===")
    print("Test period: April 2017 to September 2017")
    print("Evaluation: identical product-month records")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print(f"\nComparison saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
