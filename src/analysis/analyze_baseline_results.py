
"""Analyze baseline model performance across products."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
    / "multi_product_model_comparison.csv"
)


def main() -> None:
    """Summarize model wins and compare baseline errors."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"product_id", "naive_mae", "moving_average_mae"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    df["mae_improvement_pct"] = (
        (df["naive_mae"] - df["moving_average_mae"])
        / df["naive_mae"].replace(0, float("nan"))
        * 100
    )

    moving_wins = (df["moving_average_mae"] < df["naive_mae"]).sum()
    naive_wins = (df["naive_mae"] < df["moving_average_mae"]).sum()
    ties = (df["naive_mae"] == df["moving_average_mae"]).sum()

    print(f"Products evaluated: {len(df)}")
    print(f"Moving-average wins: {moving_wins}")
    print(f"Naive wins: {naive_wins}")
    print(f"Ties: {ties}")

    print("\n=== MAE IMPROVEMENT ===")
    print(df["mae_improvement_pct"].describe().to_string())

    print("\n=== LARGEST MOVING-AVERAGE IMPROVEMENTS ===")
    print(
        df.sort_values("mae_improvement_pct", ascending=False)
        .head(10)[
            ["product_id", "naive_mae", "moving_average_mae",
             "mae_improvement_pct"]
        ]
        .to_string(index=False)
    )

    print("\n=== PRODUCTS WHERE MOVING AVERAGE PERFORMS WORSE ===")
    worse = df[df["moving_average_mae"] > df["naive_mae"]]
    print(worse[
        ["product_id", "naive_mae", "moving_average_mae",
         "mae_improvement_pct"]
    ].head(10).to_string(index=False) or "None")

    print("\n=== TEST PERIOD ENDING AT LAST OBSERVATION ===")
    if "test_ends_at_last_observation" in df.columns:
        print(
            df["test_ends_at_last_observation"]
            .value_counts(dropna=False)
            .to_string()
        )


if __name__ == "__main__":
    main()
