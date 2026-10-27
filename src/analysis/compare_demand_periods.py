
"""Compare demand patterns across periods around the source-data shift."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"
PROFILE_PATH = DATA_DIR / "monthly_demand_profile.csv"


def summarize_period(name: str, data: pd.DataFrame) -> dict:
    """Calculate descriptive statistics for a monthly demand period."""
    return {
        "period": name,
        "months": len(data),
        "total_units": int(data["total_units"].sum()),
        "average_monthly_units": round(data["total_units"].mean(), 2),
        "minimum_monthly_units": int(data["total_units"].min()),
        "maximum_monthly_units": int(data["total_units"].max()),
        "average_active_products": round(
            data["products_with_recorded_demand"].mean(), 2
        ),
        "average_order_item_rows": round(
            data["order_item_rows"].mean(), 2
        ),
    }


def main() -> None:
    """Print a comparison of candidate forecasting periods."""
    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(f"Demand profile not found: {PROFILE_PATH}")

    data = pd.read_csv(PROFILE_PATH)
    data["month"] = pd.PeriodIndex(data["month"], freq="M")

    periods = [
        ("Full period", data),
        (
            "Before source shift",
            data[data["month"] <= pd.Period("2017-09", freq="M")],
        ),
        (
            "After source shift",
            data[data["month"] >= pd.Period("2017-10", freq="M")],
        ),
    ]

    summaries = [
        summarize_period(name, subset)
        for name, subset in periods
    ]

    result = pd.DataFrame(summaries)

    print("=== FORECASTING PERIOD COMPARISON ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
