
"""Run the complete supply-chain demand forecasting pipeline."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

STAGES = [
    (
        "BUILD MONTHLY DEMAND",
        [sys.executable, "-m", "src.analysis.build_monthly_demand"],
    ),
    (
        "EVALUATE FORECASTING MODELS",
        [sys.executable, "-m", "src.forecasting.run_random_forest_experiment"],
    ),
    (
        "GENERATE FUTURE FORECASTS",
        [sys.executable, "-m", "src.forecasting.generate_future_forecasts"],
    ),
    (
        "BUILD DEMONSTRATION VISUALIZATIONS",
        [sys.executable, "-m", "src.visualization.forecast_demo"],
    ),
]


def main() -> int:
    print("\nSUPPLY CHAIN DEMAND FORECASTING PIPELINE", flush=True)

    for stage_name, command in STAGES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"RUNNING: {stage_name}", flush=True)
        print("=" * 60, flush=True)

        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

        if result.returncode != 0:
            print(
                f"\nPIPELINE FAILED AT: {stage_name} "
                f"(exit code {result.returncode})",
                flush=True,
            )
            return result.returncode or 1

        print(f"\nCOMPLETED: {stage_name}", flush=True)

    print("\nALL PIPELINE STAGES COMPLETED SUCCESSFULLY.", flush=True)
    print("Check outputs/forecasts/ for generated results.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
