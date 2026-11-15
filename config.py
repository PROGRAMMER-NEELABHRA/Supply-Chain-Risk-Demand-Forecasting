from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATACO_RAW_DATA_FILE = RAW_DATA_DIR / "integrated_supply_chain_until_aug2017.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FORECAST_OUTPUT_DIR = OUTPUTS_DIR / "forecasts"
RISK_OUTPUT_DIR = OUTPUTS_DIR / "risk"
INVENTORY_OUTPUT_DIR = OUTPUTS_DIR / "inventory"
SCENARIO_OUTPUT_DIR = OUTPUTS_DIR / "scenarios"
CHART_OUTPUT_DIR = OUTPUTS_DIR / "charts"

# Documentation
DOCS_DIR = PROJECT_ROOT / "docs"