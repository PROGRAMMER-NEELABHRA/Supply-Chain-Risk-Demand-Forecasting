"""
Supply Chain Risk & Demand Forecasting — Platform Controller & Navigation.

Enterprise Streamlit multi-page application using modern st.navigation and st.Page.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path so 'dashboard' imports resolve everywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.components.app_header import render_app_header
from dashboard.styles.theme import inject_css

# Page configuration (MUST be executed once at app root)

st.set_page_config(
    page_title="Supply Chain Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/PROGRAMMER-NEELABHRA/Supply-Chain-Risk-Demand-Forecasting#readme",
        "Report a bug": "https://github.com/PROGRAMMER-NEELABHRA/Supply-Chain-Risk-Demand-Forecasting/issues",
        "About": (
            "**Supply Chain Risk & Demand Forecasting**\n\n"
            "An analytical platform for historical demand review, forward forecasting, "
            "data-quality risk indicators, inventory planning, model evaluation, "
            "and deterministic scenario analysis.\n\n"
            "**Version:** 0.1.0\n\n"
            "**Technology:** Python 3.11+, Streamlit, Pandas, NumPy, "
            "scikit-learn, Plotly\n\n"
            "The platform supports the DataCo demonstration dataset and "
            "user-uploaded demand CSVs prepared in the User Demand Forecasting page.\n\n"
            "**Page guide**\n\n"
            "- **Executive Overview:** Portfolio demand, forecast commitments, "
            "data-quality KPIs, and provenance.\n"
            "- **Demand Analytics:** Monthly demand, SKU comparisons, seasonality, "
            "and demand regime changes.\n"
            "- **Demand Forecasting:** Selected-SKU forecasts, confidence ratings, "
            "data-quality warnings, and the forecast register.\n"
            "- **Model Performance:** Benchmark metrics, feature importance, "
            "per-product errors, and rolling evaluation.\n"
            "- **Inventory Management:** Stock depletion, safety stock, reorder "
            "points, and EOQ using explicitly supplied inputs.\n"
            "- **Supply Chain Risk:** Demand-history and forecast indicators plus "
            "the series-level reliability register.\n"
            "- **Scenario Simulator:** Compare deterministic optimistic, "
            "pessimistic, and user-scaled forecasts.\n"
            "- **User Demand Forecasting:** Upload a CSV, map date/demand/product "
            "columns, validate preparation, compare models, and generate forecasts.\n\n"
            "**Get help**\n\n"
            "Use the sidebar navigation and apply each page's filters before "
            "interpreting charts or tables. Hover over chart traces and use legends "
            "for additional detail.\n\n"
            "**Report a problem**\n\n"
            "Include the affected page, the steps needed to reproduce the issue, "
            "and any error message or unexpected result. Describe the data source "
            "and filters in use, without including sensitive or private CSV contents."
        ),
    },
)

# Render native toolbar branding before page content, then apply shared styles.
render_app_header()
inject_css()

# Define navigation structure with st.Page
overview_pages = [
    st.Page(
        "pages/01_executive_overview.py",
        title="Executive Overview",
        icon=":material/dashboard:",
    ),
]

intelligence_pages = [
    st.Page(
        "pages/02_demand_analytics.py",
        title="Demand Analytics",
        icon=":material/analytics:",
        url_path="demand_analytics",
    ),
    st.Page(
        "pages/03_demand_forecasting.py",
        title="Demand Forecasting",
        icon=":material/trending_up:",
        url_path="demand_forecasting",
    ),
    st.Page(
        "pages/04_model_performance.py",
        title="Model Performance",
        icon=":material/tune:",
        url_path="model_performance",
    ),
    st.Page(
        "pages/05_inventory_management.py",
        title="Inventory Management",
        icon=":material/inventory_2:",
        url_path="inventory_management",
    ),
]

operations_pages = [
    st.Page(
        "pages/06_supply_chain_risk.py",
        title="Supply Chain Risk",
        icon=":material/warning:",
        url_path="supply_chain_risk",
    ),
    st.Page(
        "pages/07_scenario_simulator.py",
        title="Scenario Simulator",
        icon=":material/science:",
        url_path="scenario_simulator",
    ),
    st.Page(
        "pages/08_user_demand_forecasting.py",
        title="User Demand Forecasting",
        icon=":material/upload_file:",
        url_path="user_demand_forecasting",
    ),
]

navigation_dict = {
    "OVERVIEW": overview_pages,
    "DEMAND INTELLIGENCE": intelligence_pages,
    "OPERATIONS": operations_pages,
}

pg = st.navigation(
    navigation_dict,
    position="sidebar",
)

pg.run()
