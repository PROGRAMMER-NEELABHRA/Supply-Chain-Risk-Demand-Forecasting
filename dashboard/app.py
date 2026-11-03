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

from dashboard.components.sidebar import render_sidebar_header, render_top_theme_toggle
from dashboard.styles.theme import get_current_theme, inject_css

# Page configuration (MUST be executed once at app root)
st.set_page_config(
    page_title="Supply Chain Demand Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply active theme CSS variables
inject_css()

# Render unified sidebar branding
render_sidebar_header()

# Render top-right theme toggle
render_top_theme_toggle()

# Define navigation structure with st.Page
overview_pages = [
    st.Page(
        "pages/01_executive_overview.py",
        title="Executive Overview",
        icon=":material/dashboard:",
        default=True,
    ),
]

intelligence_pages = [
    st.Page(
        "pages/02_demand_analytics.py",
        title="Demand Analytics",
        icon=":material/analytics:",
    ),
    st.Page(
        "pages/03_demand_forecasting.py",
        title="Demand Forecasting",
        icon=":material/trending_up:",
    ),
    st.Page(
        "pages/04_model_performance.py",
        title="Model Performance",
        icon=":material/tune:",
    ),
]

operations_pages = [
    st.Page(
        "pages/05_inventory_management.py",
        title="Inventory Management",
        icon=":material/inventory_2:",
    ),
    st.Page(
        "pages/06_supply_chain_risk.py",
        title="Supply Chain Risk",
        icon=":material/warning:",
    ),
    st.Page(
        "pages/07_scenario_simulator.py",
        title="Scenario Simulator",
        icon=":material/science:",
    ),
]

navigation_dict = {
    "OVERVIEW": overview_pages,
    "DEMAND INTELLIGENCE": intelligence_pages,
    "OPERATIONS": operations_pages,
}

pg = st.navigation(navigation_dict)
pg.run()
