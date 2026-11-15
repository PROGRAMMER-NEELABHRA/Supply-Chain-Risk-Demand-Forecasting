from urllib.parse import urlparse

import streamlit as st


PAGES = [
    ("", "pages/01_executive_overview.py", "Executive Overview", ":material/dashboard:"),
    ("demand_analytics", "pages/02_demand_analytics.py", "Demand Analytics", ":material/analytics:"),
    ("demand_forecasting", "pages/03_demand_forecasting.py", "Demand Forecasting", ":material/trending_up:"),
    ("user_demand_forecasting", "pages/08_user_demand_forecasting.py", "User Demand Forecasting", ":material/upload_file:"),
    ("model_performance", "pages/04_model_performance.py", "Model Performance", ":material/tune:"),
    ("inventory_management", "pages/05_inventory_management.py", "Inventory Management", ":material/inventory_2:"),
    ("supply_chain_risk", "pages/06_supply_chain_risk.py", "Supply Chain Risk", ":material/warning:"),
    ("scenario_simulator", "pages/07_scenario_simulator.py", "Scenario Simulator", ":material/science:"),
]


def render_navigation_drawer():
    """Render the dashboard page links inside the collapsible navigation drawer."""
    current_page = urlparse(st.context.url).path.rstrip("/").rsplit("/", 1)[-1]

    with st.sidebar:
        st.markdown("### Menu")
        for route, page_path, label, icon in PAGES:
            if current_page == route:
                with st.container(key="active_navigation_link"):
                    st.page_link(page_path, label=label, icon=icon)
            else:
                st.page_link(page_path, label=label, icon=icon)
        st.divider()
