import streamlit as st


def render_navigation_drawer():
    """Render page links inside the collapsible navigation drawer."""

    with st.sidebar:
        st.markdown("### Menu")
        st.page_link(
            "pages/01_executive_overview.py",
            label="Executive Overview",
            icon=":material/dashboard:",
        )
        st.page_link(
            "pages/02_demand_analytics.py",
            label="Demand Analytics",
            icon=":material/analytics:",
        )
        st.page_link(
            "pages/03_demand_forecasting.py",
            label="Demand Forecasting",
            icon=":material/trending_up:",
        )
        st.page_link(
            "pages/04_model_performance.py",
            label="Model Performance",
            icon=":material/tune:",
        )
        st.page_link(
            "pages/05_inventory_management.py",
            label="Inventory Management",
            icon=":material/inventory_2:",
        )
        st.page_link(
            "pages/06_supply_chain_risk.py",
            label="Supply Chain Risk",
            icon=":material/warning:",
        )
        st.page_link(
            "pages/07_scenario_simulator.py",
            label="Scenario Simulator",
            icon=":material/science:",
        )
        st.divider()
