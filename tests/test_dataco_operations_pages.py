from pathlib import Path

from streamlit.testing.v1 import AppTest


PAGES = Path(__file__).resolve().parents[1] / "dashboard" / "pages"


def test_inventory_risk_and_scenario_pages_render():
    for filename in (
        "05_inventory_management.py",
        "06_supply_chain_risk.py",
        "07_scenario_simulator.py",
    ):
        app = AppTest.from_file(str(PAGES / filename), default_timeout=45).run()
        assert not app.exception, (filename, app.exception)


def test_inventory_page_exposes_user_supplied_product_inputs():
    app = AppTest.from_file(
        str(PAGES / "05_inventory_management.py"), default_timeout=45
    ).run()

    assert any("DataCo has no current-stock or supplier lead-time fields" in item.value for item in app.info)
    assert len(app.dataframe) >= 1


def test_risk_page_disclaims_unavailable_supplier_and_logistics_metrics():
    app = AppTest.from_file(
        str(PAGES / "06_supply_chain_risk.py"), default_timeout=45
    ).run()

    assert any(
        "Supplier reliability, logistics bottlenecks" in item.value
        for item in app.info
    )


def test_scenario_page_exposes_download_and_horizon_controls():
    app = AppTest.from_file(
        str(PAGES / "07_scenario_simulator.py"), default_timeout=45
    ).run()

    assert not app.exception
    assert any(
        button.label == "Download scenario CSV"
        for button in app.get("download_button")
    )
    assert any(slider.label == "User multiplier" for slider in app.slider)


def test_scenario_multiplier_interactively_updates_user_scenario_total():
    app = AppTest.from_file(
        str(PAGES / "07_scenario_simulator.py"), default_timeout=45
    ).run()
    app.slider[2].set_value(1.5).run()

    user_total = next(
        metric for metric in app.metric if metric.label == "User Scenario Total"
    )

    assert user_total.value == "56,089.9 units"
    assert "+50.0%" in user_total.delta
