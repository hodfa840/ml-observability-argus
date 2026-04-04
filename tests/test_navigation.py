"""Navigation and scroll-position tests.

Verifies that switching between dashboard pages resets the scroll
position to the top of the main content area.

Usage:
    pytest tests/test_navigation.py -v
    (dashboard must be running at DASHBOARD_URL)
"""
from __future__ import annotations

import os
import time

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8501")
NAV_PAGES = ["Overview", "Drift Analysis", "Feature Insights", "Retraining Log", "Live Demo"]
WAIT = 6


def _dashboard_reachable() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(DASHBOARD_URL, timeout=3)
        return True
    except Exception:
        return False


def _make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1600,1000")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def _navigate_to(driver: webdriver.Chrome, page: str, wait: int = WAIT) -> None:
    """Click a sidebar navigation radio button and wait for the page to load."""
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stSidebar']"))
    )
    radios = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSidebar'] label")
    for label in radios:
        if label.text.strip() == page:
            label.click()
            time.sleep(wait)
            return
    raise AssertionError(f"Navigation label '{page}' not found in sidebar")


def _main_scroll_top(driver: webdriver.Chrome) -> int:
    """Return the scrollTop of the main Streamlit content container."""
    return driver.execute_script(
        "var el = document.querySelector('[data-testid=\"stMain\"]') "
        "|| document.querySelector('.main') "
        "|| document.querySelector('[data-testid=\"stAppViewContainer\"]');"
        "return el ? el.scrollTop : window.scrollY;"
    )


def _scroll_down(driver: webdriver.Chrome, px: int = 600) -> None:
    driver.execute_script(
        f"var el = document.querySelector('[data-testid=\"stMain\"]') "
        f"|| document.querySelector('.main') "
        f"|| document.querySelector('[data-testid=\"stAppViewContainer\"]');"
        f"if (el) {{ el.scrollTop = {px}; }} else {{ window.scrollBy(0, {px}); }}"
    )
    time.sleep(0.5)


@pytest.fixture(scope="module")
def driver():
    if not _dashboard_reachable():
        pytest.skip(f"Dashboard not running at {DASHBOARD_URL}")
    drv = _make_driver()
    drv.get(DASHBOARD_URL)
    time.sleep(WAIT)
    yield drv
    drv.quit()


@pytest.mark.selenium
class TestNavigation:

    def test_all_pages_load_without_error(self, driver):
        """Cycle through every page and confirm no Python traceback appears."""
        for page in NAV_PAGES:
            _navigate_to(driver, page)
            body = driver.find_element(By.TAG_NAME, "body").text
            assert "Traceback (most recent call last)" not in body, (
                f"Python traceback on page '{page}'"
            )

    def test_page_content_changes_on_navigation(self, driver):
        """Each page must show its own title, not the previous page's title."""
        _navigate_to(driver, "Live Demo")
        live_body = driver.find_element(By.TAG_NAME, "body").text
        assert "Live Demo" in live_body

        _navigate_to(driver, "Overview")
        overview_body = driver.find_element(By.TAG_NAME, "body").text
        assert "Rolling RMSE" in overview_body, (
            "Overview content not visible after navigating from Live Demo"
        )

    def test_scroll_resets_to_top_on_page_change(self, driver):
        """
        Reproduce the reported bug:
        Navigate to Live Demo, scroll down, then go to Overview.
        The main content area must scroll back to y=0.
        """
        _navigate_to(driver, "Live Demo")
        _scroll_down(driver, px=700)
        scroll_before = _main_scroll_top(driver)

        _navigate_to(driver, "Overview")
        scroll_after = _main_scroll_top(driver)

        assert scroll_after <= 50, (
            f"BUG: page did not scroll to top after tab change. "
            f"scrollTop before={scroll_before}px, after={scroll_after}px. "
            f"User sees content from the old page position."
        )

    def test_scroll_resets_across_all_page_transitions(self, driver):
        """Scroll down on every page, then switch — top must always reset."""
        failures = []
        pairs = [
            ("Drift Analysis",   "Feature Insights"),
            ("Feature Insights", "Retraining Log"),
            ("Retraining Log",   "Live Demo"),
            ("Live Demo",        "Overview"),
            ("Overview",         "Drift Analysis"),
        ]
        for src, dst in pairs:
            _navigate_to(driver, src)
            _scroll_down(driver, px=500)
            _navigate_to(driver, dst)
            pos = _main_scroll_top(driver)
            if pos > 50:
                failures.append(f"{src} -> {dst}: scrollTop={pos}px (expected <=50)")

        assert not failures, (
            "Scroll position did not reset on these transitions:\n" +
            "\n".join(failures)
        )

    def test_screenshot_scroll_bug(self, driver):
        """Save before/after screenshots of the scroll bug for visual inspection."""
        from pathlib import Path
        assets = Path(__file__).resolve().parent.parent / "assets"

        _navigate_to(driver, "Live Demo")
        _scroll_down(driver, px=700)
        driver.save_screenshot(str(assets / "nav_test_before.png"))

        _navigate_to(driver, "Overview")
        driver.save_screenshot(str(assets / "nav_test_after.png"))
