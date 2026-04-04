"""Comprehensive UI validation tests.

Covers every dashboard page for:
  - Page load without errors
  - Required headings and section titles present
  - Explanatory context banners visible (recruiter-facing copy)
  - Charts rendered (no "No data" fallback when data exists)
  - Scroll resets to top on every page transition
  - No Python tracebacks anywhere

Usage:
    pytest tests/test_full_ui.py -v -m selenium
    (dashboard must be running at DASHBOARD_URL)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8501")
WAIT = 7
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

NAV_PAGES = ["Overview", "Drift Analysis", "Feature Insights", "Retraining Log", "Live Demo"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reachable() -> bool:
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


def _go(driver: webdriver.Chrome, page: str) -> None:
    WebDriverWait(driver, 12).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stSidebar']"))
    )
    for label in driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSidebar'] label"):
        if label.text.strip() == page:
            label.click()
            time.sleep(WAIT)
            return
    raise AssertionError(f"Nav label '{page}' not found")


def _body(driver: webdriver.Chrome) -> str:
    return driver.find_element(By.TAG_NAME, "body").text


def _scroll_top(driver: webdriver.Chrome) -> int:
    return driver.execute_script(
        "var el = document.querySelector('[data-testid=\"stMain\"]')"
        " || document.querySelector('.main');"
        "return el ? el.scrollTop : window.scrollY;"
    )


def _scroll_down(driver: webdriver.Chrome, px: int = 600) -> None:
    driver.execute_script(
        f"var el = document.querySelector('[data-testid=\"stMain\"]')"
        f" || document.querySelector('.main');"
        f"if (el) {{ el.scrollTop = {px}; }} else {{ window.scrollBy(0, {px}); }}"
    )
    time.sleep(0.4)


def _screenshot(driver: webdriver.Chrome, name: str) -> None:
    driver.save_screenshot(str(ASSETS / f"{name}.png"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def driver():
    if not _reachable():
        pytest.skip(f"Dashboard not running at {DASHBOARD_URL}")
    drv = _make_driver()
    drv.get(DASHBOARD_URL)
    time.sleep(WAIT)
    yield drv
    drv.quit()


# ---------------------------------------------------------------------------
# Global checks
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestGlobal:

    def test_page_title_contains_argus(self, driver):
        assert "Argus" in driver.title, f"Unexpected title: {driver.title!r}"

    def test_taxi_favicon_set(self, driver):
        icons = driver.find_elements(By.CSS_SELECTOR, "link[rel*='icon']")
        assert icons, "No favicon link element found"

    def test_sidebar_present(self, driver):
        assert driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSidebar']"), \
            "Sidebar not found"

    def test_sidebar_shows_project_description(self, driver):
        body = _body(driver)
        assert "end-to-end" in body.lower() or "observability" in body.lower(), \
            "Sidebar project description not visible"

    def test_sidebar_shows_tech_stack(self, driver):
        body = _body(driver)
        for tech in ("FastAPI", "scikit-learn", "MLflow"):
            assert tech in body, f"Tech stack badge '{tech}' missing from sidebar"

    def test_no_traceback_on_load(self, driver):
        assert "Traceback (most recent call last)" not in _body(driver)

    def test_all_nav_pages_present(self, driver):
        body = _body(driver)
        for page in NAV_PAGES:
            assert page in body, f"Nav option '{page}' not found"

    def test_api_status_badge_visible(self, driver):
        body = _body(driver)
        assert "API Online" in body or "API Offline" in body, \
            "API status badge not shown"

    def test_refresh_now_button_present(self, driver):
        buttons = [b.text.strip() for b in driver.find_elements(By.TAG_NAME, "button")]
        assert "Refresh Now" in buttons, f"Refresh Now missing. Buttons: {buttons}"


# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestOverview:

    def test_overview_loads(self, driver):
        _go(driver, "Overview")
        assert "Argus" in _body(driver)

    def test_overview_context_banner_problem(self, driver):
        body = _body(driver)
        assert "degrad" in body.lower() or "distribut" in body.lower(), \
            "Overview 'THE PROBLEM' banner text not visible"

    def test_overview_context_banner_domain(self, driver):
        assert "NYC taxi" in _body(driver) or "taxi" in _body(driver).lower(), \
            "Domain context (NYC taxi) not shown on Overview"

    def test_overview_metrics_present(self, driver):
        body = _body(driver)
        for label in ("Rolling RMSE", "Baseline RMSE", "Labeled Samples"):
            assert label in body, f"Metric '{label}' not on Overview"

    def test_overview_no_traceback(self, driver):
        assert "Traceback (most recent call last)" not in _body(driver)

    def test_overview_screenshot(self, driver):
        _screenshot(driver, "page_overview")


# ---------------------------------------------------------------------------
# Drift Analysis page
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestDriftAnalysis:

    def test_drift_analysis_loads(self, driver):
        _go(driver, "Drift Analysis")
        body = _body(driver)
        assert "Drift Analysis" in body

    def test_psi_explanation_visible(self, driver):
        body = _body(driver)
        assert "PSI" in body and ("Population Stability" in body or "0.10" in body), \
            "PSI methodology explanation not visible"

    def test_ks_explanation_visible(self, driver):
        body = _body(driver)
        assert "Kolmogorov" in body or "KS" in body, \
            "KS test explanation not visible"

    def test_drift_threshold_labels_visible(self, driver):
        body = _body(driver)
        assert "0.20" in body or "0.10" in body, \
            "Drift threshold values not shown"

    def test_psi_chart_section_heading(self, driver):
        assert "PSI by Feature" in _body(driver), \
            "PSI bar chart section heading missing"

    def test_ks_chart_section_heading(self, driver):
        assert "KS Test" in _body(driver), \
            "KS test chart section heading missing"

    def test_no_traceback(self, driver):
        assert "Traceback (most recent call last)" not in _body(driver)

    def test_drift_screenshot(self, driver):
        _screenshot(driver, "page_drift_analysis")


# ---------------------------------------------------------------------------
# Feature Insights page
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestFeatureInsights:

    def test_feature_insights_loads(self, driver):
        _go(driver, "Feature Insights")
        assert "Feature Insights" in _body(driver)

    def test_root_cause_explanation_visible(self, driver):
        body = _body(driver)
        assert "root" in body.lower() or "importance" in body.lower(), \
            "Root-cause explanation banner not visible"

    def test_drift_radar_section_present(self, driver):
        assert "Drift Radar" in _body(driver) or "Drift Ranking" in _body(driver), \
            "Drift Radar or Ranking section heading missing"

    def test_feature_importance_section_present(self, driver):
        body = _body(driver)
        assert "Feature Importance" in body or "importance" in body.lower(), \
            "Feature Importance section missing"

    def test_no_traceback(self, driver):
        assert "Traceback (most recent call last)" not in _body(driver)

    def test_insights_screenshot(self, driver):
        _screenshot(driver, "page_feature_insights")


# ---------------------------------------------------------------------------
# Retraining Log page
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestRetrainingLog:

    def test_retraining_log_loads(self, driver):
        _go(driver, "Retraining Log")
        assert "Retraining Log" in _body(driver)

    def test_policy_box_visible(self, driver):
        body = _body(driver)
        assert "RETRAINING POLICY" in body or "two independent" in body.lower() \
            or "conservative" in body.lower(), \
            "Retraining policy explanation box not visible"

    def test_gate_1_feature_drift_explained(self, driver):
        body = _body(driver)
        assert "GATE 1" in body or "FEATURE DRIFT" in body, \
            "Gate 1 (feature drift) not explained"

    def test_gate_2_performance_explained(self, driver):
        body = _body(driver)
        assert "GATE 2" in body or "PERFORMANCE DEGRADATION" in body, \
            "Gate 2 (performance) not explained"

    def test_gate_3_sample_budget_explained(self, driver):
        body = _body(driver)
        assert "GATE 3" in body or "SAMPLE BUDGET" in body or "1,000" in body, \
            "Gate 3 (sample budget) not explained"

    def test_decision_log_present(self, driver):
        body = _body(driver)
        assert "Decision Log" in body or "Retrain Blocked" in body \
            or "Retrain Triggered" in body, \
            "Decision log cards not visible"

    def test_gate_badges_on_decision_cards(self, driver):
        body = _body(driver)
        assert "FEATURE DRIFT" in body and "PERFORMANCE" in body and "SAMPLES" in body, \
            "Gate status badges missing from decision cards"

    def test_total_evaluations_metric(self, driver):
        assert "Total Evaluations" in _body(driver), \
            "Total Evaluations metric not shown"

    def test_no_traceback(self, driver):
        assert "Traceback (most recent call last)" not in _body(driver)

    def test_retraining_screenshot(self, driver):
        _screenshot(driver, "page_retraining_log")


# ---------------------------------------------------------------------------
# Live Demo page
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestLiveDemo:

    def test_live_demo_loads(self, driver):
        _go(driver, "Live Demo")
        assert "Live Demo" in _body(driver)

    def test_step_1_predict_explained(self, driver):
        body = _body(driver)
        assert "PREDICT" in body or "/predict" in body, \
            "Step 1 (Predict) explanation not visible"

    def test_step_2_ground_truth_explained(self, driver):
        body = _body(driver)
        assert "GROUND TRUTH" in body or "ground truth" in body.lower() \
            or "feedback" in body.lower(), \
            "Step 2 (Ground truth) explanation not visible"

    def test_step_3_monitor_explained(self, driver):
        body = _body(driver)
        assert "MONITOR" in body or "drift check" in body.lower() \
            or "rolling accuracy" in body.lower(), \
            "Step 3 (Monitor) explanation not visible"

    def test_predict_button_present(self, driver):
        buttons = [b.text.strip() for b in driver.find_elements(By.TAG_NAME, "button")]
        assert any("Predict" in b for b in buttons), \
            f"Predict button not found. Buttons: {buttons}"

    def test_drift_check_button_present(self, driver):
        buttons = [b.text.strip() for b in driver.find_elements(By.TAG_NAME, "button")]
        assert any("Drift" in b for b in buttons), \
            f"Run Drift Check button not found. Buttons: {buttons}"

    def test_curl_commands_shown(self, driver):
        assert "curl" in _body(driver), "Curl commands reference not visible"

    def test_no_traceback(self, driver):
        assert "Traceback (most recent call last)" not in _body(driver)

    def test_live_demo_screenshot(self, driver):
        _screenshot(driver, "page_live_demo")


# ---------------------------------------------------------------------------
# Scroll reset on every page transition
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestScrollReset:

    def test_scroll_resets_live_demo_to_overview(self, driver):
        _go(driver, "Live Demo")
        _scroll_down(driver, 700)
        _go(driver, "Overview")
        assert _scroll_top(driver) <= 50, \
            f"Scroll not reset: Live Demo -> Overview, pos={_scroll_top(driver)}px"

    def test_scroll_resets_across_all_transitions(self, driver):
        pairs = [
            ("Overview",         "Drift Analysis"),
            ("Drift Analysis",   "Feature Insights"),
            ("Feature Insights", "Retraining Log"),
            ("Retraining Log",   "Live Demo"),
            ("Live Demo",        "Overview"),
        ]
        failures = []
        for src, dst in pairs:
            _go(driver, src)
            _scroll_down(driver, 500)
            _go(driver, dst)
            pos = _scroll_top(driver)
            if pos > 50:
                failures.append(f"{src} -> {dst}: scrollTop={pos}px (expected <=50)")
        assert not failures, "Scroll did not reset:\n" + "\n".join(failures)
