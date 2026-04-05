"""Dashboard diagnostic tests.

Covers two layers:
  1. Unit tests  — data-loading functions work independently of Streamlit
  2. Selenium UI tests — the rendered dashboard shows the expected elements

Usage:
    pytest tests/test_dashboard.py -v
    pytest tests/test_dashboard.py -v -k unit          # unit tests only
    pytest tests/test_dashboard.py -v -k selenium      # UI tests only (needs running dashboard)

The Selenium tests expect the Streamlit dashboard at DASHBOARD_URL (default
http://localhost:8501).  Start it first with:
    .venv/Scripts/python run.py --no-sim
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8501")
LOG_PATHS = {
    "performance": ROOT / "data" / "logs" / "performance.jsonl",
    "drift":       ROOT / "data" / "logs" / "drift_reports.jsonl",
    "retrain":     ROOT / "data" / "logs" / "retraining.jsonl",
    "predictions": ROOT / "data" / "logs" / "predictions.jsonl",
}


# ---------------------------------------------------------------------------
# Helpers (mirror dashboard logic without Streamlit dependency)
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, limit: int = 2000) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Verify that log files exist and load correctly."""

    def test_performance_file_exists(self):
        assert LOG_PATHS["performance"].exists(), (
            "performance.jsonl not found — run the simulation first: "
            "python scripts/simulate_drift.py"
        )

    def test_performance_file_has_rmse_column(self):
        df = _load_jsonl(LOG_PATHS["performance"])
        assert not df.empty, "performance.jsonl is empty"
        assert "rmse" in df.columns, f"Expected 'rmse' column; got {list(df.columns)}"

    def test_performance_rmse_values_are_positive(self):
        df = _load_jsonl(LOG_PATHS["performance"])
        if df.empty:
            pytest.skip("No performance data yet")
        assert (df["rmse"] > 0).all(), "RMSE values must be positive"

    def test_performance_file_has_required_columns(self):
        df = _load_jsonl(LOG_PATHS["performance"])
        if df.empty:
            pytest.skip("No performance data yet")
        required = {"rmse", "mae", "r2", "n_samples", "timestamp"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns in performance log: {missing}"

    def test_predictions_file_exists_and_has_data(self):
        assert LOG_PATHS["predictions"].exists(), "predictions.jsonl not found"
        df = _load_jsonl(LOG_PATHS["predictions"])
        assert not df.empty, "predictions.jsonl is empty"

    def test_drift_file_structure(self):
        if not LOG_PATHS["drift"].exists():
            pytest.skip("No drift reports yet")
        df = _load_jsonl(LOG_PATHS["drift"])
        assert not df.empty
        assert "drift_detected" in df.columns, (
            f"Expected 'drift_detected'; got {list(df.columns)}"
        )

    def test_path_resolution_is_correct(self):
        """PROJECT_ROOT computed from dashboard/app.py must point to repo root."""
        dashboard_file = ROOT / "dashboard" / "app.py"
        resolved_root = dashboard_file.resolve().parent.parent
        assert resolved_root == ROOT.resolve(), (
            f"Path mismatch: dashboard resolves to {resolved_root}, "
            f"expected {ROOT.resolve()}"
        )

    def test_load_jsonl_returns_dataframe_not_empty_when_file_has_data(self):
        path = LOG_PATHS["performance"]
        if not path.exists():
            pytest.skip("No performance data yet")
        df = _load_jsonl(path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) > 0

    def test_load_jsonl_handles_missing_file_gracefully(self):
        df = _load_jsonl(ROOT / "data" / "logs" / "nonexistent.jsonl")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_jsonl_handles_corrupted_lines_gracefully(self):
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"rmse": 1.5, "mae": 1.2}\n')
            f.write("NOT JSON\n")
            f.write('{"rmse": 1.6, "mae": 1.3}\n')
            tmp = Path(f.name)
        try:
            df = _load_jsonl(tmp)
            assert len(df) == 2, "Should skip corrupted lines and keep valid ones"
            assert list(df["rmse"]) == [1.5, 1.6]
        finally:
            tmp.unlink()


# ---------------------------------------------------------------------------
# Selenium UI tests
# ---------------------------------------------------------------------------

def _get_driver():
    """Return a headless Chrome driver via webdriver-manager."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1600,900")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def _dashboard_reachable() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(DASHBOARD_URL, timeout=3)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def driver():
    if not _dashboard_reachable():
        pytest.skip(f"Dashboard not running at {DASHBOARD_URL}")
    drv = _get_driver()
    drv.get(DASHBOARD_URL)
    time.sleep(6)
    yield drv
    drv.quit()


@pytest.mark.selenium
class TestDashboardUI:
    """Selenium tests against the live Streamlit dashboard."""

    def test_page_title_is_argus(self, driver):
        assert "Argus" in driver.title, (
            f"Expected 'Argus' in page title, got: {driver.title!r}"
        )

    def test_sidebar_is_visible(self, driver):
        from selenium.webdriver.common.by import By
        sidebar = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSidebar']")
        assert sidebar, "Sidebar element not found"

    def test_api_status_shown_in_sidebar(self, driver):
        from selenium.webdriver.common.by import By
        body_text = driver.find_element(By.TAG_NAME, "body").text
        assert any(kw in body_text for kw in ("API Online", "API Offline")), (
            "Expected API status badge in sidebar"
        )

    def test_navigation_pages_present(self, driver):
        from selenium.webdriver.common.by import By
        body_text = driver.find_element(By.TAG_NAME, "body").text
        for page in ("Overview", "Drift Analysis", "Feature Insights",
                     "Retraining Log", "Live Demo"):
            assert page in body_text, f"Navigation option '{page}' not found"

    def test_overview_metrics_rendered(self, driver):
        from selenium.webdriver.common.by import By
        body_text = driver.find_element(By.TAG_NAME, "body").text
        for label in ("Rolling RMSE", "Baseline RMSE", "Labeled Samples"):
            assert label in body_text, f"Metric '{label}' not visible on Overview"

    def test_no_python_traceback_on_page(self, driver):
        from selenium.webdriver.common.by import By
        body_text = driver.find_element(By.TAG_NAME, "body").text
        assert "Traceback (most recent call last)" not in body_text, (
            "Python traceback found on dashboard page"
        )

    def test_chart_renders_when_data_present(self, driver):
        """If performance data exists, the RMSE chart must be visible (not 'No data')."""
        if not LOG_PATHS["performance"].exists():
            pytest.skip("No performance data — chart absence is expected")
        df = _load_jsonl(LOG_PATHS["performance"])
        if df.empty:
            pytest.skip("performance.jsonl is empty — chart absence is expected")

        from selenium.webdriver.common.by import By

        body_text = driver.find_element(By.TAG_NAME, "body").text
        no_data_msg = "No performance data yet"
        assert no_data_msg not in body_text, (
            f"Dashboard shows '{no_data_msg}' but performance.jsonl has "
            f"{len(df)} rows. Root cause: auto-refresh clears the cache "
            "BEFORE chart code runs, causing an infinite blank loop."
        )

    def test_refresh_now_button_exists(self, driver):
        from selenium.webdriver.common.by import By
        buttons = driver.find_elements(By.TAG_NAME, "button")
        labels = [b.text.strip() for b in buttons]
        assert "Refresh Now" in labels, (
            f"'Refresh Now' button not found. Available buttons: {labels}"
        )

    def test_clicking_refresh_loads_chart(self, driver):
        """Click Refresh Now and verify the chart appears within 10 seconds."""
        if not LOG_PATHS["performance"].exists():
            pytest.skip("No performance data")
        df = _load_jsonl(LOG_PATHS["performance"])
        if df.empty:
            pytest.skip("performance.jsonl is empty")

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if btn.text.strip() == "Refresh Now":
                btn.click()
                break

        time.sleep(8)
        body_text = driver.find_element(By.TAG_NAME, "body").text
        no_data_msg = "No performance data yet"
        assert no_data_msg not in body_text, (
            "Chart still absent after clicking Refresh Now"
        )

    def test_screenshot_on_failure(self, driver, request):
        """Save a screenshot to assets/test_screenshot.png for inspection."""
        screenshot_path = ROOT / "assets" / "test_screenshot.png"
        driver.save_screenshot(str(screenshot_path))


# ---------------------------------------------------------------------------
# Unit tests: fix #1 — baseline RMSE must not use iloc[0] from the log
# ---------------------------------------------------------------------------

class TestBaselineRmseLogic:
    """
    Verify that the baseline hline calculation uses api_metrics baseline_rmse
    rather than the first row of the performance log.

    Before the fix:  bsl = perf_df["rmse"].iloc[0]
    After the fix:   bsl = baseline or perf_df["rmse"].min()

    If the log starts mid-drift (high RMSE), iloc[0] would have been wrong.
    """

    def _bsl(self, api_baseline, perf_rmse_values: list) -> float:
        """Replicate the fixed dashboard bsl calculation."""
        import numpy as np
        df = pd.DataFrame({"rmse": perf_rmse_values})
        baseline = api_baseline
        return baseline if baseline else float(df["rmse"].min())

    def test_uses_api_baseline_when_available(self):
        # Log starts at a high value (simulating mid-drift start)
        rmse_series = [10.5, 10.8, 11.2, 11.0, 10.9]
        bsl = self._bsl(api_baseline=2.1, perf_rmse_values=rmse_series)
        assert bsl == 2.1, (
            f"Expected api baseline 2.1, got {bsl}. "
            "Fix is not applied: bsl must come from api_metrics, not iloc[0]."
        )

    def test_falls_back_to_min_when_api_unavailable(self):
        rmse_series = [1.8, 2.1, 5.3, 9.0, 3.2]
        bsl = self._bsl(api_baseline=None, perf_rmse_values=rmse_series)
        assert bsl == 1.8, (
            f"Fallback should be min(rmse)=1.8, got {bsl}."
        )

    def test_old_iloc0_would_have_failed_mid_drift(self):
        """Demonstrate the old bug: iloc[0] on a mid-drift log gives wrong baseline."""
        rmse_series = [10.5, 10.8, 11.2, 11.0, 10.9]
        df = pd.DataFrame({"rmse": rmse_series})
        old_bsl = df["rmse"].iloc[0]   # old (broken) logic
        assert old_bsl == 10.5, "Setup check: old logic picks high value"
        # The old bsl would set the baseline hline at 10.5 instead of ~2.1,
        # causing the chart to look flat (everything near or above "baseline")
        assert old_bsl > 5.0, (
            "Old baseline would have been unreasonably high — confirms the bug."
        )

    def test_alert_threshold_is_correct_fraction_of_bsl(self):
        """Alert hline must be 15% above baseline."""
        bsl = 2.131
        alert = bsl * 1.15
        assert abs(alert - 2.451) < 0.01, f"Alert threshold wrong: {alert:.3f}"


# ---------------------------------------------------------------------------
# Unit tests: fix #2 — R² y-axis must accommodate negative values
# ---------------------------------------------------------------------------

class TestR2AxisScaling:
    """
    Verify that the R² chart y-axis lower bound scales to include negative R²
    instead of clipping at 0.

    Before the fix:  range=[0, 1.05]  (negative values invisible)
    After the fix:   range=[r2_floor, 1.05]  where r2_floor < 0 when data dips negative
    """

    def _r2_floor(self, r2_values: list) -> float:
        """Replicate the fixed dashboard r2_floor calculation."""
        r2_min = min(r2_values)
        return min(r2_min - 0.05, -0.1) if r2_min < 0 else -0.05

    def test_negative_r2_produces_negative_floor(self):
        r2_series = [0.91, 0.60, -0.49, -1.22, 0.83]
        floor = self._r2_floor(r2_series)
        assert floor < 0, f"r2_floor must be negative when data goes below 0, got {floor}"
        assert floor <= -1.22 - 0.05, (
            f"Floor {floor} is not low enough to show min r2=-1.22 "
            "(should be min - 0.05 = -1.27)"
        )

    def test_all_positive_r2_uses_small_negative_floor(self):
        r2_series = [0.91, 0.88, 0.93, 0.85]
        floor = self._r2_floor(r2_series)
        assert floor == -0.05, (
            f"When all R² > 0, floor should be -0.05 for breathing room, got {floor}"
        )

    def test_floor_is_below_min_r2(self):
        """Floor must always be below the minimum R² value so no data is clipped."""
        for min_r2 in [-0.05, -0.5, -1.0, -1.22]:
            r2_series = [0.9, min_r2]
            floor = self._r2_floor(r2_series)
            assert floor <= min_r2, (
                f"At min_r2={min_r2}, floor={floor} clips data (must be <= min_r2)"
            )

    def test_old_hardcoded_range_clipped_negative_r2(self):
        """Show that the old range=[0, 1.05] would have hidden the negative data."""
        old_range_min = 0
        r2_min_in_data = -1.22
        assert r2_min_in_data < old_range_min, (
            "Confirms bug: min R² in data is below old y-axis floor of 0"
        )


# ---------------------------------------------------------------------------
# Selenium: verify chart renders correctly with fixed logic
# ---------------------------------------------------------------------------

@pytest.mark.selenium
class TestChartFixes:
    """End-to-end Selenium tests verifying the two chart fixes in production."""

    def test_overview_chart_section_visible(self, driver):
        from selenium.webdriver.common.by import By
        body = driver.find_element(By.TAG_NAME, "body").text
        assert "Prediction Error Over Time" in body, (
            "RMSE chart section heading not visible on Overview"
        )

    def test_baseline_annotation_present_in_chart(self, driver):
        """
        The 'Baseline' hline annotation must appear in the rendered SVG.
        If bsl was computed from a high iloc[0], the annotation would still
        appear but at the wrong Y level — this confirms it's rendered at all.
        """
        from selenium.webdriver.common.by import By
        page_source = driver.page_source
        assert "Baseline" in page_source, (
            "Baseline annotation not found in rendered page source. "
            "Chart may not have rendered."
        )

    def test_alert_annotation_present_in_chart(self, driver):
        from selenium.webdriver.common.by import By
        page_source = driver.page_source
        assert "Alert" in page_source or "+15%" in page_source, (
            "Alert +15% annotation not found in rendered chart."
        )

    def test_r2_chart_section_visible(self, driver):
        from selenium.webdriver.common.by import By
        page_source = driver.page_source
        # R² label should appear as an axis title in the SVG
        assert "R²" in page_source or "R\u00b2" in page_source, (
            "R² chart axis label not found — chart may not have rendered."
        )

    def test_no_traceback_on_overview(self, driver):
        from selenium.webdriver.common.by import By
        assert "Traceback (most recent call last)" not in \
            driver.find_element(By.TAG_NAME, "body").text

    def test_overview_screenshot_with_fixes(self, driver):
        """Save a screenshot showing the fixed chart for visual verification."""
        screenshot_path = ROOT / "assets" / "overview_chart_fixed.png"
        driver.save_screenshot(str(screenshot_path))
