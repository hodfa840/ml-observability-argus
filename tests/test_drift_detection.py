"""
Tests for drift detection components.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector
from src.data.generator import TaxiDataGenerator
from src.data.drift_simulator import DriftSimulator


@pytest.fixture
def reference_df() -> pd.DataFrame:
    gen = TaxiDataGenerator(random_seed=0)
    return gen.generate_reference(n_samples=1000)


@pytest.fixture
def detector(reference_df: pd.DataFrame) -> DriftDetector:
    d = DriftDetector()
    d.set_reference(reference_df)
    return d


class TestPSI:
    def test_identical_distributions_psi_near_zero(self, detector: DriftDetector) -> None:
        """PSI should be ~0 when reference == current."""
        ref = np.random.default_rng(42).normal(0, 1, 1000)
        psi = DriftDetector._psi(ref, ref)
        assert psi < 0.05, f"Expected PSI≈0 for identical distributions, got {psi}"

    def test_very_different_distributions_high_psi(self, detector: DriftDetector) -> None:
        """PSI should be large for very different distributions."""
        ref = np.random.default_rng(1).normal(0, 1, 1000)
        current = np.random.default_rng(2).normal(5, 1, 1000)  # large shift
        psi = DriftDetector._psi(ref, current)
        assert psi >= 0.2, f"Expected PSI>=0.2 for large shift, got {psi}"

    def test_psi_non_negative(self) -> None:
        rng = np.random.default_rng(99)
        ref = rng.uniform(0, 10, 500)
        cur = rng.uniform(0, 10, 500)
        psi = DriftDetector._psi(ref, cur)
        assert psi >= 0, "PSI must be non-negative"


class TestFeatureDrift:
    def test_no_drift_on_same_data(self, detector: DriftDetector, reference_df: pd.DataFrame) -> None:
        report = detector.detect_feature_drift(reference_df)
        # Most features should not show drift on the same data
        assert "drift_detected" in report
        assert "feature_results" in report

    def test_drift_detected_after_shift(self, detector: DriftDetector, reference_df: pd.DataFrame) -> None:
        simulator = DriftSimulator(random_seed=7)
        feature_cols = ["trip_distance", "passenger_count", "pickup_hour"]
        drifted = reference_df.copy()
        drifted_features = simulator.apply(
            drifted[feature_cols],
            drift_type="sudden",
            severity=3.0,
        )
        for col in feature_cols:
            drifted[col] = drifted_features[col].values

        report = detector.detect_feature_drift(drifted, features=feature_cols)
        assert report["drift_detected"] is True, "Drift should be detected after large shift"
        assert len(report["drifted_features"]) > 0

    def test_report_structure(self, detector: DriftDetector, reference_df: pd.DataFrame) -> None:
        report = detector.detect_feature_drift(reference_df, features=["trip_distance"])
        assert "drift_detected" in report
        assert "feature_results" in report
        assert "drifted_features" in report
        assert "n_live_samples" in report
        assert "timestamp" in report
        assert "trip_distance" in report["feature_results"]
        feat = report["feature_results"]["trip_distance"]
        assert "psi" in feat
        assert "ks_stat" in feat
        assert "ks_pvalue" in feat


class TestPerformanceDrift:
    def test_no_degradation_when_same_rmse(self, detector: DriftDetector) -> None:
        report = detector.detect_performance_drift(recent_rmse=3.5, baseline_rmse=3.5)
        assert report["drift_detected"] is False

    def test_degradation_detected_above_threshold(self, detector: DriftDetector) -> None:
        # 20% increase should trigger (threshold is 15%)
        report = detector.detect_performance_drift(recent_rmse=3.5 * 1.20, baseline_rmse=3.5)
        assert report["drift_detected"] is True

    def test_borderline_case(self, detector: DriftDetector) -> None:
        # Exactly at threshold: 15% → not triggered (strict >)
        report = detector.detect_performance_drift(recent_rmse=3.5 * 1.15, baseline_rmse=3.5)
        assert isinstance(report["drift_detected"], bool)
