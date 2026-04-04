"""
Tests for the intelligent retraining trigger and pipeline.
"""
from __future__ import annotations

import pytest


# Fixtures
_FEATURE_DRIFT_DETECTED = {
    "drift_detected": True,
    "drifted_features": ["trip_distance", "pickup_hour"],
    "feature_results": {
        "trip_distance": {"psi": 0.31, "ks_stat": 0.2, "ks_pvalue": 0.001, "drifted": True},
        "pickup_hour": {"psi": 0.24, "ks_stat": 0.18, "ks_pvalue": 0.01, "drifted": True},
    },
    "n_live_samples": 500,
    "timestamp": "2024-01-01T00:00:00Z",
}

_FEATURE_DRIFT_NONE = {
    "drift_detected": False,
    "drifted_features": [],
    "feature_results": {
        "trip_distance": {"psi": 0.05, "ks_stat": 0.05, "ks_pvalue": 0.5, "drifted": False},
    },
    "n_live_samples": 500,
    "timestamp": "2024-01-01T00:00:00Z",
}

_PERF_DEGRADED = {
    "drift_detected": True,
    "recent_rmse": 5.2,
    "baseline_rmse": 3.5,
    "pct_change": 48.6,
    "timestamp": "2024-01-01T00:00:00Z",
}

_PERF_OK = {
    "drift_detected": False,
    "recent_rmse": 3.6,
    "baseline_rmse": 3.5,
    "pct_change": 2.8,
    "timestamp": "2024-01-01T00:00:00Z",
}


class TestRetrainingTrigger:
    @pytest.fixture
    def trigger(self):
        from src.retraining.trigger import RetrainingTrigger
        return RetrainingTrigger()

    def test_trigger_when_both_drift_and_perf_degrade(self, trigger) -> None:
        decision = trigger.should_retrain(
            feature_drift_report=_FEATURE_DRIFT_DETECTED,
            performance_report=_PERF_DEGRADED,
            samples_since_last_retrain=2000,
        )
        assert decision["should_retrain"] is True

    def test_no_trigger_when_only_feature_drift(self, trigger) -> None:
        decision = trigger.should_retrain(
            feature_drift_report=_FEATURE_DRIFT_DETECTED,
            performance_report=_PERF_OK,
            samples_since_last_retrain=2000,
        )
        assert decision["should_retrain"] is False

    def test_no_trigger_when_only_perf_degraded(self, trigger) -> None:
        decision = trigger.should_retrain(
            feature_drift_report=_FEATURE_DRIFT_NONE,
            performance_report=_PERF_DEGRADED,
            samples_since_last_retrain=2000,
        )
        assert decision["should_retrain"] is False

    def test_no_trigger_insufficient_samples(self, trigger) -> None:
        decision = trigger.should_retrain(
            feature_drift_report=_FEATURE_DRIFT_DETECTED,
            performance_report=_PERF_DEGRADED,
            samples_since_last_retrain=10,  # way below minimum
        )
        assert decision["should_retrain"] is False
        assert any("samples" in b.lower() for b in decision["blocking_reasons"])

    def test_cooldown_blocks_retrain(self, trigger) -> None:
        import time
        trigger._last_retrain_time = time.time()  # just retrained
        decision = trigger.should_retrain(
            feature_drift_report=_FEATURE_DRIFT_DETECTED,
            performance_report=_PERF_DEGRADED,
            samples_since_last_retrain=2000,
        )
        assert decision["should_retrain"] is False
        assert any("cooldown" in b.lower() for b in decision["blocking_reasons"])

    def test_decision_structure(self, trigger) -> None:
        decision = trigger.should_retrain(
            feature_drift_report=_FEATURE_DRIFT_DETECTED,
            performance_report=_PERF_DEGRADED,
            samples_since_last_retrain=2000,
        )
        required_keys = {"should_retrain", "reasons", "blocking_reasons", "feature_drift",
                         "performance_drift", "samples_since_last_retrain", "timestamp"}
        assert required_keys.issubset(set(decision.keys()))


class TestRootCauseAnalyzer:
    @pytest.fixture
    def rca(self):
        from src.monitoring.root_cause_analyzer import RootCauseAnalyzer
        return RootCauseAnalyzer()

    def test_no_drift_returns_monitor_action(self, rca) -> None:
        result = rca.analyze({"drift_detected": False, "feature_results": {}, "drifted_features": []})
        assert result["action_recommended"] == "monitor"
        assert result["root_causes"] == []

    def test_drift_returns_ranked_causes(self, rca) -> None:
        result = rca.analyze(_FEATURE_DRIFT_DETECTED)
        assert len(result["root_causes"]) > 0
        # Sorted by rca_score descending
        scores = [c["rca_score"] for c in result["root_causes"]]
        assert scores == sorted(scores, reverse=True)

    def test_primary_cause_is_top_ranked(self, rca) -> None:
        result = rca.analyze(_FEATURE_DRIFT_DETECTED)
        top = result["root_causes"][0]["feature"]
        assert result["primary_cause"] == top
