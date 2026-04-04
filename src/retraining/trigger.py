"""Intelligent retraining trigger with a dual-gate decision.

Retraining fires only when ALL conditions hold:
  1. Feature drift detected (PSI >= threshold or KS p-value < threshold)
  2. Performance degradation detected (RMSE increased by >= N%)
  3. Minimum samples collected since last retrain
  4. Cooldown period has elapsed
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class RetrainingTrigger:
    """Evaluate whether all retraining conditions are met."""

    def __init__(self) -> None:
        self._last_retrain_time: Optional[float] = None
        self._retrain_log_path = resolve(settings.retraining.retrain_log_path)
        self._cooldown_seconds = settings.retraining.cooldown_hours * 3600

    def should_retrain(
        self,
        feature_drift_report: dict,
        performance_report: dict,
        samples_since_last_retrain: int,
    ) -> dict:
        """Return a decision dict with should_retrain, reasons, and blocking_reasons."""
        reasons: list[str] = []
        blocking: list[str] = []

        feature_drifted = feature_drift_report.get("drift_detected", False)
        perf_drifted = performance_report.get("drift_detected", False)

        if feature_drifted:
            drifted_features = feature_drift_report.get("drifted_features", [])
            reasons.append(f"Feature drift detected in: {drifted_features}")
        else:
            blocking.append("No significant feature drift.")

        if perf_drifted:
            pct = performance_report.get("pct_change", 0.0)
            reasons.append(f"Performance degraded by {pct:.1f}% vs baseline.")
        else:
            blocking.append("Performance within acceptable range.")

        min_samples = settings.retraining.min_samples_since_last_retrain
        if samples_since_last_retrain < min_samples:
            blocking.append(
                f"Only {samples_since_last_retrain} new samples (need {min_samples})."
            )

        if self._last_retrain_time is not None:
            elapsed = time.time() - self._last_retrain_time
            if elapsed < self._cooldown_seconds:
                remaining = int(self._cooldown_seconds - elapsed)
                blocking.append(
                    f"Cooldown active: {remaining}s remaining."
                )

        should_retrain = (
            feature_drifted
            and perf_drifted
            and samples_since_last_retrain >= min_samples
            and (
                self._last_retrain_time is None
                or (time.time() - self._last_retrain_time) >= self._cooldown_seconds
            )
        )

        decision = {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "blocking_reasons": blocking,
            "feature_drift": feature_drifted,
            "performance_drift": perf_drifted,
            "samples_since_last_retrain": samples_since_last_retrain,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if should_retrain:
            log.warning("RETRAINING TRIGGERED! Reasons: %s", reasons)
        else:
            log.info("Retraining NOT triggered. Blocking: %s", blocking or ["none"])

        self._log_decision(decision)
        return decision

    def record_retrain_completed(self) -> None:
        """Reset cooldown after a successful retraining run."""
        self._last_retrain_time = time.time()
        log.info("Retraining cooldown reset at %s", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def _log_decision(self, decision: dict) -> None:
        with open(self._retrain_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(decision) + "\n")
