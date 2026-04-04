"""Drift detection using PSI and Kolmogorov-Smirnov test.

PSI interpretation:
  PSI < 0.1  -> no significant drift
  PSI < 0.2  -> moderate drift
  PSI >= 0.2 -> significant drift, action required
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

log = get_logger(__name__)

_N_BINS = 10
_PSI_THRESHOLD = settings.monitoring.drift.psi_threshold
_KS_PVALUE = settings.monitoring.drift.ks_pvalue_threshold


class DriftDetector:
    """Detect feature and performance drift between reference and live distributions."""

    def __init__(self, reference_df: Optional[pd.DataFrame] = None) -> None:
        self._reference: Optional[pd.DataFrame] = None
        self._report_path = resolve(settings.monitoring.drift_report_path)

        if reference_df is not None:
            self.set_reference(reference_df)

    def set_reference(self, df: pd.DataFrame) -> None:
        """Set the reference (training) distribution."""
        self._reference = df.copy()
        log.info("Reference distribution set (%d samples, %d features)", *df.shape)

    def has_reference(self) -> bool:
        return self._reference is not None

    def detect_feature_drift(
        self,
        live_df: pd.DataFrame,
        features: Optional[list[str]] = None,
    ) -> dict:
        """Compute PSI and KS test for each feature.

        Returns a report dict with drift_detected, feature_results,
        drifted_features, n_live_samples, and timestamp.
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference() before detect_feature_drift().")

        features = features or [c for c in self._reference.columns if c != "timestamp"]
        results: dict = {}

        for feat in features:
            if feat not in self._reference.columns or feat not in live_df.columns:
                continue
            ref_vals = self._reference[feat].dropna().to_numpy(dtype=float)
            live_vals = live_df[feat].dropna().to_numpy(dtype=float)
            psi = self._psi(ref_vals, live_vals)
            ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, live_vals)
            drifted = bool((psi >= _PSI_THRESHOLD) or (ks_pvalue < _KS_PVALUE))

            results[feat] = {
                "psi": round(float(psi), 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 4),
                "drifted": drifted,
            }

        drifted_features = [f for f, v in results.items() if v["drifted"]]
        drift_detected = len(drifted_features) > 0

        report = {
            "drift_detected": drift_detected,
            "feature_results": results,
            "drifted_features": drifted_features,
            "n_live_samples": len(live_df),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self._save_report(report, "feature")

        if drift_detected:
            log.warning("FEATURE DRIFT detected! Drifted features: %s", drifted_features)
        else:
            log.info("No significant feature drift detected.")

        return report

    def detect_performance_drift(
        self,
        recent_rmse: float,
        baseline_rmse: float,
    ) -> dict:
        """Check if recent RMSE exceeds the baseline by the configured threshold."""
        threshold = settings.monitoring.performance.degradation_threshold
        pct_change = (recent_rmse - baseline_rmse) / max(baseline_rmse, 1e-9)
        degraded = pct_change > threshold

        report = {
            "drift_detected": degraded,
            "recent_rmse": round(recent_rmse, 4),
            "baseline_rmse": round(baseline_rmse, 4),
            "pct_change": round(pct_change * 100, 2),
            "threshold_pct": threshold * 100,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self._save_report(report, "performance")

        if degraded:
            log.warning(
                "PERFORMANCE DRIFT detected! RMSE increased by %.1f%% "
                "(%.4f -> %.4f, threshold=%.0f%%)",
                pct_change * 100, baseline_rmse, recent_rmse, threshold * 100,
            )
        return report

    @staticmethod
    def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = _N_BINS) -> float:
        """Population Stability Index."""
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] -= 1e-9
        bins[-1] += 1e-9

        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current, bins=bins)[0]

        ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
        cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def _save_report(self, report: dict, report_type: str) -> None:
        report["report_type"] = report_type
        with open(self._report_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(report, default=_json_default) + "\n")


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
