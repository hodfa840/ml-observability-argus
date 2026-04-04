"""Rolling performance monitor with delayed-feedback support.

Predictions are logged immediately. Ground truth is matched by request_id
and may arrive hours after the prediction was made.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional, Deque

import numpy as np

from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class _PredictionRecord:
    __slots__ = ("request_id", "prediction", "ground_truth", "timestamp", "features")

    def __init__(
        self,
        request_id: str,
        prediction: float,
        timestamp: float,
        features: dict,
    ) -> None:
        self.request_id = request_id
        self.prediction = prediction
        self.ground_truth: Optional[float] = None
        self.timestamp = timestamp
        self.features = features


class PerformanceMonitor:
    """Rolling window performance tracker with delayed-feedback support."""

    def __init__(self) -> None:
        window_size = settings.monitoring.window_size
        self._pending: dict[str, _PredictionRecord] = {}
        self._matched: Deque[_PredictionRecord] = deque(maxlen=window_size)
        self._baseline_rmse: Optional[float] = None
        self._perf_log_path = resolve(settings.monitoring.performance_log_path)

    def log_prediction(self, request_id: str, prediction: float, features: dict) -> None:
        record = _PredictionRecord(
            request_id=request_id,
            prediction=prediction,
            timestamp=time.time(),
            features=features,
        )
        self._pending[request_id] = record

    def log_ground_truth(self, request_id: str, actual: float) -> bool:
        """Match ground truth to a pending prediction. Returns True if matched."""
        record = self._pending.pop(request_id, None)
        if record is None:
            log.warning("Ground truth for unknown request_id=%s ignored.", request_id)
            return False

        record.ground_truth = actual
        self._matched.append(record)
        self._append_feedback_log(record)
        log.debug(
            "Ground truth matched: request_id=%s pred=%.2f actual=%.2f delay=%.1fs",
            request_id, record.prediction, actual, time.time() - record.timestamp,
        )
        return True

    def compute_metrics(self) -> Optional[dict]:
        """Compute rolling metrics over matched predictions. Returns None if insufficient data."""
        min_samples = settings.monitoring.min_samples_for_evaluation
        matched = list(self._matched)

        if len(matched) < min_samples:
            log.debug("Only %d matched samples, need %d.", len(matched), min_samples)
            return None

        preds = np.array([r.prediction for r in matched])
        actuals = np.array([r.ground_truth for r in matched])

        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        mae = float(np.mean(np.abs(preds - actuals)))
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-9))

        metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "n_samples": len(matched),
            "n_pending": len(self._pending),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self._append_performance_log(metrics)
        return metrics

    def set_baseline_rmse(self, rmse: float) -> None:
        self._baseline_rmse = rmse
        log.info("Baseline RMSE set to %.4f", rmse)

    def get_baseline_rmse(self) -> Optional[float]:
        return self._baseline_rmse

    def matched_count(self) -> int:
        return len(self._matched)

    def pending_count(self) -> int:
        return len(self._pending)

    def get_matched_dataframe(self):
        import pandas as pd
        records = list(self._matched)
        if not records:
            return pd.DataFrame()
        rows = []
        for r in records:
            row = dict(r.features)
            row["prediction"] = r.prediction
            row["ground_truth"] = r.ground_truth
            row["timestamp"] = r.timestamp
            rows.append(row)
        return pd.DataFrame(rows)

    def _append_performance_log(self, metrics: dict) -> None:
        with open(self._perf_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")

    def _append_feedback_log(self, record: _PredictionRecord) -> None:
        feedback_path = resolve(settings.delayed_feedback.feedback_log_path)
        entry = {
            "request_id": record.request_id,
            "prediction": record.prediction,
            "ground_truth": record.ground_truth,
            "prediction_timestamp": record.timestamp,
            "feedback_timestamp": time.time(),
            "delay_seconds": time.time() - record.timestamp,
        }
        with open(feedback_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
