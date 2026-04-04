"""
Monitoring & drift endpoints.

GET  /monitor/metrics     → rolling performance metrics
GET  /monitor/drift       → run drift check on recent live data vs reference
POST /monitor/retrain     → manually trigger retraining evaluation
GET  /monitor/history     → drift + performance history
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    DriftCheckResponse,
    PerformanceMetricsResponse,
    RetrainingResponse,
)
from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

router = APIRouter(prefix="/monitor", tags=["Monitoring"])
log = get_logger(__name__)


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_metrics(request: Request) -> PerformanceMetricsResponse:
    """Return rolling performance metrics from matched predictions."""
    monitor = request.app.state.monitor
    metrics = monitor.compute_metrics()
    baseline = monitor.get_baseline_rmse()

    if metrics is None:
        return PerformanceMetricsResponse(
            rmse=None, mae=None, r2=None,
            n_samples=monitor.matched_count(),
            n_pending=monitor.pending_count(),
            baseline_rmse=baseline,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    return PerformanceMetricsResponse(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        n_samples=metrics["n_samples"],
        n_pending=monitor.pending_count(),
        baseline_rmse=baseline,
        timestamp=metrics["timestamp"],
    )


@router.get("/drift", response_model=DriftCheckResponse)
async def check_drift(request: Request) -> DriftCheckResponse:
    """
    Run drift detection on live data collected since last check.

    Combines feature drift (PSI/KS) + performance drift into one report
    and runs root-cause analysis if drift is detected.
    """
    app_state = request.app.state
    drift_detector = app_state.drift_detector
    rca = app_state.rca
    monitor = app_state.monitor

    if not drift_detector.has_reference():
        raise HTTPException(status_code=503, detail="Reference dataset not loaded yet.")

    # Get live feature data from matched predictions
    live_df = monitor.get_matched_dataframe()
    min_samples = settings.monitoring.drift.min_samples_for_drift_test

    if len(live_df) < min_samples:
        return DriftCheckResponse(
            drift_detected=False,
            root_cause=[],
            performance_drop=None,
            action="insufficient_data",
            feature_results={},
            drifted_features=[],
            rca_details=None,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    # Feature drift
    feature_cols = [c for c in settings.data.features if c in live_df.columns]
    feat_report = drift_detector.detect_feature_drift(live_df, features=feature_cols)

    # Performance drift
    metrics = monitor.compute_metrics()
    baseline = monitor.get_baseline_rmse()
    perf_report: dict = {"drift_detected": False}
    performance_drop = None

    if metrics and baseline:
        perf_report = drift_detector.detect_performance_drift(metrics["rmse"], baseline)
        if perf_report["drift_detected"]:
            pct = perf_report["pct_change"]
            performance_drop = f"{pct:.1f}%"

    # RCA
    rca_result: dict = {}
    if feat_report["drift_detected"]:
        rca_result = rca.analyze(feat_report)

    # Retraining decision
    trigger = app_state.trigger
    decision = trigger.should_retrain(
        feature_drift_report=feat_report,
        performance_report=perf_report,
        samples_since_last_retrain=app_state.samples_since_last_retrain,
    )

    action = "no_action"
    if decision["should_retrain"]:
        action = "retraining_triggered"
        # Fire async retraining (background)
        import asyncio
        asyncio.create_task(_run_retraining(app_state, rca_result))
    elif feat_report["drift_detected"]:
        action = "drift_detected_monitoring"

    return DriftCheckResponse(
        drift_detected=feat_report["drift_detected"] or perf_report.get("drift_detected", False),
        root_cause=feat_report.get("drifted_features", []),
        performance_drop=performance_drop,
        action=action,
        feature_results=feat_report.get("feature_results", {}),
        drifted_features=feat_report.get("drifted_features", []),
        rca_details=rca_result.get("root_causes"),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@router.post("/retrain", response_model=RetrainingResponse)
async def trigger_retrain(request: Request) -> RetrainingResponse:
    """Manually trigger a retraining run (bypasses drift gates)."""
    app_state = request.app.state
    monitor = app_state.monitor
    pipeline = app_state.retrain_pipeline

    live_df = monitor.get_matched_dataframe()
    if len(live_df) < 50:
        return RetrainingResponse(
            triggered=False,
            promoted=None,
            improvement_pct=None,
            root_causes=[],
            action="insufficient_labeled_data",
            message=f"Only {len(live_df)} labeled samples available. Need at least 50.",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    result = pipeline.run(
        training_df=live_df,
        eval_df=live_df.sample(frac=0.2, random_state=42),
        tags={"trigger": "manual"},
    )

    if result["promoted"]:
        model = app_state.registry.load_champion()
        if model:
            app_state.model = model
            app_state.model_version = f"v{int(time.time())}"
            if result["challenger_metrics"].get("rmse"):
                monitor.set_baseline_rmse(result["challenger_metrics"]["rmse"])

    return RetrainingResponse(
        triggered=True,
        promoted=result["promoted"],
        improvement_pct=result.get("improvement_pct"),
        root_causes=[c["feature"] for c in result.get("root_causes", [])],
        action="champion_promoted" if result["promoted"] else "challenger_not_promoted",
        message="Manual retraining completed.",
        timestamp=result["timestamp"],
    )


@router.get("/history")
async def get_history(
    limit: int = 100,
    log_type: str = "drift",
) -> list[dict]:
    """
    Return recent log entries.

    log_type: 'drift' | 'performance' | 'retrain' | 'feedback'
    """
    log_paths = {
        "drift": resolve(settings.monitoring.drift_report_path),
        "performance": resolve(settings.monitoring.performance_log_path),
        "retrain": resolve(settings.retraining.retrain_log_path),
        "feedback": resolve(settings.delayed_feedback.feedback_log_path),
    }

    path = log_paths.get(log_type)
    if path is None:
        raise HTTPException(status_code=400, detail=f"Unknown log_type '{log_type}'")

    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


# ------------------------------------------------------------------
# Background task helper
# ------------------------------------------------------------------

async def _run_retraining(app_state: Any, rca_result: dict) -> None:
    """Fire-and-forget retraining coroutine."""
    try:
        monitor = app_state.monitor
        pipeline = app_state.retrain_pipeline
        live_df = monitor.get_matched_dataframe()

        if len(live_df) < 50:
            log.warning("Not enough labeled samples for retraining (%d).", len(live_df))
            return

        result = pipeline.run(
            training_df=live_df,
            eval_df=live_df.sample(frac=0.2, random_state=42),
            rca_report=rca_result,
            tags={"trigger": "auto_drift"},
        )

        if result["promoted"]:
            model = app_state.registry.load_champion()
            if model:
                app_state.model = model
                app_state.model_version = f"v{int(time.time())}"
                if result["challenger_metrics"].get("rmse"):
                    monitor.set_baseline_rmse(result["challenger_metrics"]["rmse"])
            app_state.trigger.record_retrain_completed()
            app_state.samples_since_last_retrain = 0
            log.info("Auto-retraining complete — new champion promoted.")
    except Exception as exc:
        log.error("Background retraining failed: %s", exc, exc_info=True)
