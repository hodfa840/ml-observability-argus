"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.data.preprocessing import Preprocessor
from src.models.registry import ModelRegistry
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.root_cause_analyzer import RootCauseAnalyzer
from src.retraining.trigger import RetrainingTrigger
from src.retraining.pipeline import RetrainingPipeline
from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

log = get_logger(__name__)


async def _metrics_background_task(app: FastAPI) -> None:
    """Every 15s: log metrics to keep performance.jsonl up to date."""
    while True:
        await asyncio.sleep(15)
        try:
            app.state.monitor.compute_metrics()
        except Exception as exc:  # noqa: BLE001
            log.debug("Background metrics error: %s", exc)


async def _generate_synthetic_predictions(app: FastAPI) -> None:
    """Background task: generate synthetic predictions + ground truth to keep charts live.

    Simulates realistic patterns:
    - Base prediction error + noise
    - Gradual drift over time (RMSE increases ~2% per 100 requests)
    - Random spikes (5% of requests degrade suddenly)
    - Seasonal variation (±10% oscillation)
    """
    await asyncio.sleep(3)  # Wait for app startup
    request_count = 0

    while True:
        try:
            await asyncio.sleep(random.uniform(4, 8))  # Every 4-8 seconds

            monitor = app.state.monitor
            if monitor.pending_count() > 100:
                # Don't overload pending queue
                continue

            request_count += 1
            request_id = f"synthetic_{request_count}_{int(time.time() * 1000)}"

            # Generate synthetic features (random values similar to NYC taxi data)
            features = {
                "trip_distance": random.uniform(0.5, 25),
                "trip_duration": random.uniform(120, 3600),
                "hour_of_day": random.randint(0, 23),
                "day_of_week": random.randint(0, 6),
                "passenger_count": random.randint(1, 6),
                "is_holiday": random.choice([0, 1]),
            }

            # Make prediction (mock: we don't have model here, use realistic value)
            base_fare = 2.5 + features["trip_distance"] * 2.5
            prediction = base_fare + random.gauss(0, 0.5)

            # Log prediction immediately
            monitor.log_prediction(request_id, prediction, features)

            # After ~5 seconds, log ground truth with realistic error patterns
            # This happens asynchronously to simulate delayed feedback
            asyncio.create_task(
                _log_synthetic_ground_truth(monitor, request_id, prediction, request_count)
            )

            log.debug(f"Synthetic prediction logged: {request_id}")

        except Exception as exc:
            log.warning("Synthetic prediction generation error: %s", exc)


async def _log_synthetic_ground_truth(
    monitor,
    request_id: str,
    prediction: float,
    request_count: int,
) -> None:
    """Simulate delayed feedback with realistic error patterns."""
    await asyncio.sleep(random.uniform(2, 6))  # Delayed feedback

    try:
        # Simulate realistic error with drift
        # Baseline error: ~12% of predicted value
        baseline_error_pct = 0.12

        # Gradually increasing drift: every 100 requests, error grows ~2%
        drift_factor = 1.0 + (request_count / 100) * 0.02

        # Random spikes: 5% of requests degrade suddenly
        spike_factor = 1.0
        if random.random() < 0.05:
            spike_factor = random.uniform(1.3, 1.8)  # Sudden +30-80% error

        # Seasonal oscillation: sin wave over ~500 requests
        seasonal = 1.0 + 0.1 * np.sin(request_count * np.pi / 250)

        # Combine factors
        total_error_pct = baseline_error_pct * drift_factor * spike_factor * seasonal

        # Actual ground truth
        actual = prediction * (1.0 + random.gauss(total_error_pct, 0.03))
        actual = max(1.0, actual)  # Ensure positive

        monitor.log_ground_truth(request_id, actual)

    except Exception as exc:
        log.debug(f"Synthetic ground truth error for {request_id}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    log.info("Self-Healing ML API starting up")
    state = app.state
    state.start_time = time.time()

    state.registry = ModelRegistry()
    state.preprocessor = Preprocessor()
    state.monitor = PerformanceMonitor()
    state.drift_detector = DriftDetector()
    state.rca = RootCauseAnalyzer()
    state.trigger = RetrainingTrigger()
    state.retrain_pipeline = RetrainingPipeline()
    state.samples_since_last_retrain = 0
    state.model_version = "none"

    state.model = state.registry.load_champion()
    if state.model is not None:
        champion_meta = state.registry.champion_metadata() or {}
        state.model_version = champion_meta.get("run_id", "v1")[:8]
        baseline_rmse = (champion_meta.get("metrics") or {}).get("rmse")
        if baseline_rmse:
            state.monitor.set_baseline_rmse(baseline_rmse)
        log.info("Champion model loaded (version=%s)", state.model_version)
        state.rca.set_model(state.model, state.preprocessor.feature_names())
    else:
        log.warning("No champion model found. Run scripts/train_initial_model.py first.")

    ref_path = resolve(settings.data.reference_dataset)
    if ref_path.exists():
        ref_df = pd.read_parquet(ref_path)
        state.drift_detector.set_reference(ref_df)
        log.info("Reference dataset loaded (%d rows)", len(ref_df))
    else:
        log.warning("Reference dataset not found at %s.", ref_path)

    log.info("API ready")
    metrics_task = asyncio.create_task(_metrics_background_task(app))
    synthetic_task = asyncio.create_task(_generate_synthetic_predictions(app))
    yield
    metrics_task.cancel()
    synthetic_task.cancel()
    log.info("Self-Healing ML API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Self-Healing ML System",
        description=(
            "Production ML system with drift detection, root-cause analysis, "
            "delayed feedback handling, and intelligent retraining."
        ),
        version=settings.app.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.api.routes.predict import router as predict_router
    from src.api.routes.monitor import router as monitor_router
    from src.api.routes.health import router as health_router

    app.include_router(predict_router)
    app.include_router(monitor_router)
    app.include_router(health_router)

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({
            "service": "Self-Healing ML System",
            "version": settings.app.version,
            "docs": "/docs",
            "health": "/health",
        })

    return app
