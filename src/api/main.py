"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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
    bg_task = asyncio.create_task(_metrics_background_task(app))
    yield
    bg_task.cancel()
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
