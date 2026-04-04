"""Health check endpoint."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse
from src.utils.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness and readiness probe."""
    app_state = request.app.state
    return HealthResponse(
        status="ok",
        model_loaded=app_state.model is not None,
        model_version=app_state.model_version,
        uptime_seconds=round(time.time() - app_state.start_time, 1),
        pending_feedback_count=app_state.monitor.pending_count(),
        matched_feedback_count=app_state.monitor.matched_count(),
        version=settings.app.version,
    )
