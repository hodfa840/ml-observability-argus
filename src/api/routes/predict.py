"""
Prediction endpoint.

POST /predict  →  returns duration prediction + logs to monitor
POST /feedback →  submit delayed ground truth
"""
from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from src.utils.config import resolve, settings
from src.utils.logging_config import get_logger

router = APIRouter(prefix="/predict", tags=["Prediction"])
log = get_logger(__name__)

_PREDICTION_LOG = resolve(settings.api.prediction_log_path)


@router.post("", response_model=PredictionResponse)
async def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    """
    Predict taxi trip duration from input features.

    The prediction is:
    1. Logged to disk (for drift analysis)
    2. Registered in the performance monitor (awaiting ground truth)
    """
    app_state = request.app.state
    model = app_state.model
    preprocessor = app_state.preprocessor
    monitor = app_state.monitor

    if model is None:
        raise HTTPException(status_code=503, detail="No trained model available. Please train first.")

    features = body.model_dump()
    request_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    df = pd.DataFrame([features])
    X = preprocessor.transform(df)
    prediction = float(model.predict(X)[0])
    prediction = max(1.0, round(prediction, 2))

    # Log prediction
    log_entry = {
        "request_id": request_id,
        "timestamp": timestamp,
        "features": features,
        "prediction": prediction,
        "model_version": app_state.model_version,
    }
    with open(_PREDICTION_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry) + "\n")

    # Register in performance monitor
    monitor.log_prediction(request_id, prediction, features)
    app_state.samples_since_last_retrain += 1

    log.debug("Prediction: request_id=%s  pred=%.2f min", request_id, prediction)

    return PredictionResponse(
        request_id=request_id,
        predicted_duration_min=prediction,
        model_version=app_state.model_version,
        timestamp=timestamp,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """
    Submit the delayed ground truth for a previous prediction.

    This simulates labels arriving hours after the prediction was made
    (e.g., after the trip completes and data is reconciled).
    """
    monitor = request.app.state.monitor
    matched = monitor.log_ground_truth(body.request_id, body.actual_duration_min)

    if matched:
        msg = f"Ground truth matched for request_id={body.request_id}"
    else:
        msg = f"No pending prediction found for request_id={body.request_id}"

    return FeedbackResponse(request_id=body.request_id, matched=matched, message=msg)
