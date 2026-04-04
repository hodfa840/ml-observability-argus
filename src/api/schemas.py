"""
Pydantic schemas for all API request/response bodies.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

class PredictionRequest(BaseModel):
    """Features required for a single trip duration prediction."""
    passenger_count: int = Field(..., ge=1, le=6, description="Number of passengers")
    trip_distance: float = Field(..., gt=0, le=50, description="Trip distance in miles")
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_dow: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    pickup_month: int = Field(..., ge=1, le=12)
    pickup_is_weekend: int = Field(..., ge=0, le=1)
    rate_code_id: int = Field(1, ge=1, le=5)
    payment_type: int = Field(1, ge=1, le=4)
    pu_location_zone: int = Field(1, ge=1, le=50)
    do_location_zone: int = Field(1, ge=1, le=50)
    vendor_id: int = Field(1, ge=1, le=2)


class PredictionResponse(BaseModel):
    request_id: str
    predicted_duration_min: float
    model_version: str
    timestamp: str


# ----------------------------------------------------------
# Ground Truth / Delayed Feedback
# ----------------------------------------------------------

class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="Must match a prior prediction request_id")
    actual_duration_min: float = Field(..., gt=0, description="Ground truth trip duration")


class FeedbackResponse(BaseModel):
    request_id: str
    matched: bool
    message: str


# ----------------------------------------------------------
# Monitoring / Drift
# ----------------------------------------------------------

class DriftCheckResponse(BaseModel):
    drift_detected: bool
    root_cause: list[str]
    performance_drop: Optional[str]
    action: str
    feature_results: dict[str, Any]
    drifted_features: list[str]
    rca_details: Optional[list[dict]]
    timestamp: str


class PerformanceMetricsResponse(BaseModel):
    rmse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]
    n_samples: Optional[int]
    n_pending: int
    baseline_rmse: Optional[float]
    timestamp: str


# ----------------------------------------------------------
# Retraining
# ----------------------------------------------------------

class RetrainingResponse(BaseModel):
    triggered: bool
    promoted: Optional[bool]
    improvement_pct: Optional[float]
    root_causes: list[str]
    action: str
    message: str
    timestamp: str


# ----------------------------------------------------------
# Health
# ----------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    pending_feedback_count: int
    matched_feedback_count: int
    version: str
