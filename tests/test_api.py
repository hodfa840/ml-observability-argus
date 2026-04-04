"""
FastAPI endpoint tests using httpx + TestClient.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_endpoint(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_root_redirect(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "service" in data


class TestPrediction:
    _valid_payload = {
        "passenger_count": 2,
        "trip_distance": 3.5,
        "pickup_hour": 8,
        "pickup_dow": 1,
        "pickup_month": 3,
        "pickup_is_weekend": 0,
        "rate_code_id": 1,
        "payment_type": 1,
        "pu_location_zone": 10,
        "do_location_zone": 25,
        "vendor_id": 1,
    }

    def test_predict_valid_input(self, client: TestClient) -> None:
        r = client.post("/predict", json=self._valid_payload)
        # If model not trained yet, 503 is expected; otherwise 200
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "request_id" in data
            assert "predicted_duration_min" in data
            assert data["predicted_duration_min"] > 0

    def test_predict_invalid_input(self, client: TestClient) -> None:
        bad = dict(self._valid_payload, passenger_count=99)  # out of range
        r = client.post("/predict", json=bad)
        assert r.status_code == 422  # validation error

    def test_predict_missing_field(self, client: TestClient) -> None:
        bad = {k: v for k, v in self._valid_payload.items() if k != "trip_distance"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_feedback_unknown_id(self, client: TestClient) -> None:
        r = client.post(
            "/predict/feedback",
            json={"request_id": "nonexistent-id", "actual_duration_min": 12.5},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] is False


class TestMonitoring:
    def test_metrics_endpoint(self, client: TestClient) -> None:
        r = client.get("/monitor/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "n_pending" in data

    def test_history_endpoint(self, client: TestClient) -> None:
        r = client.get("/monitor/history?log_type=drift&limit=10")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_history_invalid_type(self, client: TestClient) -> None:
        r = client.get("/monitor/history?log_type=INVALID")
        assert r.status_code == 400
