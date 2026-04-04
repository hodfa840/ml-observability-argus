"""Production drift simulation script.

Sends requests to the FastAPI endpoint to simulate traffic with configurable
drift types and delayed feedback.

Usage:
    python scripts/simulate_drift.py
    python scripts/simulate_drift.py --drift-type sudden
    python scripts/simulate_drift.py --drift-type mixed --steps 1000 --delay 0.05
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generator import TaxiDataGenerator
from src.data.drift_simulator import DriftSimulator
from src.utils.config import settings
from src.utils.logging_config import get_logger

log = get_logger("simulate_drift")

API_URL = "http://localhost:8000"
DRIFT_TYPES = ["gradual", "sudden", "seasonal", "mixed"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate production drift")
    p.add_argument("--drift-type", choices=DRIFT_TYPES, default="gradual")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--delay", type=float, default=0.1)
    p.add_argument("--feedback-lag", type=int, default=20)
    p.add_argument("--api-url", default=API_URL)
    p.add_argument("--severity", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api = args.api_url

    try:
        health = requests.get(f"{api}/health", timeout=5).json()
        log.info("API online — model=%s, uptime=%.0fs", health.get("model_version"), health.get("uptime_seconds"))
    except Exception as e:
        log.error("Cannot reach API at %s: %s", api, e)
        log.error("Start the API first: uvicorn app:app --reload")
        sys.exit(1)

    gen = TaxiDataGenerator(random_seed=42)
    simulator = DriftSimulator(random_seed=99)

    base_df = gen.generate(n_samples=args.steps * args.batch_size)
    feature_cols = [c for c in settings.data.features if c in base_df.columns]

    log.info("Starting drift simulation: type=%s, steps=%d, severity=%.2f",
             args.drift_type, args.steps, args.severity)

    pending_feedback: deque = deque()
    stats = {"predictions": 0, "feedback_sent": 0, "drift_alerts": 0, "retrain_events": 0}

    for step in range(args.steps):
        batch_start = (step * args.batch_size) % len(base_df)
        batch = base_df.iloc[batch_start: batch_start + args.batch_size].copy()

        if args.drift_type != "sudden" or step == args.steps // 3:
            drifted = simulator.apply(
                batch[feature_cols],
                drift_type=args.drift_type,
                severity=args.severity,
                step=step,
                total_steps=args.steps,
            )
            for col in feature_cols:
                if col in drifted.columns:
                    batch[col] = drifted[col].values

        for _, row in batch.iterrows():
            payload = {
                "passenger_count": int(max(1, min(6, round(row.get("passenger_count", 2))))),
                "trip_distance": float(max(0.1, min(50, row.get("trip_distance", 3)))),
                "pickup_hour": int(max(0, min(23, round(row.get("pickup_hour", 8))))),
                "pickup_dow": int(max(0, min(6, round(row.get("pickup_dow", 1))))),
                "pickup_month": int(max(1, min(12, round(row.get("pickup_month", 1))))),
                "pickup_is_weekend": int(row.get("pickup_is_weekend", 0)),
                "rate_code_id": int(max(1, min(5, round(row.get("rate_code_id", 1))))),
                "payment_type": int(max(1, min(4, round(row.get("payment_type", 1))))),
                "pu_location_zone": int(max(1, min(50, round(row.get("pu_location_zone", 10))))),
                "do_location_zone": int(max(1, min(50, round(row.get("do_location_zone", 25))))),
                "vendor_id": int(max(1, min(2, round(row.get("vendor_id", 1))))),
            }
            try:
                r = requests.post(f"{api}/predict", json=payload, timeout=5)
                if r.status_code == 200:
                    result = r.json()
                    actual = float(row.get("trip_duration_min", result["predicted_duration_min"] * random.uniform(0.8, 1.2)))
                    pending_feedback.append((step, result["request_id"], actual))
                    stats["predictions"] += 1
            except Exception as e:
                log.debug("Prediction failed: %s", e)

        while pending_feedback and (step - pending_feedback[0][0]) >= args.feedback_lag:
            _, req_id, actual = pending_feedback.popleft()
            try:
                requests.post(
                    f"{api}/predict/feedback",
                    json={"request_id": req_id, "actual_duration_min": actual},
                    timeout=3,
                )
                stats["feedback_sent"] += 1
            except Exception:
                pass

        if step > 0 and step % 50 == 0:
            try:
                r = requests.get(f"{api}/monitor/drift", timeout=10)
                drift = r.json()
                if drift.get("drift_detected"):
                    stats["drift_alerts"] += 1
                    log.warning(
                        "Step %d — DRIFT DETECTED features=%s action=%s",
                        step, drift.get("drifted_features"), drift.get("action"),
                    )
                    if drift.get("action") == "retraining_triggered":
                        stats["retrain_events"] += 1
                else:
                    log.info("Step %d — predictions=%d feedback=%d",
                             step, stats["predictions"], stats["feedback_sent"])
            except Exception as e:
                log.debug("Drift check failed: %s", e)

        time.sleep(args.delay)

    log.info("Simulation complete: predictions=%d, feedback=%d, drift_alerts=%d, retrain=%d",
             stats["predictions"], stats["feedback_sent"], stats["drift_alerts"], stats["retrain_events"])


if __name__ == "__main__":
    main()
