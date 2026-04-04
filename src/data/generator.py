"""Synthetic NYC-style taxi trip duration dataset generator.

Produces a realistic, time-dependent dataset where trip duration depends
non-linearly on distance, time-of-day, and day-of-week. Rush-hour patterns
create temporal non-stationarity. Noise is heteroscedastic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.utils.logging_config import get_logger

log = get_logger(__name__)


class TaxiDataGenerator:
    """Generate synthetic taxi trip duration data with realistic temporal patterns."""

    VENDOR_IDS = [1, 2]
    RATE_CODES = [1, 2, 3, 4, 5]
    PAYMENT_TYPES = [1, 2, 3, 4]
    PU_ZONES = list(range(1, 51))
    DO_ZONES = list(range(1, 51))

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)
        log.info("TaxiDataGenerator initialised (seed=%d)", random_seed)

    def generate(
        self,
        n_samples: int,
        start_date: Optional[datetime] = None,
        freq_seconds: int = 60,
    ) -> pd.DataFrame:
        """Generate a DataFrame of taxi trips."""
        if start_date is None:
            start_date = datetime(2023, 1, 1, 0, 0, 0)

        timestamps = [
            start_date + timedelta(seconds=i * freq_seconds)
            for i in range(n_samples)
        ]

        df = self._generate_features(timestamps)
        df["trip_duration_min"] = self._compute_duration(df)
        log.info("Generated %d trip samples starting %s", n_samples, start_date)
        return df

    def generate_reference(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate a stable reference dataset with no drift."""
        return self.generate(n_samples, start_date=datetime(2023, 1, 1))

    def _generate_features(self, timestamps: list[datetime]) -> pd.DataFrame:
        n = len(timestamps)
        ts = pd.to_datetime(timestamps)

        passenger_count = self.rng.integers(1, 7, size=n)
        trip_distance = self._realistic_distance(n)
        pickup_hour = ts.hour.to_numpy()
        pickup_dow = ts.dayofweek.to_numpy()
        pickup_month = ts.month.to_numpy()
        pickup_is_weekend = (pickup_dow >= 5).astype(int)
        rate_code_id = self.rng.choice(self.RATE_CODES, size=n, p=[0.85, 0.06, 0.04, 0.03, 0.02])
        payment_type = self.rng.choice(self.PAYMENT_TYPES, size=n, p=[0.65, 0.30, 0.03, 0.02])
        pu_location_zone = self.rng.choice(self.PU_ZONES, size=n)
        do_location_zone = self.rng.choice(self.DO_ZONES, size=n)
        vendor_id = self.rng.choice(self.VENDOR_IDS, size=n)

        return pd.DataFrame({
            "timestamp": ts,
            "vendor_id": vendor_id,
            "passenger_count": passenger_count,
            "trip_distance": trip_distance,
            "pickup_hour": pickup_hour,
            "pickup_dow": pickup_dow,
            "pickup_month": pickup_month,
            "pickup_is_weekend": pickup_is_weekend,
            "rate_code_id": rate_code_id,
            "payment_type": payment_type,
            "pu_location_zone": pu_location_zone,
            "do_location_zone": do_location_zone,
        })

    def _realistic_distance(self, n: int) -> np.ndarray:
        base = self.rng.lognormal(mean=0.8, sigma=0.7, size=n)
        long_trip_mask = self.rng.random(n) < 0.05
        base[long_trip_mask] *= self.rng.uniform(3, 8, size=long_trip_mask.sum())
        return np.clip(base, 0.1, 50.0)

    def _compute_duration(self, df: pd.DataFrame) -> np.ndarray:
        dist = df["trip_distance"].to_numpy()
        hour = df["pickup_hour"].to_numpy()
        dow = df["pickup_dow"].to_numpy()
        rate = df["rate_code_id"].to_numpy()

        base_duration = 3.5 * np.power(dist, 0.75) + 2.0

        rush_morning = ((hour >= 7) & (hour <= 9)).astype(float)
        rush_evening = ((hour >= 16) & (hour <= 19)).astype(float)
        weekday = (dow < 5).astype(float)
        congestion = 1.0 + 0.35 * rush_morning * weekday + 0.45 * rush_evening * weekday

        rate_effect = np.where(rate == 2, 8.0, np.where(rate == 3, 12.0, 0.0))

        night = ((hour >= 0) & (hour <= 5)).astype(float)
        speed_factor = 1.0 - 0.15 * night

        duration = base_duration * congestion * speed_factor + rate_effect

        noise_std = 0.1 * duration + 0.5
        noise = self.rng.normal(0, noise_std)

        return np.clip(duration + noise, 1.0, 120.0)
