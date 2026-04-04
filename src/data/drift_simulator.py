"""Drift simulation engine.

Supports four drift types:
  gradual  - features shift linearly over N steps
  sudden   - abrupt distribution change at a single point
  seasonal - sinusoidal oscillation
  mixed    - combination of gradual and seasonal
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Optional, Sequence

from src.utils.logging_config import get_logger

log = get_logger(__name__)

DriftType = Literal["gradual", "sudden", "seasonal", "mixed"]


class DriftSimulator:
    """Inject configurable drift into feature DataFrames."""

    DRIFTABLE_CONTINUOUS = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
    ]
    DRIFTABLE_CATEGORICAL = [
        "rate_code_id",
        "payment_type",
        "pu_location_zone",
        "do_location_zone",
        "vendor_id",
    ]

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)
        log.info("DriftSimulator initialised (seed=%d)", random_seed)

    def apply(
        self,
        df: pd.DataFrame,
        drift_type: DriftType = "gradual",
        affected_features: Optional[Sequence[str]] = None,
        severity: float = 1.0,
        step: int = 0,
        total_steps: int = 500,
    ) -> pd.DataFrame:
        """Apply drift to `df` and return a modified copy."""
        df = df.copy()

        if affected_features is None:
            n_features = self.rng.integers(2, 4)
            affected_features = list(
                self.rng.choice(
                    self.DRIFTABLE_CONTINUOUS,
                    size=min(n_features, len(self.DRIFTABLE_CONTINUOUS)),
                    replace=False,
                )
            )

        log.debug(
            "Applying %s drift (step=%d/%d, features=%s, severity=%.2f)",
            drift_type, step, total_steps, affected_features, severity,
        )

        if drift_type == "gradual":
            df = self._gradual(df, affected_features, severity, step, total_steps)
        elif drift_type == "sudden":
            df = self._sudden(df, affected_features, severity)
        elif drift_type == "seasonal":
            df = self._seasonal(df, affected_features, severity, step)
        elif drift_type == "mixed":
            df = self._gradual(df, affected_features[:1], severity * 0.5, step, total_steps)
            df = self._seasonal(df, affected_features[1:2], severity * 0.7, step)
        else:
            raise ValueError(f"Unknown drift_type: {drift_type!r}")

        return df

    def generate_drift_scenario(
        self,
        base_df: pd.DataFrame,
        drift_type: DriftType = "gradual",
        n_steps: int = 1000,
        severity: float = 1.0,
    ) -> tuple[list[pd.DataFrame], dict]:
        """Generate a full drift scenario as a sequence of DataFrames."""
        batch_size = max(1, len(base_df) // n_steps)
        drifted_batches: list[pd.DataFrame] = []
        metadata: dict = {
            "drift_type": drift_type,
            "n_steps": n_steps,
            "severity": severity,
            "affected_features": [],
        }

        n_features = self.rng.integers(2, 4)
        affected = list(
            self.rng.choice(
                self.DRIFTABLE_CONTINUOUS,
                size=min(n_features, len(self.DRIFTABLE_CONTINUOUS)),
                replace=False,
            )
        )
        metadata["affected_features"] = affected

        for step in range(n_steps):
            batch = base_df.sample(n=batch_size, replace=True, random_state=int(self.rng.integers(0, 100000)))
            drifted = self.apply(
                batch,
                drift_type=drift_type,
                affected_features=affected,
                severity=severity,
                step=step,
                total_steps=n_steps,
            )
            drifted_batches.append(drifted)

        log.info(
            "Generated drift scenario: type=%s, steps=%d, features=%s",
            drift_type, n_steps, affected,
        )
        return drifted_batches, metadata

    def _gradual(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        severity: float,
        step: int,
        total_steps: int,
    ) -> pd.DataFrame:
        progress = step / max(total_steps, 1)
        for feat in features:
            if feat not in df.columns:
                continue
            col = df[feat].to_numpy(dtype=float)
            shift = severity * progress * col.std() * 2.0
            df[feat] = col + shift + self.rng.normal(0, 0.1 * shift + 0.01, size=len(col))
        return df

    def _sudden(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        severity: float,
    ) -> pd.DataFrame:
        for feat in features:
            if feat not in df.columns:
                continue
            col = df[feat].to_numpy(dtype=float)
            df[feat] = col + severity * col.std() * 3.0
        return df

    def _seasonal(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        severity: float,
        step: int,
        period: int = 100,
    ) -> pd.DataFrame:
        angle = 2 * np.pi * (step % period) / period
        for feat in features:
            if feat not in df.columns:
                continue
            col = df[feat].to_numpy(dtype=float)
            amplitude = severity * col.std() * 1.5
            df[feat] = col + amplitude * np.sin(angle)
        return df
