"""Feature preprocessing pipeline shared by training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.utils.config import settings
from src.utils.logging_config import get_logger

log = get_logger(__name__)

FEATURE_COLS: list[str] = settings.data.features
TARGET_COL: str = settings.data.target

_CLIP_RANGES: dict[str, tuple[float, float]] = {
    "trip_distance": (0.1, 50.0),
    "passenger_count": (1.0, 6.0),
    "pickup_hour": (0.0, 23.0),
    "pickup_dow": (0.0, 6.0),
    "pickup_month": (1.0, 12.0),
}

_CATEGORICAL_COLS: set[str] = {
    "vendor_id",
    "rate_code_id",
    "payment_type",
    "pu_location_zone",
    "do_location_zone",
    "pickup_is_weekend",
}


class Preprocessor:
    """Stateless preprocessing utility."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a clean feature DataFrame aligned to FEATURE_COLS."""
        df = df.copy()

        for col, (lo, hi) in _CLIP_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        for col in _CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(int)

        for col in FEATURE_COLS:
            if col not in df.columns:
                log.warning("Missing feature '%s' — filling with 0", col)
                df[col] = 0

        return df[FEATURE_COLS]

    def transform_with_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Return (X, y). y is None if target not in df."""
        y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
        X = self.transform(df)
        return X, y

    def feature_names(self) -> list[str]:
        return list(FEATURE_COLS)
