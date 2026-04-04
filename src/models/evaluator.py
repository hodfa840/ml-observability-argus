"""Champion vs. challenger model comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.preprocessing import Preprocessor
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class ModelEvaluator:
    """Compare champion vs. challenger models on a held-out dataset."""

    def __init__(self) -> None:
        self.preprocessor = Preprocessor()

    def evaluate_single(self, model: Any, df: pd.DataFrame) -> dict:
        """Compute regression metrics for a single model."""
        X, y = self.preprocessor.transform_with_target(df)
        if y is None:
            raise ValueError("DataFrame must contain the target column.")
        y_pred = model.predict(X)
        return self._metrics(y.to_numpy(), y_pred)

    def compare(self, champion: Any, challenger: Any, df: pd.DataFrame) -> dict:
        """Side-by-side comparison returning metrics and a promotion recommendation."""
        champ_m = self.evaluate_single(champion, df)
        chal_m = self.evaluate_single(challenger, df)

        delta_rmse = chal_m["rmse"] - champ_m["rmse"]
        improvement_pct = -delta_rmse / max(champ_m["rmse"], 1e-9) * 100
        recommendation = "promote" if improvement_pct > 0 else "keep_champion"

        result = {
            "champion_metrics": champ_m,
            "challenger_metrics": chal_m,
            "delta_rmse": round(delta_rmse, 4),
            "improvement_pct": round(improvement_pct, 2),
            "recommendation": recommendation,
        }

        log.info(
            "Model comparison: Champion RMSE=%.4f | Challenger RMSE=%.4f | "
            "delta=%.4f (%.2f%%) -> %s",
            champ_m["rmse"], chal_m["rmse"], delta_rmse, improvement_pct,
            recommendation.upper(),
        )
        return result

    def _metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }
