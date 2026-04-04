"""Root-cause analysis for detected drift.

Ranks features by RCA score = PSI * (1 + model_importance).
This weights drift signal by model sensitivity — a drifted feature
that matters to the model is ranked higher than one the model ignores.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

log = get_logger(__name__)


class RootCauseAnalyzer:
    """Explain drift by combining PSI signal with feature importance."""

    def __init__(self, model: Optional[Any] = None, feature_names: Optional[list[str]] = None) -> None:
        self._model = model
        self._feature_names = feature_names or []
        self._importances: dict[str, float] = {}

        if model is not None and feature_names is not None:
            self._load_importances(model, feature_names)

    def set_model(self, model: Any, feature_names: list[str]) -> None:
        self._model = model
        self._feature_names = feature_names
        self._load_importances(model, feature_names)

    def analyze(self, drift_report: dict, top_k: int = 5) -> dict:
        """Produce a root-cause analysis from a drift detector report.

        Returns a dict with root_causes, primary_cause, explanation,
        and action_recommended.
        """
        feature_results: dict = drift_report.get("feature_results", {})
        drifted_features: list[str] = drift_report.get("drifted_features", [])

        if not drifted_features:
            return {
                "root_causes": [],
                "primary_cause": "none",
                "explanation": "No drift detected.",
                "action_recommended": "monitor",
            }

        rows = []
        for feat in drifted_features:
            psi = feature_results[feat]["psi"]
            importance = self._importances.get(feat, 0.0)
            rca_score = float(psi) * (1.0 + float(importance))
            rows.append({
                "feature": feat,
                "psi": round(psi, 4),
                "ks_stat": round(feature_results[feat].get("ks_stat", 0.0), 4),
                "ks_pvalue": round(feature_results[feat].get("ks_pvalue", 1.0), 4),
                "importance": round(float(importance), 4),
                "rca_score": round(rca_score, 4),
            })

        rows.sort(key=lambda r: r["rca_score"], reverse=True)
        top_causes = rows[:top_k]
        primary = top_causes[0]["feature"] if top_causes else "unknown"

        explanation = self._build_explanation(top_causes)
        action = self._recommend_action(top_causes)

        result = {
            "root_causes": top_causes,
            "primary_cause": primary,
            "explanation": explanation,
            "action_recommended": action,
        }

        log.info(
            "RCA complete — primary cause: %s (rca_score=%.4f), action: %s",
            primary, top_causes[0]["rca_score"] if top_causes else 0.0, action,
        )
        return result

    def _load_importances(self, model: Any, feature_names: list[str]) -> None:
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            self._importances = {
                name: float(imp) for name, imp in zip(feature_names, imps)
            }
        else:
            log.warning("Model has no feature_importances_; RCA scores will use PSI only.")

    def _build_explanation(self, causes: list[dict]) -> str:
        if not causes:
            return "No drift-causing features identified."
        lines = [
            f"  - {c['feature']}: PSI={c['psi']:.3f}, importance={c['importance']:.3f}"
            for c in causes
        ]
        top = causes[0]["feature"]
        return (
            f"Drift is primarily driven by '{top}'. "
            f"Top {len(causes)} contributing feature(s):\n" + "\n".join(lines)
        )

    @staticmethod
    def _recommend_action(causes: list[dict]) -> str:
        if not causes:
            return "monitor"
        max_psi = max(c["psi"] for c in causes)
        max_importance = max(c["importance"] for c in causes)
        if max_psi >= 0.25 and max_importance >= 0.1:
            return "retrain_immediately"
        elif max_psi >= 0.2:
            return "retrain_recommended"
        else:
            return "monitor_closely"
