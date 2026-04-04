"""Retraining orchestration: train challenger, compare, promote if better."""
from __future__ import annotations

import time
from typing import Optional, Any

import pandas as pd

from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry
from src.models.evaluator import ModelEvaluator
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class RetrainingPipeline:
    """End-to-end retraining orchestrator."""

    def __init__(self) -> None:
        self.trainer = ModelTrainer()
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator()

    def run(
        self,
        training_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        rca_report: Optional[dict] = None,
        tags: Optional[dict] = None,
    ) -> dict:
        """Execute a full retraining cycle.

        Trains a challenger, compares against champion on eval_df,
        promotes if challenger wins. Returns a result summary dict.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log.info("Retraining pipeline start (%s)", ts)

        run_tags = {
            "trigger": "drift_detected",
            "primary_root_cause": (rca_report or {}).get("primary_cause", "unknown"),
        }
        if tags:
            run_tags.update(tags)

        log.info("Training challenger model on %d samples ...", len(training_df))
        train_result = self.trainer.train(
            training_df,
            run_name=f"retrain_{int(time.time())}",
            tags=run_tags,
        )
        challenger = train_result["model"]
        chal_metrics = train_result["metrics"]
        run_id = train_result["run_id"]

        self.registry.save_challenger(
            challenger,
            metadata={
                "metrics": chal_metrics,
                "run_id": run_id,
                "trained_at": ts,
                "training_samples": len(training_df),
                "rca": rca_report,
            },
        )

        champion = self.registry.load_champion()
        promoted = False
        comparison: dict = {}

        if champion is not None:
            comparison = self.evaluator.compare(champion, challenger, eval_df)
            log.info(
                "Comparison result: improvement=%.2f%% -> %s",
                comparison["improvement_pct"], comparison["recommendation"],
            )
            if comparison["recommendation"] == "promote":
                promoted = self.registry.promote_challenger()
        else:
            log.info("No champion exists; auto-promoting first trained model.")
            self.registry.save_champion(
                challenger,
                metadata={
                    "metrics": chal_metrics,
                    "run_id": run_id,
                    "trained_at": ts,
                    "training_samples": len(training_df),
                    "note": "initial_champion",
                },
            )
            promoted = True
            comparison = {
                "champion_metrics": {},
                "challenger_metrics": chal_metrics,
                "improvement_pct": 100.0,
            }

        result = {
            "promoted": promoted,
            "challenger_metrics": chal_metrics,
            "champion_metrics": comparison.get("champion_metrics", {}),
            "improvement_pct": comparison.get("improvement_pct", 0.0),
            "run_id": run_id,
            "root_causes": (rca_report or {}).get("root_causes", []),
            "timestamp": ts,
        }

        log.info("Retraining pipeline end — promoted=%s, RMSE=%.4f", promoted, chal_metrics["rmse"])
        return result
