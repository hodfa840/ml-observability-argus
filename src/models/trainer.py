"""Model training with MLflow experiment tracking."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.preprocessing import Preprocessor
from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class ModelTrainer:
    """Train and evaluate a GradientBoosting model with MLflow tracking."""

    def __init__(self) -> None:
        self.preprocessor = Preprocessor()
        self._setup_mlflow()

    def train(
        self,
        df: pd.DataFrame,
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> dict:
        """Train a new model on `df`.

        Returns a dict with: model, metrics, feature_importances, run_id, artifact_uri.
        """
        X, y = self.preprocessor.transform_with_target(df)
        if y is None:
            raise ValueError("Training DataFrame must contain the target column.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=settings.model.evaluation.test_size,
            random_state=settings.model.hyperparams.random_state,
        )

        hp = settings.model.hyperparams
        model = GradientBoostingRegressor(
            n_estimators=hp.n_estimators,
            max_depth=hp.max_depth,
            learning_rate=hp.learning_rate,
            subsample=hp.subsample,
            min_samples_split=hp.min_samples_split,
            random_state=hp.random_state,
        )

        with mlflow.start_run(run_name=run_name or f"train_{int(time.time())}") as run:
            mlflow.set_tags(tags or {})
            mlflow.log_params({
                "n_estimators": hp.n_estimators,
                "max_depth": hp.max_depth,
                "learning_rate": hp.learning_rate,
                "subsample": hp.subsample,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            })

            log.info("Training GradientBoosting on %d samples ...", len(X_train))
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            metrics = self._evaluate(model, X_test, y_test)
            metrics["train_time_sec"] = round(train_time, 2)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            fi = self._feature_importances(model)
            fi_path = resolve("data/logs/feature_importances.json")
            fi.to_json(fi_path, orient="records", indent=2)
            mlflow.log_artifact(str(fi_path))

            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri("model")

        log.info(
            "Training complete — RMSE=%.4f, MAE=%.4f, R2=%.4f (run_id=%s)",
            metrics["rmse"], metrics["mae"], metrics["r2"], run_id,
        )

        return {
            "model": model,
            "metrics": metrics,
            "feature_importances": fi,
            "run_id": run_id,
            "artifact_uri": artifact_uri,
            "preprocessor": self.preprocessor,
        }

    def _evaluate(
        self,
        model: GradientBoostingRegressor,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        y_pred = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}

    def _feature_importances(self, model: GradientBoostingRegressor) -> pd.DataFrame:
        names = self.preprocessor.feature_names()
        importances = model.feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _setup_mlflow(self) -> None:
        tracking_uri = resolve(settings.mlflow.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri.as_uri())
        mlflow.set_experiment(settings.mlflow.experiment_name)
        log.info("MLflow tracking -> %s", tracking_uri)
