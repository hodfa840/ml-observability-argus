"""File-based model registry with champion/challenger slots."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Any

import joblib

from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class ModelRegistry:
    """Load, save, and promote ML models between champion and challenger slots."""

    def __init__(self) -> None:
        self._champion_dir = resolve(settings.model.champion_path)
        self._challenger_dir = resolve(settings.model.challenger_path)

    def save_champion(self, model: Any, metadata: dict) -> Path:
        return self._save(model, metadata, self._champion_dir, "champion")

    def save_challenger(self, model: Any, metadata: dict) -> Path:
        return self._save(model, metadata, self._challenger_dir, "challenger")

    def load_champion(self) -> Optional[Any]:
        return self._load(self._champion_dir, "champion")

    def load_challenger(self) -> Optional[Any]:
        return self._load(self._challenger_dir, "challenger")

    def champion_metadata(self) -> Optional[dict]:
        return self._load_meta(self._champion_dir)

    def challenger_metadata(self) -> Optional[dict]:
        return self._load_meta(self._challenger_dir)

    def promote_challenger(self) -> bool:
        """Replace champion with challenger if challenger has lower RMSE by threshold."""
        champ_meta = self.champion_metadata()
        chal_meta = self.challenger_metadata()

        if chal_meta is None:
            log.warning("No challenger to promote.")
            return False

        challenger = self.load_challenger()
        if challenger is None:
            log.warning("Challenger model file missing.")
            return False

        threshold = settings.model.evaluation.promotion_threshold

        if champ_meta is not None:
            champ_rmse = champ_meta.get("metrics", {}).get("rmse", float("inf"))
            chal_rmse = chal_meta.get("metrics", {}).get("rmse", float("inf"))
            improvement = (champ_rmse - chal_rmse) / max(champ_rmse, 1e-9)

            if improvement < threshold:
                log.info(
                    "Challenger RMSE=%.4f vs Champion RMSE=%.4f — improvement %.2f%% "
                    "below threshold %.0f%%. Not promoting.",
                    chal_rmse, champ_rmse, improvement * 100, threshold * 100,
                )
                return False

            log.info(
                "Challenger RMSE=%.4f improves Champion RMSE=%.4f by %.2f%%. Promoting.",
                chal_rmse, champ_rmse, improvement * 100,
            )

        chal_meta["promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save_champion(challenger, chal_meta)
        return True

    def has_champion(self) -> bool:
        return (self._champion_dir / "model.joblib").exists()

    def _save(self, model: Any, metadata: dict, directory: Path, slot: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.joblib"
        meta_path = directory / "metadata.json"

        joblib.dump(model, model_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        log.info("Saved %s model -> %s", slot, model_path)
        return model_path

    def _load(self, directory: Path, slot: str) -> Optional[Any]:
        model_path = directory / "model.joblib"
        if not model_path.exists():
            log.debug("No %s model at %s", slot, model_path)
            return None
        model = joblib.load(model_path)
        log.debug("Loaded %s model from %s", slot, model_path)
        return model

    def _load_meta(self, directory: Path) -> Optional[dict]:
        meta_path = directory / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
