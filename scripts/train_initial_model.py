"""Bootstrap script: generate data, train the initial champion model.

Usage:
    python scripts/train_initial_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.generator import TaxiDataGenerator
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry
from src.utils.config import settings, resolve
from src.utils.logging_config import get_logger

log = get_logger("bootstrap")


def main() -> None:
    gen = TaxiDataGenerator(random_seed=settings.simulation.random_seed)
    n_train = settings.data.n_reference_samples * 2

    log.info("Generating %d training samples ...", n_train)
    train_df = gen.generate(n_samples=n_train)

    ref_df = gen.generate_reference(n_samples=settings.data.n_reference_samples)
    ref_path = resolve(settings.data.reference_dataset)
    ref_df.to_parquet(ref_path, index=False)
    log.info("Reference dataset saved -> %s (%d rows)", ref_path, len(ref_df))

    trainer = ModelTrainer()
    result = trainer.train(train_df, run_name="initial_champion", tags={"phase": "bootstrap"})

    model = result["model"]
    metrics = result["metrics"]

    log.info("Training metrics:")
    for k, v in metrics.items():
        log.info("  %-20s %s", k, v)

    registry = ModelRegistry()
    registry.save_champion(
        model,
        metadata={
            "metrics": metrics,
            "run_id": result["run_id"],
            "trained_at": pd.Timestamp.now().isoformat(),
            "training_samples": len(train_df),
            "note": "initial_champion",
            "feature_importances": result["feature_importances"].to_dict("records"),
        },
    )

    log.info("Champion model saved to registry.")
    print("\nNext steps:")
    print("  Start API:        uvicorn app:app --reload")
    print("  Start dashboard:  python -m streamlit run dashboard/app.py")
    print("  Run simulation:   python scripts/simulate_drift.py")


if __name__ == "__main__":
    main()
