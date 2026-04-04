"""End-to-end demo: runs the full self-healing ML lifecycle in-process.

No API server required. Demonstrates:
  1. Data generation
  2. Initial model training
  3. Drift injection
  4. Drift detection (PSI + KS)
  5. Root-cause analysis
  6. Retraining trigger evaluation
  7. Challenger training and promotion

Usage:
    python scripts/demo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.generator import TaxiDataGenerator
from src.data.drift_simulator import DriftSimulator
from src.data.preprocessing import Preprocessor
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry
from src.models.evaluator import ModelEvaluator
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.root_cause_analyzer import RootCauseAnalyzer
from src.retraining.trigger import RetrainingTrigger
from src.retraining.pipeline import RetrainingPipeline
from src.utils.logging_config import get_logger

log = get_logger("demo")

SEP = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main() -> None:
    print(SEP)
    print("  SELF-HEALING ML SYSTEM — DEMO")
    print(SEP)

    gen = TaxiDataGenerator(random_seed=42)
    simulator = DriftSimulator(random_seed=99)
    preprocessor = Preprocessor()

    section("STEP 1: Generate Training Data")
    train_df = gen.generate(n_samples=8000)
    ref_df = gen.generate_reference(n_samples=2000)
    print(f"  Training samples : {len(train_df):,}")
    print(f"  Reference samples: {len(ref_df):,}")

    section("STEP 2: Train Initial Champion Model")
    trainer = ModelTrainer()
    result = trainer.train(train_df, run_name="demo_champion")
    champion_model = result["model"]
    champion_metrics = result["metrics"]

    print(f"  RMSE : {champion_metrics['rmse']}")
    print(f"  MAE  : {champion_metrics['mae']}")
    print(f"  R2   : {champion_metrics['r2']}")
    print(f"  run_id: {result['run_id']}")

    registry = ModelRegistry()
    registry.save_champion(champion_model, {"metrics": champion_metrics, "run_id": result["run_id"]})

    section("STEP 3: Set Up Monitoring")
    detector = DriftDetector(reference_df=ref_df)
    monitor = PerformanceMonitor()
    rca = RootCauseAnalyzer(model=champion_model, feature_names=preprocessor.feature_names())
    monitor.set_baseline_rmse(champion_metrics["rmse"])
    print(f"  Baseline RMSE: {champion_metrics['rmse']}")

    pre_drift_df = gen.generate(n_samples=500)
    X_pre, y_pre = preprocessor.transform_with_target(pre_drift_df)
    preds_pre = champion_model.predict(X_pre)
    for i, (pred, actual) in enumerate(zip(preds_pre, y_pre)):
        rid = f"pre_{i}"
        monitor.log_prediction(rid, float(pred), {})
        monitor.log_ground_truth(rid, float(actual))

    pre_metrics = monitor.compute_metrics()
    print(f"  Pre-drift RMSE: {pre_metrics['rmse'] if pre_metrics else 'N/A'}")

    section("STEP 4: Inject Drift")
    live_df = gen.generate(n_samples=1000)
    drifted_df = simulator.apply(
        live_df[preprocessor.feature_names()],
        drift_type="sudden",
        severity=1.5,
        step=500,
        total_steps=1000,
    )

    print("  Drift type: sudden")
    for feat in ["trip_distance", "pickup_hour", "passenger_count"]:
        if feat in live_df.columns and feat in drifted_df.columns:
            orig_mean = live_df[feat].mean()
            drift_mean = drifted_df[feat].mean()
            print(f"    {feat:<25} mean: {orig_mean:.3f} -> {drift_mean:.3f}  (delta={drift_mean - orig_mean:+.3f})")

    X_drifted = drifted_df.reindex(columns=preprocessor.feature_names(), fill_value=0)
    y_drifted = live_df["trip_duration_min"].values
    preds_post = champion_model.predict(X_drifted)

    monitor2 = PerformanceMonitor()
    monitor2.set_baseline_rmse(champion_metrics["rmse"])
    for i, (pred, actual) in enumerate(zip(preds_post, y_drifted)):
        rid = f"post_{i}"
        monitor2.log_prediction(rid, float(pred), {})
        monitor2.log_ground_truth(rid, float(actual))

    post_metrics = monitor2.compute_metrics()
    print(f"\n  Post-drift RMSE: {post_metrics['rmse'] if post_metrics else 'N/A'}")

    section("STEP 5: Drift Detection")
    feature_drift_report = detector.detect_feature_drift(
        drifted_df.reindex(columns=preprocessor.feature_names(), fill_value=0),
        features=preprocessor.feature_names(),
    )
    perf_drift_report = detector.detect_performance_drift(
        recent_rmse=post_metrics["rmse"] if post_metrics else 999,
        baseline_rmse=champion_metrics["rmse"],
    )

    print(f"  Feature drift detected : {feature_drift_report['drift_detected']}")
    print(f"  Drifted features       : {feature_drift_report['drifted_features']}")
    print(f"  Performance degraded   : {perf_drift_report['drift_detected']}")
    if post_metrics:
        pct = (post_metrics["rmse"] - champion_metrics["rmse"]) / champion_metrics["rmse"] * 100
        print(f"  Performance drop       : {pct:.1f}%")

    section("STEP 6: Root-Cause Analysis")
    rca_result = rca.analyze(feature_drift_report)
    print(f"  Primary cause    : {rca_result['primary_cause']}")
    print(f"  Action           : {rca_result['action_recommended']}")
    print(f"  Explanation:\n    {rca_result['explanation'][:300]}")

    section("STEP 7: Retraining Decision")
    trigger = RetrainingTrigger()
    decision = trigger.should_retrain(
        feature_drift_report=feature_drift_report,
        performance_report=perf_drift_report,
        samples_since_last_retrain=2000,
    )
    print(f"  should_retrain : {decision['should_retrain']}")
    print(f"  Reasons        : {decision['reasons']}")
    print(f"  Blocking       : {decision['blocking_reasons']}")

    section("STEP 8: Retrain and Promote")
    retrain_df = drifted_df.copy()
    retrain_df["trip_duration_min"] = y_drifted

    pipeline = RetrainingPipeline()
    retrain_result = pipeline.run(
        training_df=retrain_df.reindex(
            columns=preprocessor.feature_names() + ["trip_duration_min"], fill_value=0
        ),
        eval_df=pre_drift_df.sample(200, random_state=42),
        rca_report=rca_result,
        tags={"trigger": "demo"},
    )

    print(f"  Promoted           : {retrain_result['promoted']}")
    print(f"  Improvement        : {retrain_result['improvement_pct']:.2f}%")
    print(f"  Champion RMSE      : {retrain_result['champion_metrics'].get('rmse', 'N/A')}")
    print(f"  Challenger RMSE    : {retrain_result['challenger_metrics'].get('rmse', 'N/A')}")

    section("STEP 9: Example API Response")
    api_response = {
        "drift_detected": feature_drift_report["drift_detected"],
        "root_cause": feature_drift_report["drifted_features"][:3],
        "performance_drop": f"{pct:.1f}%" if post_metrics else "N/A",
        "action": rca_result["action_recommended"],
        "rca_details": rca_result["root_causes"][:3],
        "retrain_triggered": decision["should_retrain"],
        "model_promoted": retrain_result["promoted"],
        "new_model_improvement": f"{retrain_result['improvement_pct']:.2f}%",
    }
    print(json.dumps(api_response, indent=2, default=str))

    print(f"\n{SEP}")
    print("  DEMO COMPLETE")
    print(f"  uvicorn app:app --reload")
    print(f"  python -m streamlit run dashboard/app.py")
    print(SEP)


if __name__ == "__main__":
    main()
