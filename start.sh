#!/bin/bash
set -e

echo "=== Argus: starting up ==="

mkdir -p data/logs data/raw data/processed \
         data/model_registry/champion \
         data/model_registry/challenger \
         data/mlruns

echo "--- Training initial champion model ---"
python scripts/train_initial_model.py

echo "--- Starting FastAPI on port 8000 ---"
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 &

echo "--- Waiting for API to be ready ---"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API ready after ${i}s"
        break
    fi
    sleep 2
done

echo "--- Running simulation to populate monitoring data ---"
python scripts/simulate_drift.py \
    --steps 300 \
    --delay 0.05 \
    --drift-type gradual \
    --batch-size 10 &

echo "--- Starting Streamlit on port 7860 ---"
exec streamlit run dashboard/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
