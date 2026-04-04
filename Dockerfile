# ============================================================
# Self-Healing ML System — Dockerfile (API)
# ============================================================
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying code (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data directories
RUN mkdir -p data/logs data/raw data/processed \
             data/model_registry/champion \
             data/model_registry/challenger \
             data/mlruns

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Train initial model if registry is empty, then start API
CMD ["sh", "-c", \
     "python scripts/train_initial_model.py && \
      uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2"]
