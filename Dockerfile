# ============================================================
# Argus — combined API + Dashboard (Hugging Face Spaces / Docker)
# Exposes port 7860 (Streamlit) — FastAPI runs internally on 8000
# ============================================================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/logs data/raw data/processed \
             data/model_registry/champion \
             data/model_registry/challenger \
             data/mlruns && \
    chmod +x start.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    API_URL=http://localhost:8000

# HF Spaces requires port 7860; FastAPI runs on 8000 internally
EXPOSE 7860

CMD ["./start.sh"]
