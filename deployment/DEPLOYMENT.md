# Deployment Guide

> **Goal**: Deploy the API for free on Render, the dashboard on Streamlit Community Cloud.
> No credit card required for either.

---

## Part 1 — Deploy API on Render (Free)

### Prerequisites
- GitHub account
- Render account at [render.com](https://render.com) (sign up with GitHub)

### Steps

1. **Push your repo to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Self-Healing ML System"
   git remote add origin https://github.com/YOUR_USERNAME/self-healing-ml.git
   git push -u origin main
   ```

2. **Create a New Web Service on Render**
   - Go to [render.com/dashboard](https://dashboard.render.com)
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repo
   - Set these fields:

   | Field | Value |
   |-------|-------|
   | **Name** | `self-healing-ml-api` |
   | **Runtime** | `Python 3` |
   | **Branch** | `main` |
   | **Root Directory** | *(leave empty)* |
   | **Build Command** | `pip install -r requirements.txt && python scripts/train_initial_model.py` |
   | **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
   | **Plan** | `Free` |

3. **Add Environment Variable**
   - Key: `PYTHONPATH`, Value: `/opt/render/project/src`

4. **Add a Disk** (for model persistence)
   - Click **"Add Disk"**
   - Mount path: `/opt/render/project/src/data`
   - Size: 1 GB (free tier allows this)

5. **Deploy**
   - Click **"Create Web Service"**
   - First deploy takes ~5 minutes (model training included)

6. **Verify**
   ```bash
   curl https://self-healing-ml-api.onrender.com/health
   ```

> **Note on free tier**: Render spins down free services after 15 minutes of inactivity.
> The first request after idle takes ~30 seconds to cold-start.
> Use a cron pinger (e.g., UptimeRobot) to keep it warm.

---

## Part 2 — Deploy Dashboard on Streamlit Community Cloud (Free)

### Prerequisites
- GitHub account with the repo pushed
- Streamlit account at [streamlit.io](https://streamlit.io) (sign up with GitHub)

### Steps

1. **Copy Streamlit config to repo root**
   ```bash
   cp -r deployment/streamlit/.streamlit .streamlit
   git add .streamlit
   git commit -m "Add Streamlit theme config"
   git push
   ```

2. **Create `requirements_dashboard.txt`** (lighter deps for Streamlit Cloud)
   ```
   streamlit>=1.33.0
   plotly>=5.20.0
   requests>=2.31.0
   pandas>=2.2.0
   numpy>=1.26.0
   ```

3. **Deploy on Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **"New app"**
   - Fill in:

   | Field | Value |
   |-------|-------|
   | **Repository** | `YOUR_USERNAME/self-healing-ml` |
   | **Branch** | `main` |
   | **Main file path** | `dashboard/app.py` |

4. **Add Secret (API URL)**
   - In Streamlit Cloud → **Settings → Secrets**
   - Add:
   ```toml
   API_URL = "https://self-healing-ml-api.onrender.com"
   ```

5. **Deploy** — takes ~2 minutes

6. **Access your dashboard**
   - URL: `https://YOUR_USERNAME-self-healing-ml-dashboard.streamlit.app`

---

## Part 3 — Local Development

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/self-healing-ml.git
cd self-healing-ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train initial model
python scripts/train_initial_model.py

# 3. Start API (terminal 1)
uvicorn app:app --reload --port 8000

# 4. Start dashboard (terminal 2)
streamlit run dashboard/app.py

# 5. Run drift simulation (terminal 3)
python scripts/simulate_drift.py --drift-type gradual --steps 500

# 6. Run end-to-end demo (no API needed)
python scripts/demo.py
```

---

## Part 4 — Docker (Local Full Stack)

```bash
docker compose up --build
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

---

## Example API Calls

```bash
# Health check
curl https://YOUR_API.onrender.com/health

# Make a prediction
curl -X POST https://YOUR_API.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "passenger_count": 2,
    "trip_distance": 3.5,
    "pickup_hour": 8,
    "pickup_dow": 1,
    "pickup_month": 3,
    "pickup_is_weekend": 0,
    "rate_code_id": 1,
    "payment_type": 1,
    "pu_location_zone": 10,
    "do_location_zone": 25,
    "vendor_id": 1
  }'

# Submit ground truth (delayed feedback)
curl -X POST https://YOUR_API.onrender.com/predict/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "YOUR_REQUEST_ID", "actual_duration_min": 14.5}'

# Check drift
curl https://YOUR_API.onrender.com/monitor/drift

# View performance metrics
curl https://YOUR_API.onrender.com/monitor/metrics

# View recent drift history
curl "https://YOUR_API.onrender.com/monitor/history?log_type=drift&limit=10"

# Manually trigger retraining
curl -X POST https://YOUR_API.onrender.com/monitor/retrain
```

---

## Estimated Free Tier Limits

| Service | Free Tier | Notes |
|---------|-----------|-------|
| Render Web Service | 750h/month | Sleeps after 15min idle |
| Render Disk | 1 GB | Persists between deploys |
| Streamlit Community Cloud | Unlimited | Public repos only |
| MLflow (local/Docker) | Unlimited | No hosted free tier |
