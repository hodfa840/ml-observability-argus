"""Argus — production ML observability dashboard.

Pages:
  Overview          - rolling error metrics and system status
  Drift Analysis    - PSI, KS-test, and heatmap per feature
  Feature Insights  - importance comparison and drift ranking
  Retraining Log    - history of retraining decisions
  Live Demo         - interactive API playground

Run with:
    python -m streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as _components

st.set_page_config(
    page_title="Argus",
    page_icon="\U0001f695",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b1120;
    }
    .block-container { padding-top: 1.5rem; }

    /* ── Typography ────────────────────────────────────── */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f8ef7, #a78bfa, #22d3a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .sub-title {
        color: #7a93b8;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        letter-spacing: 0.03em;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2eaf5;
        border-left: 3px solid #4f8ef7;
        padding-left: 0.65rem;
        margin: 1.2rem 0 0.6rem 0;
        letter-spacing: 0.01em;
    }

    /* ── Metric cards ───────────────────────────────────── */
    div[data-testid="metric-container"] {
        background: #151f32;
        border: 1px solid #2d3f5a;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        transition: border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #4f8ef7;
    }
    div[data-testid="metric-container"] label {
        color: #7a93b8 !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e2eaf5 !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
    }

    /* ── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0d1828;
        border-right: 1px solid #1c2a3f;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #7a93b8;
        font-size: 0.85rem;
    }

    /* ── Status badges ──────────────────────────────────── */
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 12px;
        background: #151f32;
        border-radius: 7px;
        margin-bottom: 5px;
        border-left: 3px solid transparent;
    }
    .status-label { color: #7a93b8; font-size: 0.82rem; }
    .status-value { color: #e2eaf5; font-size: 0.82rem; font-weight: 700; }

    /* ── Alert banners ───────────────────────────────────── */
    .drift-alert {
        background: linear-gradient(135deg, #1e0d3b, #120d2b);
        border: 1px solid #7c3aed;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin: 0.4rem 0 0.8rem 0;
    }
    .drift-alert-title { font-size: 1rem; font-weight: 700; color: #c4b5fd; }
    .drift-alert-body  { color: #ddd8fe; font-size: 0.87rem; margin-top: 5px; }

    /* ── Tables ──────────────────────────────────────────── */
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* ── Decision cards ──────────────────────────────────── */
    .decision-card {
        border-radius: 9px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .decision-title { font-size: 0.95rem; font-weight: 700; color: #e2eaf5; }
    .decision-time  { color: #7a93b8; font-size: 0.78rem; }
    .decision-body  { color: #c4b5fd; font-size: 0.83rem; margin-top: 5px; }
    .decision-block { color: #7a93b8; font-size: 0.78rem; margin-top: 3px; }

    /* ── Responsive grid helpers ────────────────────────── */
    .ag-grid-3 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
    }
    .ag-grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
    }
    .ag-grid-3-sm {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin-top: 12px;
    }

    /* ── Mobile breakpoint ──────────────────────────────── */
    @media (max-width: 768px) {
        .main-title { font-size: 1.7rem; }
        .sub-title  { font-size: 0.85rem; }
        .ag-grid-3, .ag-grid-2, .ag-grid-3-sm { grid-template-columns: 1fr; }
        .block-container {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
        }
        .decision-card { padding: 10px 12px; }
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_PATHS = {
    "performance": PROJECT_ROOT / "data" / "logs" / "performance.jsonl",
    "drift":       PROJECT_ROOT / "data" / "logs" / "drift_reports.jsonl",
    "retrain":     PROJECT_ROOT / "data" / "logs" / "retraining.jsonl",
    "feedback":    PROJECT_ROOT / "data" / "logs" / "feedback.jsonl",
    "predictions": PROJECT_ROOT / "data" / "logs" / "predictions.jsonl",
    "importances": PROJECT_ROOT / "data" / "logs" / "feature_importances.json",
}

ACCENT   = "#4f8ef7"
ACCENT2  = "#a78bfa"
OK       = "#22d3a0"
WARN     = "#fbbf24"
ERR      = "#f87171"
PURPLE   = "#7c3aed"
TEAL     = "#06b6d4"
SURFACE  = "#151f32"
BORDER   = "#2d3f5a"

FEATURE_LABELS = {
    "trip_distance":    "Trip Distance",
    "passenger_count":  "Passenger Count",
    "pickup_hour":      "Pickup Hour",
    "pickup_dow":       "Day of Week",
    "pickup_month":     "Month",
    "pickup_is_weekend": "Is Weekend",
    "rate_code_id":     "Rate Code",
    "payment_type":     "Payment Type",
    "pu_location_zone": "Pickup Zone",
    "do_location_zone": "Dropoff Zone",
    "vendor_id":        "Vendor",
}

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=12, b=8),
    font=dict(color="#e2eaf5", size=11),
)


@st.cache_data(ttl=10)
def load_jsonl(path: Path, limit: int = 2000) -> pd.DataFrame:
    """Read the last `limit` lines of a JSONL log file into a DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return pd.DataFrame(records) if records else pd.DataFrame()


@st.cache_data(ttl=15)
def load_importances(path: Path) -> pd.DataFrame:
    """Load feature importance JSON into a DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def api_health() -> dict:
    """Fetch the /health endpoint; return empty dict if unreachable."""
    try:
        http_resp = requests.get(f"{API_URL}/health", timeout=3)
        return http_resp.json()
    except requests.RequestException:
        return {}


@st.cache_data(ttl=10)
def api_metrics() -> dict:
    """Fetch the /monitor/metrics endpoint; return empty dict if unreachable."""
    try:
        http_resp = requests.get(f"{API_URL}/monitor/metrics", timeout=3)
        return http_resp.json()
    except requests.RequestException:
        return {}


def _pct_color(pct: float | None) -> str:
    if pct is None:
        return ACCENT
    return ERR if pct > 15 else (WARN if pct > 5 else OK)


def _plotly_layout(**kwargs) -> dict:
    base = dict(PLOTLY_BASE)
    base.update(kwargs)
    return base


with st.sidebar:
    st.markdown(
        '<div style="font-size:1.5rem;font-weight:800;'
        'background:linear-gradient(90deg,#4f8ef7,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'margin-bottom:2px">\U0001f695 Argus</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#7a93b8;font-size:0.78rem;margin-bottom:12px">'
        'NYC Taxi  \u00b7  ML Observability Platform</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#c8d8ee;font-size:0.79rem;line-height:1.6;'
        'margin-bottom:10px;padding:10px 12px;background:#111d30;'
        'border-radius:7px;border:1px solid #1c2a3f">'
        'End-to-end MLOps platform: serves predictions via REST API, '
        'monitors live data for distribution shift, and triggers '
        'automated retraining when quality degrades.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:4px">'
        + "".join(
            f'<span style="background:#1c2a3f;color:#7a93b8;font-size:0.68rem;'
            f'font-weight:700;padding:2px 7px;border-radius:4px;'
            f'letter-spacing:0.04em">{t}</span>'
            for t in ["FastAPI", "scikit-learn", "MLflow", "Streamlit", "Plotly", "Docker"]
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    health = api_health()
    api_online = health.get("status") == "ok"
    if api_online:
        st.success("API Online")
    else:
        st.warning("API Offline — showing logged data")

    if api_online:
        st.markdown(
            f'<div style="background:#1c2a3f;border-radius:8px;'
            f'padding:10px 12px;margin-top:8px;font-size:0.82rem">'
            f'<div style="color:#7a93b8;margin-bottom:6px;font-weight:700;'
            f'letter-spacing:0.05em;font-size:0.72rem">SYSTEM INFO</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
            f'<span style="color:#7a93b8">Model</span>'
            f'<span style="color:#e2eaf5;font-weight:600">'
            f'{health.get("model_version", "N/A")}</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
            f'<span style="color:#7a93b8">Uptime</span>'
            f'<span style="color:#e2eaf5;font-weight:600">'
            f'{health.get("uptime_seconds", 0):.0f}s</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
            f'<span style="color:#7a93b8">Pending</span>'
            f'<span style="color:#fbbf24;font-weight:600">'
            f'{health.get("pending_feedback_count", 0)}</span></div>'
            f'<div style="display:flex;justify-content:space-between">'
            f'<span style="color:#7a93b8">Matched</span>'
            f'<span style="color:#22d3a0;font-weight:600">'
            f'{health.get("matched_feedback_count", 0)}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="color:#7a93b8;font-size:0.72rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:8px">NAVIGATION</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "",
        ["Overview", "Drift Analysis", "Feature Insights", "Retraining Log", "Live Demo"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (15s)", value=True)
    if st.button("Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(
        f'<div style="color:#7a93b8;font-size:0.75rem">API: '
        f'<code style="color:#4f8ef7">{API_URL}</code></div>',
        unsafe_allow_html=True,
    )

# ── Overview ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown('<p class="main-title">\U0001f695 Argus</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Production ML observability  '
        '&nbsp;·&nbsp;  Drift detection  '
        '&nbsp;·&nbsp;  Automated retraining</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="background:#0d1f38;border:1px solid #2d3f5a;border-radius:10px;'
        'padding:16px 20px;margin:12px 0 18px 0" class="ag-grid-3">'

        '<div>'
        '<div style="color:#4f8ef7;font-size:0.72rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:5px">THE PROBLEM</div>'
        '<div style="color:#c8d8ee;font-size:0.83rem;line-height:1.6">'
        'ML models degrade in production when real-world data distributions shift '
        'away from training data. Manual monitoring does not scale.'
        '</div>'
        '</div>'

        '<div>'
        '<div style="color:#a78bfa;font-size:0.72rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:5px">WHAT ARGUS DOES</div>'
        '<div style="color:#c8d8ee;font-size:0.83rem;line-height:1.6">'
        'Serves predictions via FastAPI, continuously measures feature drift '
        'and accuracy, then automatically retrains and promotes a new model '
        'when degradation is confirmed.'
        '</div>'
        '</div>'

        '<div>'
        '<div style="color:#22d3a0;font-size:0.72rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:5px">DOMAIN</div>'
        '<div style="color:#c8d8ee;font-size:0.83rem;line-height:1.6">'
        'NYC taxi trip duration prediction. GradientBoostingRegressor trained '
        'on TLC data. Simulated temporal drift shifts pickup patterns over time.'
        '</div>'
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    metrics = api_metrics()
    rmse    = metrics.get("rmse")
    baseline = metrics.get("baseline_rmse")
    mae     = metrics.get("mae")
    r2      = metrics.get("r2")
    n_samp  = metrics.get("n_samples", 0)

    pct_change = (rmse - baseline) / baseline * 100 if rmse and baseline else None
    delta_str = f"{pct_change:+.1f}%" if pct_change is not None else None

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rolling RMSE",   f"{rmse:.3f}" if rmse else "N/A", delta=delta_str)
    col2.metric("Baseline RMSE",  f"{baseline:.3f}" if baseline else "N/A")
    col3.metric("MAE",            f"{mae:.3f}" if mae else "N/A")
    col4.metric("R\u00b2",        f"{r2:.3f}" if r2 else "N/A")
    col5.metric("Labeled Samples", n_samp)

    st.markdown("---")

    left, right = st.columns([3, 1])

    with left:
        st.markdown('<p class="section-header">Prediction Error Over Time</p>',
                    unsafe_allow_html=True)
        perf_df = load_jsonl(LOG_PATHS["performance"])

        if not perf_df.empty and "rmse" in perf_df.columns:
            perf_df["idx"] = range(len(perf_df))
            bsl = baseline or perf_df["rmse"].min()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=perf_df["idx"], y=perf_df["rmse"],
                mode="lines", name="Rolling RMSE",
                line=dict(color=ACCENT, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(79,142,247,0.07)",
            ))

            if "mae" in perf_df.columns:
                fig.add_trace(go.Scatter(
                    x=perf_df["idx"], y=perf_df["mae"],
                    mode="lines", name="Rolling MAE",
                    line=dict(color=ACCENT2, width=1.6, dash="dot"),
                    opacity=0.75,
                ))

            fig.add_hline(y=bsl, line_dash="dash", line_color=OK, line_width=1.4,
                          annotation_text="Baseline",
                          annotation_position="bottom right",
                          annotation_font_color=OK)
            fig.add_hline(y=bsl * 1.15, line_dash="dot", line_color=WARN, line_width=1.2,
                          annotation_text="Alert +15%",
                          annotation_position="top right",
                          annotation_font_color=WARN)

            retrain_df = load_jsonl(LOG_PATHS["retrain"])
            if not retrain_df.empty and "should_retrain" in retrain_df.columns:
                triggered_idx = retrain_df[retrain_df["should_retrain"]].index.tolist()
                for ti in triggered_idx:
                    mapped = min(ti, len(perf_df) - 1)
                    fig.add_vrect(
                        x0=max(0, mapped - 1), x1=min(len(perf_df) - 1, mapped + 1),
                        fillcolor="rgba(124,58,237,0.35)", line_width=0,
                        annotation_text="Retrain", annotation_position="top left",
                        annotation_font_color=ACCENT2,
                    )

            fig.update_layout(
                **_plotly_layout(
                    height=320,
                    xaxis=dict(title="Monitoring Window", gridcolor=BORDER, showgrid=True),
                    yaxis=dict(title="RMSE (minutes)", gridcolor=BORDER, showgrid=True),
                    legend=dict(orientation="h", y=1.08, x=0),
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            if "r2" in perf_df.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=perf_df["idx"], y=perf_df["r2"],
                    mode="lines", name="R\u00b2",
                    line=dict(color=TEAL, width=1.8),
                    fill="tozeroy",
                    fillcolor="rgba(6,182,212,0.07)",
                ))
                fig2.add_hline(y=0.8, line_dash="dot", line_color=WARN, line_width=1.0,
                               annotation_text="Quality floor (0.80)",
                               annotation_position="bottom right",
                               annotation_font_color=WARN)
                r2_min = float(perf_df["r2"].min())
                r2_floor = min(r2_min - 0.05, -0.1) if r2_min < 0 else -0.05
                fig2.update_layout(
                    **_plotly_layout(
                        height=160,
                        xaxis=dict(title="", gridcolor=BORDER),
                        yaxis=dict(title="R\u00b2", gridcolor=BORDER, range=[r2_floor, 1.05]),
                        showlegend=False,
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No performance data yet — run the simulation script to generate data.")

    with right:
        st.markdown('<p class="section-header">System Status</p>', unsafe_allow_html=True)

        drift_df   = load_jsonl(LOG_PATHS["drift"])
        retrain_df = load_jsonl(LOG_PATHS["retrain"])

        last_drift = drift_df.iloc[-1].to_dict() if not drift_df.empty else {}
        rmse_ok = not (
            rmse and baseline and rmse > baseline * 1.1
        )
        feat_drift = bool(last_drift.get("drift_detected"))
        model_ok   = bool(health.get("model_loaded", True))

        status_items = [
            ("Feature Drift",  "DETECTED" if feat_drift else "STABLE",
             ERR if feat_drift else OK),
            ("Performance",    "DEGRADED" if not rmse_ok else "HEALTHY",
             ERR if not rmse_ok else OK),
            ("Model",          "LOADED" if model_ok else "NOT LOADED",
             OK if model_ok else ERR),
            ("API",            "ONLINE" if api_online else "OFFLINE",
             OK if api_online else ERR),
        ]

        for label, value, colour in status_items:
            st.markdown(
                f'<div class="status-row" style="border-left-color:{colour}">'
                f'<span class="status-label">{label}</span>'
                f'<span class="status-value" style="color:{colour}">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        n_triggered = int(retrain_df["should_retrain"].sum()) \
            if not retrain_df.empty and "should_retrain" in retrain_df.columns \
            else len(retrain_df)

        st.markdown(
            f'<div style="margin-top:12px;padding:14px 16px;background:{SURFACE};'
            f'border-radius:9px;border:1px solid {BORDER}">'
            f'<div style="color:#7a93b8;font-size:0.72rem;font-weight:700;'
            f'letter-spacing:0.07em;text-transform:uppercase">Retraining Events</div>'
            f'<div style="color:#e2eaf5;font-size:2.1rem;font-weight:800;'
            f'margin-top:4px">{n_triggered}</div>'
            f'<div style="color:#7a93b8;font-size:0.78rem">triggered / '
            f'{len(retrain_df)} evaluations</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not perf_df.empty and "rmse" in perf_df.columns and len(perf_df) >= 10:
            recent = perf_df["rmse"].tail(20).values
            spark_fig = go.Figure()
            spark_fig.add_trace(go.Scatter(
                y=recent, mode="lines",
                line=dict(color=ACCENT, width=2),
                fill="tozeroy",
                fillcolor="rgba(79,142,247,0.1)",
            ))
            spark_fig.update_layout(
                **_plotly_layout(
                    height=90,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    showlegend=False,
                )
            )
            st.markdown(
                '<div style="color:#7a93b8;font-size:0.75rem;'
                'margin-top:12px;margin-bottom:2px">Recent RMSE trend</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(spark_fig, use_container_width=True)


# ── Drift Analysis ────────────────────────────────────────────────────────────
elif page == "Drift Analysis":
    st.markdown('<p class="main-title">Drift Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Feature-level distribution monitoring '
        '— PSI and Kolmogorov-Smirnov</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:#0d1f38;border:1px solid #2d3f5a;border-radius:9px;'
        'padding:13px 18px;margin:10px 0 16px 0" class="ag-grid-2">'

        '<div>'
        '<div style="color:#fbbf24;font-size:0.72rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:4px">PSI — POPULATION STABILITY INDEX</div>'
        '<div style="color:#c8d8ee;font-size:0.81rem;line-height:1.55">'
        'Compares the distribution of each feature between training data and live '
        'production data. '
        '<span style="color:#22d3a0">PSI &lt; 0.10</span> = stable, '
        '<span style="color:#fbbf24">0.10&ndash;0.20</span> = moderate shift, '
        '<span style="color:#f87171">PSI &gt; 0.20</span> = significant drift.'
        '</div>'
        '</div>'

        '<div>'
        '<div style="color:#06b6d4;font-size:0.72rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:4px">KS TEST — KOLMOGOROV-SMIRNOV</div>'
        '<div style="color:#c8d8ee;font-size:0.81rem;line-height:1.55">'
        'A non-parametric statistical test that checks whether two samples come '
        'from the same distribution. A p-value below 0.05 rejects the null hypothesis '
        '— the feature distribution has changed significantly.'
        '</div>'
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    drift_df = load_jsonl(LOG_PATHS["drift"])

    if drift_df.empty:
        st.info("No drift reports yet — run the simulation to generate data.")
        st.stop()

    feat_df = drift_df[drift_df["report_type"] == "feature"].copy() \
        if "report_type" in drift_df.columns else drift_df.copy()

    if feat_df.empty:
        st.info("No feature drift reports yet.")
        st.stop()

    last = feat_df.iloc[-1]
    feat_results = last.get("feature_results", {})
    if isinstance(feat_results, str):
        import ast
        feat_results = ast.literal_eval(feat_results)

    if last.get("drift_detected"):
        drifted = last.get("drifted_features", [])
        st.markdown(
            f'<div class="drift-alert">'
            f'<div class="drift-alert-title">Drift Detected</div>'
            f'<div class="drift-alert-body">'
            f'Features with significant drift: '
            f'<strong>{", ".join(FEATURE_LABELS.get(f, f) for f in drifted)}</strong>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("No significant drift detected in the latest report.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    if feat_results:
        psi_rows = [
            {
                "Feature": FEATURE_LABELS.get(k, k),
                "PSI": v.get("psi", 0),
                "Drifted": v.get("drifted", False),
            }
            for k, v in feat_results.items()
        ]
        psi_df = pd.DataFrame(psi_rows).sort_values("PSI", ascending=True)

        with col1:
            st.markdown('<p class="section-header">PSI by Feature</p>',
                        unsafe_allow_html=True)
            bar_cols = [ERR if d else OK for d in psi_df["Drifted"]]
            fig_psi = go.Figure(go.Bar(
                x=psi_df["PSI"], y=psi_df["Feature"],
                orientation="h",
                marker=dict(color=bar_cols, opacity=0.85),
                text=psi_df["PSI"].round(3).astype(str),
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig_psi.add_vline(x=0.1, line_dash="dot", line_color=WARN, line_width=1.2,
                              annotation_text="Moderate (0.10)",
                              annotation_font_color=WARN, annotation_font_size=9)
            fig_psi.add_vline(x=0.2, line_dash="dash", line_color=ERR, line_width=1.4,
                              annotation_text="Drift (0.20)",
                              annotation_font_color=ERR, annotation_font_size=9)
            fig_psi.update_layout(
                **_plotly_layout(
                    height=360,
                    xaxis=dict(title="PSI Score", gridcolor=BORDER),
                    yaxis=dict(gridcolor=BORDER),
                )
            )
            st.plotly_chart(fig_psi, use_container_width=True)

        with col2:
            st.markdown('<p class="section-header">KS Test p-values</p>',
                        unsafe_allow_html=True)
            ks_rows = [
                {
                    "Feature": FEATURE_LABELS.get(k, k),
                    "KS p-value": v.get("ks_pvalue", 1.0),
                    "KS Stat": v.get("ks_stat", 0),
                }
                for k, v in feat_results.items()
            ]
            ks_df = pd.DataFrame(ks_rows).sort_values("KS p-value", ascending=True)
            cols_ks = [ERR if p < 0.05 else (WARN if p < 0.1 else OK)
                       for p in ks_df["KS p-value"]]

            fig_ks = go.Figure(go.Bar(
                x=ks_df["KS p-value"], y=ks_df["Feature"],
                orientation="h",
                marker=dict(color=cols_ks, opacity=0.85),
                text=ks_df["KS p-value"].round(4).astype(str),
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig_ks.add_vline(x=0.05, line_dash="dash", line_color=ERR, line_width=1.4,
                             annotation_text="Significance (0.05)",
                             annotation_font_color=ERR, annotation_font_size=9)
            fig_ks.add_vline(x=0.1, line_dash="dot", line_color=WARN, line_width=1.2)
            fig_ks.update_layout(
                **_plotly_layout(
                    height=360,
                    xaxis=dict(title="p-value", gridcolor=BORDER),
                    yaxis=dict(gridcolor=BORDER),
                )
            )
            st.plotly_chart(fig_ks, use_container_width=True)

    st.markdown('<p class="section-header">PSI Heatmap Over Time (Top Features)</p>',
                unsafe_allow_html=True)
    if len(feat_df) >= 1 and feat_df.iloc[0].get("feature_results"):
        all_feature_psi: dict[str, list[float]] = {}
        windows: list[int] = []
        for i, row in feat_df.tail(16).iterrows():
            fr = row.get("feature_results", {})
            if isinstance(fr, str):
                import ast
                fr = ast.literal_eval(fr)
            if not fr:
                continue
            windows.append(len(windows))
            for fname, fvals in fr.items():
                all_feature_psi.setdefault(fname, []).append(fvals.get("psi", 0))

        if all_feature_psi and windows:
            feat_order = sorted(all_feature_psi, key=lambda k: max(all_feature_psi[k]),
                                reverse=True)[:8]
            heat_data = np.array([all_feature_psi[f] for f in feat_order])
            feat_labels = [FEATURE_LABELS.get(f, f) for f in feat_order]

            fig_heat = go.Figure(go.Heatmap(
                z=heat_data,
                x=[f"W{i+1}" for i in windows],
                y=feat_labels,
                colorscale=[
                    [0.0,  "#0b1120"],
                    [0.15, "#1c2a3f"],
                    [0.4,  WARN],
                    [0.7,  ERR],
                    [1.0,  "#7f0000"],
                ],
                zmin=0, zmax=0.5,
                colorbar=dict(
                    title="PSI",
                    title_side="right",
                    tickfont=dict(size=10),
                ),
                text=np.round(heat_data, 3).astype(str),
                texttemplate="%{text}",
                textfont=dict(size=9),
            ))
            fig_heat.update_layout(
                **_plotly_layout(
                    height=280,
                    xaxis=dict(side="bottom"),
                    yaxis=dict(gridcolor=BORDER),
                )
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No feature drift data yet to show the heatmap.")

    st.markdown('<p class="section-header">Drift Events Timeline</p>',
                unsafe_allow_html=True)
    if "drift_detected" in feat_df.columns:
        values = feat_df["drift_detected"].astype(int).tolist()
        colours_t = [ERR if v else OK for v in values]
        fig_t = go.Figure(go.Scatter(
            y=values, mode="markers+lines",
            marker=dict(color=colours_t, size=9, line=dict(width=1.5, color="#0b1120")),
            line=dict(color=BORDER, width=1),
        ))
        fig_t.update_layout(
            **_plotly_layout(
                height=160,
                yaxis=dict(
                    tickvals=[0, 1],
                    ticktext=["No Drift", "Drift"],
                    gridcolor=BORDER,
                ),
                xaxis=dict(title="Check index", gridcolor=BORDER),
            )
        )
        st.plotly_chart(fig_t, use_container_width=True)


# ── Feature Insights ──────────────────────────────────────────────────────────
elif page == "Feature Insights":
    st.markdown('<p class="main-title">Feature Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Drift ranking, root-cause scores, '
        'and model feature importance</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:#0d1f38;border:1px solid #2d3f5a;border-left:3px solid #a78bfa;'
        'border-radius:9px;padding:13px 18px;margin:10px 0 16px 0">'
        '<div style="color:#a78bfa;font-size:0.72rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:6px">ROOT-CAUSE ANALYSIS</div>'
        '<div style="color:#c8d8ee;font-size:0.82rem;line-height:1.6">'
        'Not all drifted features are equally dangerous. A feature that has shifted '
        'but contributes little to the model\'s predictions is low risk. '
        'The <strong style="color:#e2eaf5">drift radar</strong> and '
        '<strong style="color:#e2eaf5">ranking table</strong> cross-reference '
        'PSI drift scores with model feature importance, so the highest-risk '
        'root causes are immediately visible.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    drift_df = load_jsonl(LOG_PATHS["drift"])
    imp_df   = load_importances(LOG_PATHS["importances"])

    if not drift_df.empty:
        feat_df = drift_df[drift_df["report_type"] == "feature"] \
            if "report_type" in drift_df.columns else drift_df
        if not feat_df.empty:
            last = feat_df.iloc[-1]
            feat_results = last.get("feature_results", {})
            if isinstance(feat_results, str):
                import ast
                feat_results = ast.literal_eval(feat_results)

            if feat_results:
                rows = [
                    {
                        "Feature":       FEATURE_LABELS.get(k, k),
                        "raw_name":      k,
                        "PSI":           round(v.get("psi", 0), 4),
                        "KS Statistic":  round(v.get("ks_stat", 0), 4),
                        "KS p-value":    round(v.get("ks_pvalue", 1), 4),
                        "Drifted":       "Yes" if v.get("drifted") else "No",
                    }
                    for k, v in feat_results.items()
                ]
                rows.sort(key=lambda r: r["PSI"], reverse=True)

                left_col, right_col = st.columns([1, 1])

                with left_col:
                    st.markdown('<p class="section-header">Drift Ranking Table</p>',
                                unsafe_allow_html=True)
                    rank_df = pd.DataFrame(rows).drop(columns=["raw_name"])
                    st.dataframe(rank_df, use_container_width=True, hide_index=True)

                with right_col:
                    st.markdown('<p class="section-header">Drift Radar</p>',
                                unsafe_allow_html=True)
                    feats  = [r["Feature"] for r in rows]
                    psis   = [r["PSI"] for r in rows]
                    fig_rad = go.Figure(go.Scatterpolar(
                        r=psis + [psis[0]],
                        theta=feats + [feats[0]],
                        fill="toself",
                        fillcolor="rgba(79,142,247,0.15)",
                        line=dict(color=ACCENT, width=2.2),
                        name="PSI",
                    ))
                    fig_rad.update_layout(
                        **_plotly_layout(
                            height=380,
                            polar=dict(
                                bgcolor="rgba(0,0,0,0)",
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, max(psis + [0.3])],
                                    gridcolor=BORDER,
                                    tickfont=dict(size=9),
                                ),
                                angularaxis=dict(gridcolor=BORDER),
                            ),
                        )
                    )
                    st.plotly_chart(fig_rad, use_container_width=True)

    if not imp_df.empty and "feature" in imp_df.columns and "importance" in imp_df.columns:
        st.markdown("---")
        st.markdown('<p class="section-header">Model Feature Importance</p>',
                    unsafe_allow_html=True)
        imp_sorted = imp_df.sort_values("importance", ascending=True).tail(11)
        feat_labels = [FEATURE_LABELS.get(f, f) for f in imp_sorted["feature"]]
        imp_vals    = imp_sorted["importance"].tolist()

        cmap_cols = [ACCENT if v < 0.15 else (WARN if v < 0.25 else ERR)
                     for v in imp_vals]
        fig_imp = go.Figure(go.Bar(
            x=imp_vals, y=feat_labels,
            orientation="h",
            marker=dict(color=cmap_cols, opacity=0.88),
            text=[f"{v:.3f}" for v in imp_vals],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_imp.update_layout(
            **_plotly_layout(
                height=350,
                xaxis=dict(title="Feature Importance", gridcolor=BORDER),
                yaxis=dict(gridcolor=BORDER),
            )
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance data not available — train the initial model first.")


# ── Retraining Log ────────────────────────────────────────────────────────────
elif page == "Retraining Log":
    st.markdown('<p class="main-title">Retraining Log</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Automated retraining decisions — '
        'drift detection, performance gating, and champion-challenger promotion</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Design explanation ────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#0f1e38;border:1px solid #2d3f5a;border-left:3px solid #4f8ef7;'
        'border-radius:8px;padding:14px 18px;margin-bottom:18px">'
        '<div style="color:#4f8ef7;font-size:0.78rem;font-weight:700;'
        'letter-spacing:0.07em;margin-bottom:8px">RETRAINING POLICY</div>'
        '<div style="color:#c8d8ee;font-size:0.84rem;line-height:1.7">'
        'Retraining requires <strong style="color:#e2eaf5">two independent signals</strong> '
        'to fire simultaneously — this is intentional conservative design to avoid '
        'unnecessary churn in production.'
        '</div>'
        '<div class="ag-grid-3-sm">'
        '<div style="background:#151f32;border-radius:6px;padding:10px 12px">'
        '<div style="color:#a78bfa;font-size:0.75rem;font-weight:700;margin-bottom:4px">'
        'GATE 1 &mdash; FEATURE DRIFT</div>'
        '<div style="color:#c8d8ee;font-size:0.79rem;line-height:1.5">'
        'PSI &gt; 0.20 or KS p &lt; 0.05 on any feature. '
        'Detects distribution shift in the incoming data pipeline.</div>'
        '</div>'
        '<div style="background:#151f32;border-radius:6px;padding:10px 12px">'
        '<div style="color:#f87171;font-size:0.75rem;font-weight:700;margin-bottom:4px">'
        'GATE 2 &mdash; PERFORMANCE DEGRADATION</div>'
        '<div style="color:#c8d8ee;font-size:0.79rem;line-height:1.5">'
        'Rolling RMSE exceeds baseline by more than 15%. '
        'Confirms the model is actually harmed, not just seeing new data.</div>'
        '</div>'
        '<div style="background:#151f32;border-radius:6px;padding:10px 12px">'
        '<div style="color:#22d3a0;font-size:0.75rem;font-weight:700;margin-bottom:4px">'
        'GATE 3 &mdash; SAMPLE BUDGET</div>'
        '<div style="color:#c8d8ee;font-size:0.79rem;line-height:1.5">'
        'At least 1,000 new labeled samples required. '
        'Prevents retraining on insufficient data that would destabilize the model.</div>'
        '</div>'
        '</div>'
        '<div style="color:#7a93b8;font-size:0.76rem;margin-top:10px;font-style:italic">'
        'Current status: feature drift is active but performance has not degraded '
        '(RMSE is below baseline), so retraining is correctly suppressed. '
        'The model is handling the shifted distribution without accuracy loss.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    retrain_df = load_jsonl(LOG_PATHS["retrain"])

    if retrain_df.empty:
        st.info("No retraining evaluations yet.")
        st.stop()

    has_flag  = "should_retrain" in retrain_df.columns
    triggered = int(retrain_df["should_retrain"].sum()) if has_flag else len(retrain_df)
    blocked   = int((~retrain_df["should_retrain"]).sum()) if has_flag else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluations", len(retrain_df))
    c2.metric("Triggered",         triggered)
    c3.metric("Blocked",           blocked)
    c4.metric("Trigger Rate",      f"{triggered / len(retrain_df) * 100:.0f}%"
              if len(retrain_df) else "0%")

    if triggered > 0 and blocked > 0:
        st.markdown('<p class="section-header">Decision Breakdown</p>',
                    unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Triggered", "Blocked"],
            values=[triggered, blocked],
            marker=dict(colors=[PURPLE, BORDER], line=dict(color="#0b1120", width=2)),
            hole=0.55,
            textfont=dict(size=12),
        ))
        fig_pie.update_layout(
            **_plotly_layout(
                height=220,
                showlegend=True,
                legend=dict(orientation="h", x=0.25),
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Decision Log  (most recent first)</p>',
                unsafe_allow_html=True)

    for _, row in retrain_df.tail(25).iloc[::-1].iterrows():
        should          = bool(row.get("should_retrain", False))
        feat_drift      = bool(row.get("feature_drift", False))
        perf_drift      = bool(row.get("performance_drift", False))
        n_samples       = int(row.get("samples_since_last_retrain", 0))
        bg_col          = "#1a0d3b" if should else "#0f1929"
        brd_col         = PURPLE if should else BORDER
        label           = "Retrain Triggered" if should else "Retrain Blocked"
        label_color     = "#c4b5fd" if should else "#f87171"
        reasons         = row.get("reasons", [])
        blocking        = row.get("blocking_reasons", [])
        ts              = row.get("timestamp", "")

        gate1_col = OK if feat_drift else BORDER
        gate1_txt = "DRIFT DETECTED" if feat_drift else "NO DRIFT"
        gate2_col = ERR if perf_drift else OK
        gate2_txt = "DEGRADED" if perf_drift else "STABLE"
        gate3_col = OK if n_samples >= 1000 else WARN
        gate3_txt = f"{n_samples:,} samples" if n_samples else "—"

        reasons_html = (
            "<div style='color:#c8d8ee;font-size:0.83rem;margin-top:8px;line-height:1.6'>" +
            "<br>".join(str(r) for r in reasons) +
            "</div>"
        ) if reasons else ""

        blocking_html = (
            "<div style='color:#7a93b8;font-size:0.77rem;margin-top:6px;padding-top:6px;"
            "border-top:1px solid #2d3f5a'>"
            "<span style='color:#f87171;font-weight:700'>Blocked: </span>" +
            " &nbsp;&bull;&nbsp; ".join(str(b) for b in blocking) +
            "</div>"
        ) if blocking else ""

        gate_badges = (
            f'<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">'
            f'<span style="background:#0b1120;border:1px solid {gate1_col};color:{gate1_col};'
            f'border-radius:4px;padding:2px 8px;font-size:0.72rem;font-weight:700">'
            f'FEATURE DRIFT: {gate1_txt}</span>'
            f'<span style="background:#0b1120;border:1px solid {gate2_col};color:{gate2_col};'
            f'border-radius:4px;padding:2px 8px;font-size:0.72rem;font-weight:700">'
            f'PERFORMANCE: {gate2_txt}</span>'
            f'<span style="background:#0b1120;border:1px solid {gate3_col};color:{gate3_col};'
            f'border-radius:4px;padding:2px 8px;font-size:0.72rem;font-weight:700">'
            f'SAMPLES: {gate3_txt}</span>'
            f'</div>'
        )

        st.markdown(
            f'<div class="decision-card" '
            f'style="background:{bg_col};border:1px solid {brd_col};margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-size:0.95rem;font-weight:700;color:{label_color}">{label}</span>'
            f'<span class="decision-time">{ts}</span>'
            f'</div>'
            f'{reasons_html}'
            f'{gate_badges}'
            f'{blocking_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Live Demo ─────────────────────────────────────────────────────────────────
elif page == "Live Demo":
    st.markdown('<p class="main-title">Live Demo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Interact with the prediction API in real time</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:#0d1f38;border:1px solid #2d3f5a;border-radius:9px;'
        'padding:13px 18px;margin:10px 0 16px 0" class="ag-grid-3">'

        '<div style="display:flex;gap:10px;align-items:flex-start">'
        '<div style="background:#4f8ef7;color:#0b1120;font-size:1rem;font-weight:800;'
        'border-radius:50%;width:24px;height:24px;display:flex;align-items:center;'
        'justify-content:center;flex-shrink:0;margin-top:2px">1</div>'
        '<div><div style="color:#4f8ef7;font-size:0.72rem;font-weight:700;'
        'margin-bottom:3px">PREDICT</div>'
        '<div style="color:#c8d8ee;font-size:0.80rem;line-height:1.5">'
        'Set trip parameters and call <code style="color:#a78bfa">/predict</code>. '
        'The FastAPI service runs the GradientBoosting model and returns a '
        'trip duration estimate with a unique request ID.'
        '</div></div>'
        '</div>'

        '<div style="display:flex;gap:10px;align-items:flex-start">'
        '<div style="background:#22d3a0;color:#0b1120;font-size:1rem;font-weight:800;'
        'border-radius:50%;width:24px;height:24px;display:flex;align-items:center;'
        'justify-content:center;flex-shrink:0;margin-top:2px">2</div>'
        '<div><div style="color:#22d3a0;font-size:0.72rem;font-weight:700;'
        'margin-bottom:3px">SUBMIT GROUND TRUTH</div>'
        '<div style="color:#c8d8ee;font-size:0.80rem;line-height:1.5">'
        'Paste the request ID and enter the actual trip duration. '
        'This simulates the delayed feedback loop — real ground truth '
        'arrives minutes or hours after prediction.'
        '</div></div>'
        '</div>'

        '<div style="display:flex;gap:10px;align-items:flex-start">'
        '<div style="background:#a78bfa;color:#0b1120;font-size:1rem;font-weight:800;'
        'border-radius:50%;width:24px;height:24px;display:flex;align-items:center;'
        'justify-content:center;flex-shrink:0;margin-top:2px">3</div>'
        '<div><div style="color:#a78bfa;font-size:0.72rem;font-weight:700;'
        'margin-bottom:3px">MONITOR</div>'
        '<div style="color:#c8d8ee;font-size:0.80rem;line-height:1.5">'
        'Each submitted feedback updates the rolling accuracy window. '
        'Run a drift check to see if the live data distribution '
        'has shifted from the training baseline.'
        '</div></div>'
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<p class="section-header">Make a Prediction</p>',
                    unsafe_allow_html=True)

        passenger_count = st.slider("Passengers", 1, 6, 2)
        trip_distance   = st.slider("Trip Distance (miles)", 0.5, 20.0, 3.5, step=0.5)
        pickup_hour     = st.slider("Pickup Hour (0-23)", 0, 23, 8)
        pickup_dow      = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
        )
        pickup_month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ][x - 1],
        )
        is_weekend = 1 if pickup_dow >= 5 else 0

        st.markdown(
            f'<div style="color:#7a93b8;font-size:0.8rem;margin:6px 0">'
            f'Is Weekend: <strong style="color:#e2eaf5">'
            f'{"Yes" if is_weekend else "No"}</strong></div>',
            unsafe_allow_html=True,
        )

        if st.button("Predict Duration", use_container_width=True, type="primary"):
            payload = {
                "passenger_count":  passenger_count,
                "trip_distance":    trip_distance,
                "pickup_hour":      pickup_hour,
                "pickup_dow":       pickup_dow,
                "pickup_month":     pickup_month,
                "pickup_is_weekend": is_weekend,
                "rate_code_id":     1,
                "payment_type":     1,
                "pu_location_zone": 10,
                "do_location_zone": 25,
                "vendor_id":        1,
            }
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                result = resp.json()
                if resp.status_code == 200:
                    dur = result.get("predicted_duration_min", "?")
                    st.success(f"Predicted duration: **{dur} minutes**")
                    st.json(result)
                    st.session_state["last_request_id"] = result.get("request_id", "")
                else:
                    st.error(f"API error {resp.status_code}: {result}")
            except requests.RequestException as exc:
                st.error(f"Could not reach API: {exc}")

        st.markdown("---")
        st.markdown('<p class="section-header">Check Drift Status</p>',
                    unsafe_allow_html=True)
        if st.button("Run Drift Check", use_container_width=True):
            try:
                resp = requests.get(f"{API_URL}/monitor/drift", timeout=10)
                result = resp.json()
                if result.get("drift_detected"):
                    st.error("Drift Detected — check Drift Analysis page for details.")
                else:
                    st.success("No significant drift in current window.")
                st.json(result)
            except requests.RequestException as exc:
                st.error(f"Could not reach API: {exc}")

    with right:
        st.markdown('<p class="section-header">Submit Ground Truth</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#7a93b8;font-size:0.85rem;margin-bottom:10px">'
            'Simulate delayed feedback — ground truth arriving after the trip ends.</div>',
            unsafe_allow_html=True,
        )

        req_id = st.text_input(
            "Request ID",
            value=st.session_state.get("last_request_id", ""),
            placeholder="paste request_id from prediction response",
        )
        actual = st.number_input(
            "Actual Duration (min)", min_value=1.0, max_value=120.0,
            value=15.0, step=0.5,
        )

        if st.button("Submit Feedback", use_container_width=True):
            if not req_id:
                st.warning("Paste a request_id first.")
            else:
                try:
                    resp = requests.post(
                        f"{API_URL}/predict/feedback",
                        json={"request_id": req_id, "actual_duration_min": actual},
                        timeout=5,
                    )
                    result = resp.json()
                    if result.get("matched"):
                        st.success(result.get("message", "Feedback accepted."))
                    else:
                        st.warning(result.get("message", "Request ID not found."))
                except requests.RequestException as exc:
                    st.error(f"Could not reach API: {exc}")

        st.markdown("---")
        st.markdown('<p class="section-header">Example API Response</p>',
                    unsafe_allow_html=True)
        st.code(json.dumps({
            "drift_detected":    True,
            "root_cause":        ["trip_distance", "pickup_hour"],
            "performance_drop":  "18.3%",
            "action":            "retraining_triggered",
            "drifted_features":  ["trip_distance", "pickup_hour"],
            "rca_details": [
                {
                    "feature":    "trip_distance",
                    "psi":        0.312,
                    "importance": 0.421,
                    "rca_score":  0.444,
                },
                {
                    "feature":    "pickup_hour",
                    "psi":        0.241,
                    "importance": 0.187,
                    "rca_score":  0.286,
                },
            ],
        }, indent=2), language="json")

        st.markdown("---")
        st.markdown('<p class="section-header">Quick Curl Commands</p>',
                    unsafe_allow_html=True)
        st.code(f"# Health check\ncurl {API_URL}/health\n\n"
                f"# Performance metrics\ncurl {API_URL}/monitor/metrics\n\n"
                f"# Drift report\ncurl {API_URL}/monitor/drift\n\n"
                f"# Manual retrain trigger\ncurl -X POST {API_URL}/monitor/retrain",
                language="bash")

# ── Scroll-to-top on page change (runs after all content is rendered) ─────────
# A monotonically increasing counter combined with the page name guarantees a
# unique HTML string on every navigation — even when revisiting a page —
# so Streamlit always creates a fresh iframe and re-executes the scroll script.
if st.session_state.get("_scroll_page") != page:
    _nav_count = st.session_state.get("_scroll_count", 0) + 1
    st.session_state["_scroll_page"] = page
    st.session_state["_scroll_count"] = _nav_count
    _components.html(
        f"""<script>/* {page}-{_nav_count} */
        (function() {{
            var SELECTORS = ['[data-testid="stMain"]', '.main',
                             '[data-testid="stAppViewContainer"]'];
            function findEl(win) {{
                for (var s = 0; s < SELECTORS.length; s++) {{
                    var el = win.document.querySelector(SELECTORS[s]);
                    if (el) return el;
                }}
                return null;
            }}
            function scrollToTop() {{
                // walk up available parent frames (handles HF double-iframe)
                var frames = [window.parent, window.parent.parent, window.top];
                for (var i = 0; i < frames.length; i++) {{
                    try {{
                        var el = findEl(frames[i]);
                        if (el) {{ el.scrollTop = 0; }}
                        frames[i].scrollTo(0, 0);
                    }} catch(e) {{ /* cross-origin frame, skip */ }}
                }}
            }}
            scrollToTop();
            var c = 0;
            var iv = setInterval(function() {{
                scrollToTop();
                if (++c >= 20) clearInterval(iv);
            }}, 80);
        }})();
        </script>""",
        height=1,
        scrolling=False,
    )

# ── Auto-refresh (must run AFTER all page content is rendered) ────────────────
if auto_refresh:
    import time
    time.sleep(15)
    st.rerun()
