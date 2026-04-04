"""Generate all visual assets for the README.

Outputs to assets/:
  architecture.png
  drift_detection.png
  before_after_retraining.png
  dashboard_preview.png
  feature_importance.png
  psi_heatmap.png

Usage:
    python scripts/generate_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

BG = "#0b1120"
SURFACE = "#151f32"
SURFACE2 = "#1c2a3f"
BORDER = "#2d3f5a"
ACCENT = "#4f8ef7"
ACCENT2 = "#a78bfa"
OK = "#22d3a0"
WARN = "#fbbf24"
ERROR = "#f87171"
TEXT = "#e2eaf5"
TEXT_DIM = "#7a93b8"
PURPLE = "#7c3aed"
TEAL = "#06b6d4"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_DIM,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "grid.color": BORDER,
    "grid.alpha": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
})


def styled_box(ax, x, y, w, h, label, sub="", colour=SURFACE2, border=ACCENT, fontsize=9, text_color=TEXT):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=colour,
        edgecolor=border,
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(rect)
    label_y = y + h / 2 + (0.18 if sub else 0)
    ax.text(x + w / 2, label_y, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.24, sub,
                ha="center", va="center", fontsize=6.8,
                color=TEXT_DIM, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, colour=TEXT_DIM, label="", lw=1.4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=colour, lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my + 0.14, label,
                ha="center", va="center",
                fontsize=6.5, color=colour,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, edgecolor="none", alpha=0.8),
                zorder=6)


def draw_architecture() -> None:
    fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(BG)

    ax.text(9, 9.55, "Argus", ha="center", va="center",
            fontsize=22, fontweight="bold", color=TEXT)
    ax.text(9, 9.1, "Production ML Observability  |  Drift Detection  |  Automated Retraining",
            ha="center", va="center", fontsize=10, color=TEXT_DIM)
    ax.axhline(8.8, color=BORDER, linewidth=0.8, alpha=0.5)

    styled_box(ax, 0.6, 7.4, 2.8, 0.95, "Trip Event Stream", "Live NYC taxi data", SURFACE2, ACCENT)
    styled_box(ax, 4.0, 7.4, 2.8, 0.95, "Drift Simulator", "Gradual / Sudden / Seasonal", SURFACE2, ACCENT2)
    styled_box(ax, 7.5, 7.4, 2.8, 0.95, "Reference Dataset", "Training distribution", SURFACE2, OK)
    styled_box(ax, 11.0, 7.4, 2.8, 0.95, "Delayed Labels", "Ground truth (24h lag)", SURFACE2, WARN)
    styled_box(ax, 14.5, 7.4, 2.8, 0.95, "MLflow Tracker", "Metrics & artifacts", SURFACE2, TEAL)

    styled_box(ax, 3.0, 5.6, 12.0, 1.1, "FastAPI Prediction Service",
               "POST /predict    POST /predict/feedback    GET /monitor/*", "#0d1b35", ACCENT, 11)

    styled_box(ax, 0.4, 3.5, 3.5, 1.2, "Drift Detector", "PSI  |  KS-test", SURFACE2, ACCENT2)
    styled_box(ax, 4.8, 3.5, 3.5, 1.2, "Performance Monitor", "Rolling RMSE  |  MAE  |  R2", SURFACE2, OK)
    styled_box(ax, 9.2, 3.5, 3.5, 1.2, "Root-Cause Analyzer", "PSI x Feature Importance", SURFACE2, WARN)
    styled_box(ax, 13.6, 3.5, 3.8, 1.2, "Retraining Trigger", "Dual-gate decision logic", SURFACE2, ERROR)

    styled_box(ax, 2.5, 1.3, 3.5, 1.2, "Challenger Trainer", "GradientBoosting + MLflow", SURFACE2, ACCENT)
    styled_box(ax, 7.0, 1.3, 4.0, 1.2, "Model Evaluator", "Champion vs. challenger", SURFACE2, TEAL)
    styled_box(ax, 12.0, 1.3, 3.8, 1.2, "Model Registry", "Safe champion promotion", SURFACE2, OK)

    for x in [2.0, 5.4, 8.9, 12.4, 15.9]:
        ax.annotate("", xy=(9.0, 6.7), xytext=(x, 7.4),
                    arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.2),
                    zorder=5)

    for x_top, x_bot in [(2.15, 2.15), (6.55, 6.55), (10.95, 10.95)]:
        draw_arrow(ax, x_top, 5.6, x_bot, 4.7, colour=TEXT_DIM)

    draw_arrow(ax, 15.5, 5.6, 15.5, 4.7, colour=ERROR)
    draw_arrow(ax, 2.15, 3.5, 4.25, 2.5, colour=ACCENT, label="trigger")
    draw_arrow(ax, 4.25, 2.5, 7.0, 1.9, colour=ACCENT, label="train")
    draw_arrow(ax, 11.0, 1.9, 12.0, 1.9, colour=TEAL, label="compare")
    draw_arrow(ax, 15.5, 3.5, 15.5, 2.5, colour=ERROR)
    draw_arrow(ax, 15.5, 2.5, 15.9, 1.9, colour=ERROR, label="promote")
    draw_arrow(ax, 13.8, 1.9, 9.0, 5.6, colour=OK, label="serve new")

    plt.tight_layout(pad=0.5)
    out = ASSETS_DIR / "architecture.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


def draw_drift_detection_panel() -> None:
    np.random.seed(42)
    n = 500

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle("Feature Drift Analysis  —  PSI & Kolmogorov-Smirnov Detection",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.98)

    gs_outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    features = [
        ("Trip Distance (miles)", "lognormal", 2.6, "High drift — sudden shift"),
        ("Pickup Hour", "shifted_uniform", 0.35, "Moderate drift — time bias"),
        ("Passenger Count", "discrete", 0.5, "Low drift — minor skew"),
    ]

    for col_i, (label, dist, scale, annotation) in enumerate(features):
        if dist == "lognormal":
            ref = np.random.lognormal(0.85, 0.55, n)
            live = np.random.lognormal(0.85 + scale, 0.55, n)
            ref = np.clip(ref, 0.1, 30)
            live = np.clip(live, 0.1, 30)
        elif dist == "shifted_uniform":
            ref = np.random.uniform(0, 23, n)
            shift = scale * 14
            live = np.random.beta(2, 1.2, n) * 23 * (1 + scale)
            live = np.clip(live, 0, 23)
        else:
            probs_ref = [0.08, 0.28, 0.28, 0.20, 0.11, 0.05]
            probs_live = [0.30, 0.30, 0.18, 0.12, 0.07, 0.03]
            ref = np.random.choice(range(1, 7), n, p=probs_ref).astype(float)
            live = np.random.choice(range(1, 7), n, p=probs_live).astype(float)

        ax_dist = fig.add_subplot(gs_outer[0, col_i])
        ax_dist.set_facecolor(SURFACE)

        if dist == "discrete":
            bins = np.arange(0.5, 7.5, 1.0)
            ax_dist.hist(ref, bins=bins, alpha=0.55, color=OK, density=True, label="Reference", zorder=3)
            ax_dist.hist(live, bins=bins, alpha=0.55, color=ERROR, density=True, label="Live", zorder=3)
        else:
            all_vals = np.concatenate([ref, live])
            lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
            bw = np.linspace(lo, hi, 200)
            ref_kde = gaussian_kde(ref, bw_method=0.25)(bw)
            live_kde = gaussian_kde(live, bw_method=0.25)(bw)
            ax_dist.fill_between(bw, ref_kde, alpha=0.3, color=OK, zorder=2)
            ax_dist.plot(bw, ref_kde, color=OK, linewidth=2, label="Reference", zorder=3)
            ax_dist.fill_between(bw, live_kde, alpha=0.3, color=ERROR, zorder=2)
            ax_dist.plot(bw, live_kde, color=ERROR, linewidth=2, label="Live", zorder=3)

        ax_dist.set_title(label, fontsize=10, fontweight="bold", color=TEXT, pad=6)
        ax_dist.legend(fontsize=8, framealpha=0.15, loc="upper right")
        ax_dist.grid(True, alpha=0.25)
        ax_dist.set_ylabel("Density", fontsize=8)

        q_edges = np.percentile(ref, np.linspace(0, 100, 11))
        q_edges[0] -= 1e-9
        q_edges[-1] += 1e-9
        ref_pct = np.maximum(np.histogram(ref, bins=q_edges)[0] / n, 1e-4)
        live_pct = np.maximum(np.histogram(live, bins=q_edges)[0] / n, 1e-4)
        psi_per_bin = (live_pct - ref_pct) * np.log(live_pct / ref_pct)
        total_psi = float(psi_per_bin.sum())

        ax_psi = fig.add_subplot(gs_outer[1, col_i])
        ax_psi.set_facecolor(SURFACE)
        bar_cols = [ERROR if p > 0 else ACCENT for p in psi_per_bin]
        bars = ax_psi.bar(range(len(psi_per_bin)), psi_per_bin, color=bar_cols, alpha=0.85, width=0.7, zorder=3)
        ax_psi.axhline(0, color=BORDER, linewidth=0.8)
        ax_psi.set_title(f"PSI per Decile  (Total = {total_psi:.3f})", fontsize=9, color=TEXT, pad=5)
        ax_psi.set_xlabel("Decile Bin", fontsize=8)
        ax_psi.set_ylabel("PSI", fontsize=8)
        ax_psi.grid(True, alpha=0.25, axis="y")

        if total_psi >= 0.2:
            status_label, status_col = "DRIFT", ERROR
        elif total_psi >= 0.1:
            status_label, status_col = "MODERATE", WARN
        else:
            status_label, status_col = "STABLE", OK

        ax_psi.text(0.97, 0.93, status_label, transform=ax_psi.transAxes,
                    ha="right", va="top", fontsize=10, fontweight="bold",
                    color=status_col,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=status_col, alpha=0.9))
        ax_psi.text(0.03, 0.93, annotation, transform=ax_psi.transAxes,
                    ha="left", va="top", fontsize=7.5, color=TEXT_DIM, style="italic")

    ax_summary = fig.add_subplot(gs_outer[2, :])
    ax_summary.set_facecolor(SURFACE)

    feature_names = ["Trip Distance", "Pickup Hour", "Day of Week", "Passenger Count",
                     "Pickup Zone", "Dropoff Zone", "Rate Code", "Payment Type",
                     "Vendor", "Month", "Is Weekend"]
    psi_scores = [0.42, 0.28, 0.14, 0.09, 0.07, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01]
    ks_pvals = [0.001, 0.008, 0.043, 0.15, 0.22, 0.31, 0.45, 0.60, 0.72, 0.81, 0.93]

    x = np.arange(len(feature_names))
    width = 0.38

    norm_psi = [p / max(psi_scores) for p in psi_scores]
    cols_psi = [ERROR if p >= 0.2 else (WARN if p >= 0.1 else OK) for p in psi_scores]
    cols_ks = [ERROR if p < 0.05 else (WARN if p < 0.1 else OK) for p in ks_pvals]

    b1 = ax_summary.bar(x - width / 2, psi_scores, width, color=cols_psi, alpha=0.85, label="PSI Score", zorder=3)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=8.5)
    ax_summary.axhline(0.2, color=ERROR, linestyle="--", linewidth=1.2, alpha=0.7, label="PSI drift threshold (0.20)")
    ax_summary.axhline(0.1, color=WARN, linestyle=":", linewidth=1.0, alpha=0.7, label="PSI moderate threshold (0.10)")
    ax_summary.set_ylabel("PSI Score", fontsize=9, color=TEXT_DIM)
    ax_summary.set_title("All-Feature Drift Summary  —  Current Monitoring Window",
                         fontsize=11, fontweight="bold", color=TEXT, pad=8)
    ax_summary.legend(fontsize=8.5, framealpha=0.15, loc="upper right")
    ax_summary.grid(True, alpha=0.2, axis="y")

    for bar, score in zip(b1, psi_scores):
        if score >= 0.1:
            ax_summary.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{score:.2f}", ha="center", va="bottom", fontsize=7.5,
                            color=TEXT, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS_DIR / "drift_detection.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


def draw_performance_recovery() -> None:
    np.random.seed(7)
    n = 600
    t = np.arange(n)

    baseline = 3.42
    drift_start = 180
    retrain_at = 370
    recovery_done = 430

    rmse = np.zeros(n)
    noise = np.random.normal(0, 0.08, n)

    rmse[:drift_start] = baseline + noise[:drift_start]
    for i in range(drift_start, retrain_at):
        p = (i - drift_start) / (retrain_at - drift_start)
        rmse[i] = baseline + p ** 0.75 * 3.1 + noise[i] * 0.18
    for i in range(retrain_at, n):
        rec = min(1.0, (i - retrain_at) / 55)
        target = baseline * 0.93
        rmse[i] = (baseline * (1.35 + 0.5 * (1 - rec)) * (1 - rec) + target * rec) + noise[i]

    rmse = gaussian_filter1d(rmse, sigma=2.5)

    mae = rmse * 0.78 + np.random.normal(0, 0.04, n)
    mae = gaussian_filter1d(mae, sigma=2.5)

    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    fig.suptitle("Model Performance Recovery  —  Before and After Retraining",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    ax_main = fig.add_subplot(gs[:, :2])
    ax_main.set_facecolor(SURFACE)

    ax_main.fill_between(t, rmse, alpha=0.08, color=ACCENT, zorder=1)
    ax_main.plot(t, rmse, color=ACCENT, linewidth=2.2, label="Rolling RMSE", zorder=3)
    ax_main.plot(t, mae, color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.7, label="Rolling MAE", zorder=3)

    ax_main.axhline(baseline, color=OK, linestyle="--", linewidth=1.5, alpha=0.85,
                    label=f"Baseline  ({baseline:.2f} min)", zorder=4)
    ax_main.axhline(baseline * 1.15, color=WARN, linestyle=":", linewidth=1.2, alpha=0.8,
                    label="Alert threshold  (+15%)", zorder=4)

    ax_main.axvspan(drift_start, retrain_at, alpha=0.08, color=ERROR, zorder=2, label="Drift window")
    ax_main.axvspan(retrain_at, n, alpha=0.07, color=OK, zorder=2, label="Post-retrain recovery")

    ax_main.axvline(drift_start, color=ERROR, linewidth=2.0, linestyle="--", zorder=5, alpha=0.9)
    ax_main.axvline(retrain_at, color=PURPLE, linewidth=2.5, zorder=5, alpha=0.95)

    ax_main.annotate(
        "Drift Onset\nPSI = 0.31",
        xy=(drift_start + 8, rmse[drift_start + 8]),
        xytext=(drift_start + 45, baseline + 2.2),
        fontsize=9, color=ERROR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ERROR, lw=1.5,
                        connectionstyle="arc3,rad=-0.25"),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE2, edgecolor=ERROR, alpha=0.9),
    )
    ax_main.annotate(
        "Retraining Triggered\nchallenger promoted",
        xy=(retrain_at + 4, rmse[retrain_at + 4]),
        xytext=(retrain_at + 45, baseline + 1.6),
        fontsize=9, color=PURPLE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE2, edgecolor=PURPLE, alpha=0.9),
    )
    ax_main.annotate(
        f"Peak RMSE\n{rmse[retrain_at-10]:.2f} min (+{(rmse[retrain_at-10]/baseline-1)*100:.0f}%)",
        xy=(retrain_at - 10, rmse[retrain_at - 10]),
        xytext=(retrain_at - 90, baseline + 2.8),
        fontsize=8.5, color=WARN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE2, edgecolor=WARN, alpha=0.9),
    )

    ax_main.set_xlabel("Monitoring Window (requests)", fontsize=10, color=TEXT_DIM)
    ax_main.set_ylabel("RMSE  (minutes)", fontsize=10, color=TEXT_DIM)
    ax_main.set_title("Full Production Timeline", fontsize=12, fontweight="bold", color=TEXT, pad=8)
    ax_main.legend(fontsize=8.5, framealpha=0.15, loc="upper left", ncol=2)
    ax_main.grid(True, alpha=0.22)
    ax_main.set_xlim(0, n - 1)

    segments = [
        ("Pre-drift\n(Stable)", rmse[:drift_start], OK),
        ("Drift Window\n(Degraded)", rmse[drift_start:retrain_at], ERROR),
        ("Post-retrain\n(Recovered)", rmse[retrain_at:], ACCENT),
    ]

    ax_box = fig.add_subplot(gs[0, 2])
    ax_box.set_facecolor(SURFACE)
    data_for_box = [s[1] for s in segments]
    labels_for_box = [s[0] for s in segments]
    bp = ax_box.boxplot(
        data_for_box,
        tick_labels=labels_for_box,
        patch_artist=True,
        medianprops=dict(color=TEXT, linewidth=2.2),
        flierprops=dict(marker=".", markerfacecolor=TEXT_DIM, markersize=4, alpha=0.5),
        whiskerprops=dict(color=TEXT_DIM, linewidth=1.2),
        capprops=dict(color=TEXT_DIM, linewidth=1.2),
    )
    box_cols = [OK, ERROR, ACCENT]
    for patch, col in zip(bp["boxes"], box_cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)
        patch.set_edgecolor(col)

    for i, (phase, vals, col) in enumerate(segments):
        med = np.median(vals)
        change = (med / np.median(segments[0][1]) - 1) * 100
        if i > 0:
            sign = "+" if change >= 0 else ""
            ax_box.text(i + 1, np.percentile(vals, 75) + 0.05,
                        f"{sign}{change:.0f}%",
                        ha="center", fontsize=9, fontweight="bold",
                        color=ERROR if change > 0 else OK)

    ax_box.set_ylabel("RMSE (minutes)", fontsize=9)
    ax_box.set_title("Distribution Shift", fontsize=11, fontweight="bold", color=TEXT)
    ax_box.grid(True, alpha=0.22, axis="y")

    ax_delta = fig.add_subplot(gs[1, 2])
    ax_delta.set_facecolor(SURFACE)

    window = 30
    rolling_pct = []
    for i in range(window, n):
        win_mean = rmse[i - window:i].mean()
        rolling_pct.append((win_mean / baseline - 1) * 100)
    t_rolling = np.arange(window, n)

    ax_delta.axhline(0, color=OK, linewidth=1.2, linestyle="--", alpha=0.7, label="Baseline")
    ax_delta.axhline(15, color=WARN, linewidth=1.0, linestyle=":", alpha=0.7, label="Alert (+15%)")
    ax_delta.axvspan(drift_start, retrain_at, alpha=0.08, color=ERROR, zorder=2)
    ax_delta.axvspan(retrain_at, n, alpha=0.07, color=OK, zorder=2)

    colours_line = [ERROR if v > 15 else (WARN if v > 0 else OK) for v in rolling_pct]
    for i in range(len(t_rolling) - 1):
        ax_delta.plot(t_rolling[i:i + 2], rolling_pct[i:i + 2],
                      color=colours_line[i], linewidth=1.8, alpha=0.85)

    ax_delta.set_xlabel("Monitoring Window", fontsize=9)
    ax_delta.set_ylabel("RMSE vs. Baseline (%)", fontsize=9)
    ax_delta.set_title("Degradation Percentage", fontsize=11, fontweight="bold", color=TEXT)
    ax_delta.legend(fontsize=8, framealpha=0.15)
    ax_delta.grid(True, alpha=0.22)
    ax_delta.set_xlim(0, n - 1)

    out = ASSETS_DIR / "before_after_retraining.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


def draw_dashboard_preview() -> None:
    np.random.seed(13)
    n = 350
    t = np.arange(n)

    baseline = 3.42
    rmse_vals = np.concatenate([
        baseline + np.random.normal(0, 0.09, 160),
        np.linspace(baseline, baseline + 3.4, 100) + np.random.normal(0, 0.18, 100),
        np.linspace(baseline + 3.4, baseline * 0.93, 60) + np.random.normal(0, 0.15, 60),
        baseline * 0.93 + np.random.normal(0, 0.08, 30),
    ])
    rmse_vals = gaussian_filter1d(rmse_vals, sigma=3)

    feature_names = ["Trip Distance", "Pickup Hour", "Passenger Count", "Day of Week", "Zone"]
    psi_vals = [0.42, 0.28, 0.09, 0.05, 0.03]
    rca_feats = ["Trip Distance", "Pickup Hour", "Passenger Count"]
    rca_scores = [0.521, 0.312, 0.088]

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    fig.text(0.5, 0.975, "Argus  —  Production ML Monitoring Dashboard",
             ha="center", va="center", fontsize=17, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.958, "Drift Detection  |  Root-Cause Analysis  |  Automated Retraining",
             ha="center", va="center", fontsize=10, color=TEXT_DIM)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.32,
                           top=0.93, bottom=0.07, left=0.05, right=0.97)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(SURFACE)
    ax1.fill_between(t, rmse_vals, alpha=0.12, color=ACCENT)
    ax1.plot(t, rmse_vals, color=ACCENT, linewidth=2.2, label="Rolling RMSE", zorder=3)
    ax1.axhline(baseline, color=OK, linestyle="--", linewidth=1.4, label=f"Baseline ({baseline:.2f})", alpha=0.9)
    ax1.axhline(baseline * 1.15, color=WARN, linestyle=":", linewidth=1.2, label="Alert +15%", alpha=0.85)
    ax1.axvspan(160, 260, alpha=0.09, color=ERROR, label="Drift window")
    ax1.axvline(260, color=PURPLE, linewidth=2.2, linestyle="--", alpha=0.9, label="Retrain")
    ax1.set_title("Prediction Error Over Time (Rolling RMSE)", fontsize=11, fontweight="bold", color=TEXT)
    ax1.legend(fontsize=8, framealpha=0.15, ncol=3)
    ax1.grid(True, alpha=0.22)
    ax1.set_ylabel("RMSE (min)", color=TEXT_DIM, fontsize=9)
    ax1.set_xlabel("Request count", color=TEXT_DIM, fontsize=9)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(SURFACE)
    cols_psi = [ERROR if p >= 0.2 else (WARN if p >= 0.1 else OK) for p in psi_vals]
    bars = ax2.barh(feature_names[::-1], psi_vals[::-1], color=cols_psi[::-1], alpha=0.85)
    ax2.axvline(0.2, color=ERROR, linestyle="--", linewidth=1.2, alpha=0.7, label="Drift threshold")
    ax2.axvline(0.1, color=WARN, linestyle=":", linewidth=1.0, alpha=0.7, label="Moderate")
    for bar, val in zip(bars, psi_vals[::-1]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=8.5, color=TEXT, fontweight="bold")
    ax2.set_title("PSI by Feature", fontsize=11, fontweight="bold", color=TEXT)
    ax2.set_xlabel("PSI Score", color=TEXT_DIM, fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.15, loc="lower right")
    ax2.grid(True, alpha=0.22, axis="x")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor(SURFACE)
    rca_cols = [ERROR, WARN, OK]
    bars3 = ax3.barh(rca_feats[::-1], rca_scores[::-1], color=rca_cols[::-1], alpha=0.85)
    for bar, val in zip(bars3, rca_scores[::-1]):
        ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=8.5, color=TEXT, fontweight="bold")
    ax3.set_title("Root-Cause RCA Score", fontsize=11, fontweight="bold", color=TEXT)
    ax3.set_xlabel("PSI x Feature Importance", color=TEXT_DIM, fontsize=9)
    ax3.grid(True, alpha=0.22, axis="x")
    ax3.text(0.97, 0.05, "Primary:\nTrip Distance", transform=ax3.transAxes,
             ha="right", va="bottom", fontsize=8, color=ERROR,
             bbox=dict(boxstyle="round", facecolor=SURFACE2, edgecolor=ERROR, alpha=0.9))

    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor(SURFACE)

    steps = 10
    features_heat = ["Trip Dist.", "Pickup Hr.", "Passenger", "Day/Week", "Zone"]
    psi_history = np.array([
        [0.03, 0.04, 0.05, 0.06, 0.09, 0.15, 0.28, 0.42, 0.38, 0.08],
        [0.02, 0.03, 0.04, 0.06, 0.10, 0.19, 0.28, 0.28, 0.22, 0.06],
        [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.08, 0.09, 0.07, 0.03],
        [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.04, 0.02],
        [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02],
    ])

    cmap = LinearSegmentedColormap.from_list(
        "drift_cmap", [(0, OK), (0.1 / 0.5, WARN), (1.0, ERROR)], N=256
    )
    im = ax4.imshow(psi_history, aspect="auto", cmap=cmap, vmin=0, vmax=0.5, interpolation="nearest")
    ax4.set_xticks(range(steps))
    ax4.set_xticklabels([f"W{i+1}" for i in range(steps)], fontsize=8.5)
    ax4.set_yticks(range(len(features_heat)))
    ax4.set_yticklabels(features_heat, fontsize=9)
    ax4.set_title("PSI Heatmap — Feature Drift Over Time", fontsize=11, fontweight="bold", color=TEXT)
    ax4.axvline(5.5, color=ERROR, linewidth=2, linestyle="--", alpha=0.7)
    ax4.axvline(8.5, color=OK, linewidth=2, linestyle="--", alpha=0.7)
    cbar = plt.colorbar(im, ax=ax4, orientation="horizontal", pad=0.12, fraction=0.04)
    cbar.set_label("PSI Score", fontsize=8.5, color=TEXT_DIM)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_DIM)

    for i in range(psi_history.shape[0]):
        for j in range(psi_history.shape[1]):
            val = psi_history[i, j]
            if val >= 0.1:
                ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7.5, fontweight="bold", color=TEXT)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(SURFACE)
    retrain_x = [0, 260, 420, 680, 900]
    retrain_y = [0, 0, 0, 0, 0]
    colors_r = [TEXT_DIM, PURPLE, TEXT_DIM, PURPLE, TEXT_DIM]
    labels_r = ["Blocked\nno drift", "Triggered\npromoted", "Blocked\ncooldown", "Triggered\npromoted", "Monitoring"]
    for xi, yi, ci, li in zip(retrain_x, retrain_y, colors_r, labels_r):
        ax5.scatter(xi, yi, color=ci, s=110, zorder=5, linewidths=2, edgecolors=ci)
        ax5.axvline(xi, color=ci, linewidth=1.5, alpha=0.5)
        ax5.text(xi + 12, 0.05, li, rotation=40, fontsize=7.5, color=ci, ha="left", va="bottom")
    ax5.set_xlim(-50, 1000)
    ax5.set_ylim(-0.4, 0.5)
    ax5.set_title("Retraining Events Timeline", fontsize=11, fontweight="bold", color=TEXT)
    ax5.set_yticks([])
    ax5.set_xlabel("Monitoring Step", color=TEXT_DIM, fontsize=9)
    ax5.grid(True, alpha=0.22, axis="x")

    ax6 = fig.add_subplot(gs[1, 3])
    ax6.set_facecolor(SURFACE)
    ax6.axis("off")

    system_stats = [
        ("API Status", "Online", OK),
        ("Model Version", "v4  (retrained)", ACCENT),
        ("Drift Status", "Detected", ERROR),
        ("Champion RMSE", "3.18 min", OK),
        ("Baseline RMSE", "3.42 min", TEXT_DIM),
        ("Rolling RMSE", "4.71 min", WARN),
        ("Labeled Samples", "2,341", TEXT),
        ("Pending Labels", "58", TEXT_DIM),
        ("Retrain Events", "2", ACCENT2),
        ("Cooldown Active", "No", OK),
    ]
    ax6.set_title("System Health", fontsize=11, fontweight="bold", color=TEXT, pad=8)
    for i, (key, val, col) in enumerate(system_stats):
        y = 0.94 - i * 0.093
        ax6.text(0.03, y, key, fontsize=8.8, color=TEXT_DIM, va="center", transform=ax6.transAxes)
        ax6.text(0.97, y, val, fontsize=8.8, color=col, va="center", ha="right",
                 fontweight="bold", transform=ax6.transAxes)
        if i < len(system_stats) - 1:
            ax6.plot([0.02, 0.98], [y - 0.04, y - 0.04],
                     color=BORDER, linewidth=0.6, alpha=0.5,
                     transform=ax6.transAxes)

    out = ASSETS_DIR / "dashboard_preview.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


def draw_feature_importance() -> None:
    np.random.seed(99)

    features = [
        "Trip Distance", "Pickup Hour", "Pickup Zone",
        "Dropoff Zone", "Day of Week", "Rate Code",
        "Passenger Count", "Month", "Is Weekend",
        "Payment Type", "Vendor",
    ]

    champ_imp = np.array([0.38, 0.19, 0.12, 0.10, 0.07, 0.05,
                          0.04, 0.02, 0.01, 0.01, 0.01])
    noise = np.random.normal(0, 0.012, len(features))
    chall_imp = np.clip(champ_imp + noise + np.array([0.04, -0.02, 0.01, 0.0, 0.0, 0.0,
                                                       -0.01, 0.01, 0.0, 0.0, 0.0]), 0.005, 1.0)
    chall_imp = chall_imp / chall_imp.sum()

    idx = np.argsort(champ_imp)
    features_sorted = [features[i] for i in idx]
    champ_sorted = champ_imp[idx]
    chall_sorted = chall_imp[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Feature Importance  —  Champion vs. Challenger",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.01)

    y = np.arange(len(features))
    hw = 0.35
    ax1.set_facecolor(SURFACE)
    b1 = ax1.barh(y + hw / 2, champ_sorted, hw, color=OK, alpha=0.85, label="Champion (v3)")
    b2 = ax1.barh(y - hw / 2, chall_sorted, hw, color=ACCENT, alpha=0.85, label="Challenger (v4)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(features_sorted, fontsize=9.5)
    ax1.set_xlabel("Feature Importance", fontsize=10)
    ax1.set_title("Side-by-Side Comparison", fontsize=12, fontweight="bold", color=TEXT, pad=7)
    ax1.legend(fontsize=9.5, framealpha=0.15)
    ax1.grid(True, alpha=0.22, axis="x")
    for bar, val in zip(b1, champ_sorted):
        if val >= 0.03:
            ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=7.5, color=OK)
    for bar, val in zip(b2, chall_sorted):
        if val >= 0.03:
            ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=7.5, color=ACCENT)

    ax2.set_facecolor(SURFACE)
    delta = chall_sorted - champ_sorted
    delta_cols = [OK if d > 0 else ERROR for d in delta]
    bars_d = ax2.barh(y, delta * 100, color=delta_cols, alpha=0.85)
    ax2.axvline(0, color=TEXT_DIM, linewidth=1.2)
    ax2.set_yticks(y)
    ax2.set_yticklabels(features_sorted, fontsize=9.5)
    ax2.set_xlabel("Change in Importance (percentage points)", fontsize=10)
    ax2.set_title("Challenger vs. Champion Delta", fontsize=12, fontweight="bold", color=TEXT, pad=7)
    ax2.grid(True, alpha=0.22, axis="x")
    for bar, val in zip(bars_d, delta * 100):
        offset = 0.05 if val >= 0 else -0.05
        ax2.text(val + offset, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.1f}pp", va="center", fontsize=7.5,
                 color=OK if val > 0 else ERROR, ha="left" if val >= 0 else "right")

    plt.tight_layout()
    out = ASSETS_DIR / "feature_importance.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


def draw_psi_heatmap() -> None:
    np.random.seed(21)
    features = [
        "Trip Distance", "Pickup Hour", "Pickup Zone",
        "Dropoff Zone", "Day of Week", "Rate Code",
        "Passenger Count", "Month", "Is Weekend", "Payment Type",
    ]
    n_windows = 16

    base = np.array([0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    psi_mat = np.zeros((len(features), n_windows))

    drift_peak = 11
    for w in range(n_windows):
        if w < 6:
            factor = 1.0
        elif w < drift_peak:
            factor = 1.0 + (w - 5) * 3.5
        elif w == drift_peak:
            factor = 22.0
        elif w < 14:
            factor = max(1.0, 22.0 - (w - drift_peak) * 6)
        else:
            factor = 1.0

        row_noise = np.abs(np.random.normal(0, 0.008, len(features)))
        feature_sensitivity = np.array([1.0, 0.75, 0.35, 0.28, 0.22, 0.15,
                                         0.12, 0.08, 0.06, 0.05])
        psi_mat[:, w] = np.clip(base * factor * feature_sensitivity + row_noise, 0.005, 0.8)

    cmap = LinearSegmentedColormap.from_list(
        "psi_heat", [
            (0.0, "#0b1120"),
            (0.05 / 0.8, "#1a3a5c"),
            (0.1 / 0.8, WARN),
            (0.2 / 0.8, ERROR),
            (1.0, "#7f0000"),
        ], N=512
    )

    fig, ax = plt.subplots(figsize=(18, 7), facecolor=BG)
    fig.suptitle("PSI Heatmap  —  Feature Drift Intensity Over Time",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.01)

    im = ax.imshow(psi_mat, aspect="auto", cmap=cmap, vmin=0, vmax=0.5,
                   interpolation="bilinear")

    ax.set_facecolor(SURFACE)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xticks(range(n_windows))
    ax.set_xticklabels([f"W{i+1}" for i in range(n_windows)], fontsize=9.5)
    ax.set_xlabel("Monitoring Window", fontsize=11, color=TEXT_DIM)
    ax.set_title("Each cell = PSI score for that feature in that monitoring window",
                 fontsize=9.5, color=TEXT_DIM, pad=6)

    ax.axvline(5.5, color=ERROR, linewidth=2.2, linestyle="--", alpha=0.8)
    ax.axvline(13.5, color=OK, linewidth=2.2, linestyle="--", alpha=0.8)
    ax.text(5.6, -0.65, "Drift onset", fontsize=9, color=ERROR, fontweight="bold", va="top")
    ax.text(13.6, -0.65, "Post-retrain", fontsize=9, color=OK, fontweight="bold", va="top")

    for i in range(psi_mat.shape[0]):
        for j in range(psi_mat.shape[1]):
            val = psi_mat[i, j]
            if val >= 0.1:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, fontweight="bold",
                        color=TEXT if val > 0.3 else BG)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, fraction=0.025)
    cbar.set_label("PSI Score", fontsize=10, color=TEXT_DIM)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_DIM)
    cbar.ax.axhline(0.1 / 0.5, color=WARN, linewidth=1.5, linestyle="--", alpha=0.9)
    cbar.ax.axhline(0.2 / 0.5, color=ERROR, linewidth=1.5, linestyle="--", alpha=0.9)
    cbar.ax.text(1.6, 0.1 / 0.5, "Moderate", fontsize=8, color=WARN, va="center")
    cbar.ax.text(1.6, 0.2 / 0.5, "Drift", fontsize=8, color=ERROR, va="center")

    plt.tight_layout()
    out = ASSETS_DIR / "psi_heatmap.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating visual assets for Argus ...")
    draw_architecture()
    draw_drift_detection_panel()
    draw_performance_recovery()
    draw_dashboard_preview()
    draw_feature_importance()
    draw_psi_heatmap()
    print("Done. Assets saved to:", ASSETS_DIR)
