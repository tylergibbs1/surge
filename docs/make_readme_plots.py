"""Generate the three hero charts for the README.

Outputs (all PNG @ 2×):
    docs/plots/hero_forecast.png       — 7-BA weekly forecast fan chart
    docs/plots/leaderboard.png         — horizontal bar chart of test MASE vs CIs
    docs/plots/horizon_curve.png       — MASE vs forecast horizon with naive line

Run with the local venv:
    .venv/bin/python docs/make_readme_plots.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `experiments` importable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch

from chronos import BaseChronosPipeline
from experiments.features import load_multi_ba

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "chronos2_full_v2"
OUT = ROOT / "docs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

DARK_BG = "#0a0a0a"
PANEL_BG = "#111111"
FG = "#eeeeee"
MUTED = "#888888"
ACCENT = "#4FC3F7"
GOOD = "#4CAF50"
BAD = "#f44336"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": FG,
    "axes.titlecolor": FG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#222222",
    "grid.linewidth": 0.5,
    "figure.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
})


# ------------------------------------------------------------------
# 1. HERO FORECAST — 7 BAs, one week, forecast vs actual + 80% band
# ------------------------------------------------------------------
def hero_forecast() -> Path:
    BAS = ["PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP"]
    bas = load_multi_ba(BAS, with_gen=False)

    pipe = BaseChronosPipeline.from_pretrained(
        str(MODEL_PATH),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )

    # Pick a week whose PJM peak is highest in the 2025 test set.
    pjm = bas["PJM"]
    test_slice = pjm.target[pjm.val_end:]
    test_ts = pjm.ts_utc[pjm.val_end:]
    week_hours = 24 * 7
    peak_origin, peak_val = 0, -np.inf
    for i in range(0, len(test_slice) - week_hours, 24):
        w = test_slice[i:i + week_hours].max()
        # Skip windows with suspicious SWPP gaps (min < 10% of typical)
        swpp_w = bas["SWPP"].target[bas["SWPP"].val_end + i:
                                      bas["SWPP"].val_end + i + week_hours]
        if swpp_w.min() < 5_000:  # SWPP typical load > 20 GW
            continue
        if w > peak_val:
            peak_val = w
            peak_origin = i
    start_idx = pjm.val_end + peak_origin
    ts_window = test_ts[peak_origin:peak_origin + week_hours]
    ts_dt = np.array([np.datetime64(t, "s").astype(datetime) for t in ts_window])

    context = 2048
    horizon = 24

    def forecast_week(bd):
        medians, lo, hi, true = [], [], [], []
        for off in range(0, week_hours, horizon):
            o = start_idx + off
            past = {k: v[o - context:o] for k, v in bd.covariates.items()}
            future = {k: bd.covariates[k][o:o + horizon] for k in bd.future_keys}
            task = [{"target": bd.target[o - context:o].astype(np.float32),
                     "past_covariates": past, "future_covariates": future}]
            q, _ = pipe.predict_quantiles(
                task, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
                batch_size=1)
            q = q[0].squeeze(0).float().cpu().numpy()
            lo.extend(q[:, 0]); medians.extend(q[:, 1]); hi.extend(q[:, 2])
            true.extend(bd.target[o:o + horizon])
        return (np.array(true), np.array(medians), np.array(lo), np.array(hi))

    fig, axes = plt.subplots(4, 2, figsize=(13, 12))
    axes = axes.flatten()
    mapes = []
    for i, ba in enumerate(BAS):
        true, med, lo, hi = forecast_week(bas[ba])
        ax = axes[i]
        ax.fill_between(ts_dt, lo / 1000, hi / 1000, color=ACCENT, alpha=0.18,
                        label="80% PI", zorder=1)
        ax.plot(ts_dt, true / 1000, color=FG, lw=2.0, label="Actual", zorder=3)
        ax.plot(ts_dt, med / 1000, color=ACCENT, lw=1.6, ls="--",
                label="Forecast", zorder=2)
        mape = float(np.abs((true - med) / np.where(true > 0, true, 1)).mean() * 100)
        mapes.append(mape)
        ax.set_title(f"{ba}   ·   {mape:.2f}% MAPE on this week",
                     loc="left", fontsize=10.5, pad=4)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
        ax.grid(True, alpha=0.4)
        for s in ax.spines.values():
            s.set_color("#333333")

    # Summary panel
    ax = axes[-1]
    ax.axis("off")
    text = (
        "Surge day-ahead forecast\n"
        f"Week: {ts_dt[0].strftime('%b %d, %Y')} — {ts_dt[-1].strftime('%b %d')}\n\n"
        "Per-BA weekly MAPE:\n"
        + "\n".join(f"  {ba:<5s}  {m:5.2f}%" for ba, m in zip(BAS, mapes))
        + f"\n\n  Overall  {np.mean(mapes):5.2f}%\n\n"
        "Dashed line = median forecast.\n"
        "Shaded band = 80% probability interval.\n"
        "Model: Chronos-2 fine-tuned on\n7 BAs × 7 years of public data."
    )
    ax.text(0.03, 0.97, text, color=FG, fontsize=10.5, va="top",
            family="monospace", transform=ax.transAxes)

    fig.suptitle("Surge — day-ahead forecasts vs. reality",
                 color=FG, fontsize=15, y=0.995, fontweight="bold")
    fig.text(0.5, 0.005,
             f"2025 hold-out week · 7 US balancing authorities · macro MAPE {np.mean(mapes):.2f}%",
             color=MUTED, fontsize=10, ha="center")
    fig.tight_layout(rect=[0, 0.015, 1, 0.98])
    path = OUT / "hero_forecast.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# 2. LEADERBOARD — bar chart of test MASE with 95% CIs
# ------------------------------------------------------------------
def leaderboard() -> Path:
    data = [
        ("persistence",                          2.3061, None, None, "#f44336"),
        ("Prophet (+temp regressor)",            2.0234, 1.978, 2.068, "#f44336"),
        ("seasonal-naive-24  (baseline)",        1.0440, 1.019, 1.071, "#888"),
        ("XGBoost hourly-binned (Roy '25)",      0.9010, 0.879, 0.924, "#ff9800"),
        ("N-BEATS (Pelekis '23)",                0.7144, 0.692, 0.738, "#ff9800"),
        ("Chronos-Bolt-base zero-shot",          0.6876, 0.668, 0.708, "#2196F3"),
        ("Chronos-2 zero-shot  +wx +cal",        0.5672, 0.550, 0.586, "#2196F3"),
        ("Chronos-2 LoRA ft  +wx +cal",          0.5134, 0.499, 0.530, "#4CAF50"),
        ("Chronos-2 full ft  +wx +cal",          0.4921, 0.477, 0.509, "#4CAF50"),
        ("Ensemble of 3 Chronos-2 (SOTA)",       0.4534, 0.440, 0.470, "#4FC3F7"),
    ]
    labels = [r[0] for r in data]
    mase = [r[1] for r in data]
    lows = [r[2] if r[2] is not None else r[1] for r in data]
    his  = [r[3] if r[3] is not None else r[1] for r in data]
    colors = [r[4] for r in data]
    err_low = [m - lo for m, lo in zip(mase, lows)]
    err_hi  = [hi - m for m, hi in zip(mase, his)]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(data))
    ax.barh(y, mase, xerr=[err_low, err_hi], color=colors,
            edgecolor="#222", alpha=0.92, linewidth=0.8,
            error_kw={"elinewidth": 1.2, "ecolor": FG, "capsize": 3})
    ax.axvline(1.0, color=MUTED, lw=1.0, ls="--", alpha=0.6)
    ax.text(1.02, -0.7, "naive baseline", color=MUTED, fontsize=8, va="top")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Test MASE  (lower is better, 95% CI)", color=FG, fontsize=11)
    ax.set_xlim(0, 2.4)
    ax.set_title("Surge vs. every classical & foundation baseline on 2025 hold-out",
                 color=FG, fontsize=13, fontweight="bold", loc="left", pad=12)
    for i, m in enumerate(mase):
        ax.text(m + err_hi[i] + 0.03, i, f"{m:.3f}", color=FG, fontsize=9, va="center")
    ax.grid(True, axis="x", alpha=0.25)
    for s in ax.spines.values():
        s.set_color("#333")
    fig.text(0.01, 0.01,
             "7 US BAs, macro MASE, 367 rolling 24h-ahead windows, "
             "denominator = per-BA train seasonal-naive (m=24).",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = OUT / "leaderboard.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# 3. HORIZON CURVE — MASE vs forecast horizon, with naive break-even
# ------------------------------------------------------------------
def horizon_curve() -> Path:
    # Matched-horizon results from experiments/results.tsv
    matched = {
        1:   0.1712,
        6:   0.3198,
        24:  0.4643,
        72:  0.6233,
        168: 0.7613,
    }
    # Per-step from h=168 run (first 168 values) — in order of step_ahead
    per_step = np.array([
        0.211, 0.271, 0.302, 0.312, 0.320, 0.329, 0.330, 0.335, 0.338, 0.343,
        0.355, 0.392, 0.447, 0.485, 0.524, 0.577, 0.607, 0.655, 0.694, 0.721,
        0.736, 0.748, 0.741, 0.706, 0.710, 0.687, 0.652, 0.603, 0.558, 0.532,
        0.510, 0.498, 0.488, 0.476, 0.481, 0.516, 0.559, 0.603, 0.649, 0.699,
        0.752, 0.815, 0.865, 0.892, 0.908, 0.915, 0.902, 0.879, 0.869, 0.830,
        0.784, 0.730, 0.672, 0.633, 0.595, 0.565, 0.544, 0.532, 0.539, 0.574,
        0.627, 0.667, 0.709, 0.767, 0.824, 0.894, 0.948, 0.983, 0.998, 1.004,
        0.993, 0.958, 0.947, 0.905, 0.852, 0.789, 0.726, 0.685, 0.646, 0.622,
        0.594, 0.577, 0.581, 0.614, 0.657, 0.694, 0.743, 0.799, 0.862, 0.932,
        0.988, 1.020, 1.034, 1.041, 1.029, 1.001, 0.995, 0.949, 0.893, 0.831,
        0.762, 0.718, 0.678, 0.645, 0.613, 0.594, 0.597, 0.632, 0.681, 0.724,
        0.772, 0.836, 0.897, 0.971, 1.025, 1.060, 1.077, 1.092, 1.079, 1.041,
        1.027, 0.980, 0.929, 0.860, 0.789, 0.745, 0.706, 0.679, 0.645, 0.629,
        0.629, 0.662, 0.703, 0.747, 0.802, 0.857, 0.918, 0.991, 1.047, 1.080,
        1.099, 1.110, 1.093, 1.067, 1.059, 1.016, 0.959, 0.900, 0.834, 0.783,
        0.737, 0.700, 0.665, 0.646, 0.645, 0.678, 0.726, 0.775, 0.832, 0.899,
        0.972, 1.043, 1.107, 1.144, 1.162, 1.173, 1.155, 1.125,
    ])
    xs = np.arange(1, len(per_step) + 1)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    # Seasonal-naive reference line at y=1.0
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", alpha=0.8,
               label="seasonal-naive-24 (MASE = 1.0)")
    ax.fill_between(xs, per_step, 1.0, where=per_step < 1.0, alpha=0.15,
                    color=GOOD, label="We beat naive here")
    ax.fill_between(xs, per_step, 1.0, where=per_step >= 1.0, alpha=0.18,
                    color=BAD, label="Naive wins here")
    ax.plot(xs, per_step, color=ACCENT, lw=2.0, label="Chronos-2 per-step MASE")

    # Mark the break-even point (first x where per_step >= 1.0)
    crossings = np.where(per_step >= 1.0)[0]
    if len(crossings):
        x0 = int(crossings[0] + 1)
        ax.axvline(x0, color=FG, lw=0.8, ls=":", alpha=0.5)
        ax.annotate(f"break-even at\n{x0}h ≈ {x0/24:.1f} days",
                    xy=(x0, 1.0), xytext=(x0 + 6, 0.55),
                    color=FG, fontsize=10,
                    arrowprops={"arrowstyle": "->", "color": FG, "lw": 0.8})

    # Horizontal markers for matched-horizon numbers
    for h, m in matched.items():
        ax.plot([h], [m], "o", color=FG, markersize=8, zorder=5)
        ax.annotate(f" h={h} · {m:.2f}", (h, m), color=FG, fontsize=9,
                    xytext=(6, 2), textcoords="offset points")

    ax.set_xlabel("Forecast horizon  (hours ahead)", fontsize=11, color=FG)
    ax.set_ylabel("MASE  (lower = better)", fontsize=11, color=FG)
    ax.set_xlim(0, 170); ax.set_ylim(0, 1.3)
    ax.set_xticks([1, 24, 48, 72, 120, 168])
    ax.set_title("How far ahead can Surge predict before it loses to a naive baseline?",
                 color=FG, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.grid(True, alpha=0.25)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=10)
    fig.text(0.01, 0.01,
             "Per-step MASE from a single h=168 forecast, macro over 7 BAs, 2024 val.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = OUT / "horizon_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    outputs = []
    if which in ("all", "leaderboard"):
        outputs.append(leaderboard())
    if which in ("all", "horizon"):
        outputs.append(horizon_curve())
    if which in ("all", "hero"):
        outputs.append(hero_forecast())
    for p in outputs:
        print(f"wrote {p}")
