"""Generate the three hero charts for the README.

Outputs (all PNG @ 2×):
    docs/plots/hero_forecast.png       — 7-BA weekly forecast fan chart
    docs/plots/leaderboard.png         — horizontal bar chart of test MASE vs CIs
    docs/plots/horizon_curve.png       — MASE vs forecast horizon with naive line

Run with the local venv:
    .venv/bin/python docs/make_readme_plots.py
"""
from __future__ import annotations

import os
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
MODEL_PATH = Path(os.environ.get("SURGE_MODEL_PATH",
                                str(ROOT / "models" / "chronos2_full_v2")))
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
    # future_mode="none" is the honest serving configuration: no future weather,
    # which is what the README caption claims. Do not switch this to "oracle" —
    # the chart would then show a forecast nobody can actually make.
    bas = load_multi_ba(BAS, with_gen=False, future_mode="none")

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
        def _flatlined(series: np.ndarray, run: int = 8) -> bool:
            """True if `run` or more consecutive hours are identical — the
            signature of a forward-filled gap, which reads as a drawing error
            in the hero chart even though the forecast is fine."""
            if len(series) <= run:
                return False
            same = np.diff(series) == 0
            best = cur = 0
            for flag in same:
                cur = cur + 1 if flag else 0
                best = max(best, cur)
            return best >= run - 1

        bad_window = False
        for other in BAS:
            od = bas[other]
            ow = od.target[od.val_end + i:od.val_end + i + week_hours]
            if len(ow) < week_hours or _flatlined(ow):
                bad_window = True
                break
        if bad_window or bas["SWPP"].target[bas["SWPP"].val_end + i:
                                            bas["SWPP"].val_end + i + week_hours].min() < 5_000:
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
            future = bd.future_at(o, horizon)
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
        "Causal covariates only — no\nfuture weather or renewables.\n"
        "Model: Chronos-2 fine-tuned on\n53 BAs × 7 years of public data."
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
    # Re-measured under causal covariates on the pinned 2025 window with a
    # deduplicated store. Rows we have NOT re-scored under this protocol
    # (Prophet, N-BEATS, Chronos-Bolt, surge-fm-v2) are deliberately absent
    # rather than carried over from the superseded run — plotting them beside
    # these bars would recreate the apples-to-oranges comparison being fixed.
    data = [
        ("seasonal-naive-24  (baseline)",        1.0442, None,   None,   "#888"),
        ("XGBoost hourly-binned (Roy '25)",      1.0189, None,   None,   "#ff9800"),
        ("Chronos-2 zero-shot",                  0.6203, 0.5986, 0.6439, "#2196F3"),
        ("surge-fm-v3",                          0.5937, 0.5742, 0.6137, "#4CAF50"),
        ("surge-fm-v3  + peer-BA load",          0.5804, 0.5630, 0.5977, "#4CAF50"),
        ("surge-fm-v3  + peers, re-adapted",     0.5720, 0.5555, 0.5877, "#4FC3F7"),
        ("(perfect foresight — NOT a forecast)", 0.4683, 0.4542, 0.4822, "#7E57C2"),
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
    ax.set_xlim(0, 1.25)
    ax.set_title("Surge vs. baselines, 2025 hold-out — causal covariates only",
                 color=FG, fontsize=13, fontweight="bold", loc="left", pad=12)
    for i, m in enumerate(mase):
        ax.text(m + err_hi[i] + 0.03, i, f"{m:.3f}", color=FG, fontsize=9, va="center")
    ax.grid(True, axis="x", alpha=0.25)
    for s in ax.spines.values():
        s.set_color("#333")
    fig.text(0.01, 0.01,
             "7 US RTOs, macro MASE, 365 rolling 24h-ahead windows over calendar 2025.\n"
             "Denominator = per-BA train seasonal-naive (m=24). No future weather or\n"
             "renewables except the labelled perfect-foresight bar.",
             color=MUTED, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    path = OUT / "leaderboard.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# 3. HORIZON CURVE — MASE vs forecast horizon, with naive break-even
# ------------------------------------------------------------------
def horizon_curve() -> Path:
    # Causal-covariate measurement (future_mode=none), 7 RTOs, 2025 test split.
    matched = {
        1: 0.2129,
        6: 0.3054,
        24: 0.572,
        72: 0.9009,
        168: 1.2021,
    }
    # Per-step MASE from the h=168 run, in order of step_ahead. Crosses the
    # seasonal-naive line (1.0) at h41 — far earlier than the leaky run suggested.
    per_step = np.array([
        0.229, 0.284, 0.309, 0.328, 0.351, 0.369, 0.402, 0.414, 0.418, 0.424,
        0.450, 0.484, 0.528, 0.566, 0.614, 0.671, 0.728, 0.786, 0.854, 0.896,
        0.917, 0.924, 0.916, 0.914, 0.897, 0.856, 0.831, 0.807, 0.776, 0.751,
        0.724, 0.716, 0.706, 0.702, 0.716, 0.740, 0.781, 0.832, 0.897, 0.968,
        1.057, 1.148, 1.238, 1.300, 1.327, 1.335, 1.313, 1.307, 1.278, 1.206,
        1.157, 1.099, 1.046, 1.000, 0.961, 0.935, 0.912, 0.893, 0.901, 0.918,
        0.953, 1.008, 1.083, 1.168, 1.269, 1.363, 1.459, 1.520, 1.546, 1.544,
        1.520, 1.506, 1.455, 1.384, 1.325, 1.264, 1.199, 1.143, 1.101, 1.070,
        1.046, 1.027, 1.023, 1.031, 1.056, 1.112, 1.189, 1.274, 1.373, 1.478,
        1.576, 1.643, 1.669, 1.673, 1.643, 1.636, 1.595, 1.511, 1.455, 1.382,
        1.310, 1.254, 1.209, 1.174, 1.141, 1.115, 1.118, 1.123, 1.157, 1.215,
        1.296, 1.383, 1.481, 1.578, 1.668, 1.727, 1.754, 1.748, 1.717, 1.699,
        1.651, 1.583, 1.521, 1.459, 1.381, 1.322, 1.275, 1.237, 1.214, 1.193,
        1.187, 1.187, 1.215, 1.279, 1.362, 1.446, 1.549, 1.655, 1.747, 1.807,
        1.829, 1.824, 1.787, 1.771, 1.732, 1.643, 1.587, 1.515, 1.436, 1.376,
        1.326, 1.284, 1.252, 1.224, 1.224, 1.228, 1.262, 1.322, 1.409, 1.498,
        1.588, 1.684, 1.771, 1.825, 1.857, 1.854, 1.826, 1.803,
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
    ax.set_xlim(0, 170); ax.set_ylim(0, max(1.35, float(per_step.max()) * 1.08))
    ax.set_xticks([1, 24, 48, 72, 120, 168])
    ax.set_title("How far ahead can Surge predict before it loses to a naive baseline?",
                 color=FG, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.grid(True, alpha=0.25)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=10)
    fig.text(0.01, 0.01,
             "Per-step MASE from a single h=168 forecast, macro over 7 US RTOs, "
             "2025 test split, causal covariates (no future weather).",
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
