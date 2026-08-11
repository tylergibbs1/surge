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
    # future_mode="forecast" is the honest serving configuration: a real archived
    # day-ahead weather forecast, which is what the README caption claims. Never
    # switch this to "oracle"/"oracle_om" — the chart would then show a forecast
    # nobody can actually make.
    bas = load_multi_ba(BAS, with_gen=False, future_mode="forecast")

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
        "Causal covariates only.\nDay-ahead forecast weather;\nno realized values.\n"
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
        ("seasonal-naive-24  (baseline)",          1.0442, None,   None,   "#888"),
        ("XGBoost hourly-binned (Roy '25)",        1.0189, None,   None,   "#ff9800"),
        ("Chronos-2 zero-shot,  no future wx",     0.6203, 0.5986, 0.6439, "#2196F3"),
        ("surge-fm-v3,  no future wx",             0.5937, 0.5742, 0.6137, "#4CAF50"),
        ("Chronos-2 zero-shot  + forecast wx",     0.5636, 0.5471, 0.5800, "#2196F3"),
        ("surge-fm-v3  + forecast wx",             0.5357, 0.5212, 0.5502, "#4FC3F7"),
        ("(perfect wx foresight - NOT a forecast)", 0.4882, 0.4761, 0.5000, "#7E57C2"),
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
             "Denominator = per-BA train seasonal-naive (m=24). \"forecast wx\" = real archived\n"
             "day-ahead NWP forecast (causal). Realized values used only in the labelled bar.",
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
    # Causal measurement WITH real day-ahead forecast weather (future_mode=
    # forecast), 7 RTOs, 2025 test split. Crossover moves to h67 from h41
    # without weather - weather is what buys forecast horizon.
    matched = {
        1: 0.2102,
        6: 0.2986,
        24: 0.5321,
        72: 0.6876,
        168: 0.806,
    }
    # Per-step MASE from the h=168 run, in order of step_ahead. Crosses the
    # seasonal-naive line (1.0) at h41 — far earlier than the leaky run suggested.
    per_step = np.array([
        0.220, 0.281, 0.309, 0.336, 0.361, 0.380, 0.410, 0.413, 0.415, 0.413,
        0.426, 0.442, 0.469, 0.509, 0.562, 0.617, 0.686, 0.724, 0.772, 0.806,
        0.808, 0.798, 0.772, 0.760, 0.729, 0.691, 0.666, 0.644, 0.625, 0.607,
        0.578, 0.577, 0.563, 0.563, 0.567, 0.570, 0.590, 0.633, 0.696, 0.764,
        0.828, 0.877, 0.934, 0.972, 0.978, 0.974, 0.948, 0.932, 0.889, 0.833,
        0.789, 0.744, 0.709, 0.679, 0.657, 0.638, 0.623, 0.603, 0.606, 0.613,
        0.637, 0.685, 0.753, 0.828, 0.911, 0.967, 1.017, 1.057, 1.058, 1.040,
        1.005, 0.985, 0.945, 0.891, 0.851, 0.803, 0.763, 0.732, 0.693, 0.684,
        0.673, 0.667, 0.665, 0.666, 0.682, 0.733, 0.807, 0.882, 0.948, 1.004,
        1.057, 1.098, 1.102, 1.091, 1.059, 1.047, 0.997, 0.931, 0.889, 0.840,
        0.796, 0.762, 0.730, 0.707, 0.689, 0.665, 0.666, 0.672, 0.694, 0.746,
        0.822, 0.897, 0.991, 1.043, 1.089, 1.125, 1.124, 1.106, 1.071, 1.051,
        1.005, 0.952, 0.912, 0.862, 0.812, 0.784, 0.745, 0.729, 0.726, 0.718,
        0.711, 0.711, 0.727, 0.785, 0.864, 0.938, 1.001, 1.061, 1.115, 1.153,
        1.157, 1.149, 1.117, 1.103, 1.052, 0.991, 0.945, 0.895, 0.846, 0.808,
        0.779, 0.749, 0.730, 0.703, 0.704, 0.707, 0.730, 0.787, 0.862, 0.946,
        1.048, 1.108, 1.162, 1.199, 1.201, 1.176, 1.147, 1.129,
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
             "2025 test split, real day-ahead forecast weather (causal).",
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
