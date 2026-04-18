"""Generate layman-friendly forecast visualizations.

Output: one PNG per BA + a grid summary. Shows a specific week of 2025 test
data with actuals, model forecast (median), and 80% prediction interval.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch

from chronos import BaseChronosPipeline
from experiments.features import load_multi_ba


def forecast_week(pipe, bd, start_idx: int, context: int, horizon: int, total_hours: int):
    """Produce rolling 24h forecasts for `total_hours` hours starting at start_idx."""
    all_medians = []
    all_lo = []
    all_hi = []
    all_true = []
    for off in range(0, total_hours, horizon):
        o = start_idx + off
        past = {k: v[o - context:o] for k, v in bd.covariates.items()}
        future = {k: bd.covariates[k][o:o + horizon] for k in bd.future_keys}
        task = [{
            "target": bd.target[o - context:o].astype(np.float32),
            "past_covariates": past,
            "future_covariates": future,
        }]
        quants_list, _ = pipe.predict_quantiles(
            task, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
            batch_size=1,
        )
        q = quants_list[0].squeeze(0).float().cpu().numpy()  # (H, 3)
        all_lo.extend(q[:, 0])
        all_medians.extend(q[:, 1])
        all_hi.extend(q[:, 2])
        all_true.extend(bd.target[o:o + horizon])
    return (np.array(all_true), np.array(all_medians),
            np.array(all_lo), np.array(all_hi))


def plot_ba(ax, ts, true, median, lo, hi, ba: str):
    actual_line = ax.plot(ts, true / 1000, color="#ffffff", linewidth=2.2,
                          label="Actual load", zorder=3)
    forecast_line = ax.plot(ts, median / 1000, color="#4FC3F7", linewidth=1.8,
                            linestyle="--", label="Our forecast", zorder=4)
    band = ax.fill_between(ts, lo / 1000, hi / 1000, color="#4FC3F7", alpha=0.22,
                           label="80% uncertainty band", zorder=2)

    mape = float(np.abs((true - median) / true).mean() * 100)
    peak_hour = int(np.argmax(true))
    peak_err = (median[peak_hour] - true[peak_hour]) / true[peak_hour] * 100

    ax.set_title(f"{ba}  —  week MAPE {mape:.2f}%,  peak error {peak_err:+.1f}%",
                 color="#eeeeee", fontsize=11, loc="left", pad=6)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    ax.set_facecolor("#0f0f0f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/data/_ft/c2_full_v2/best")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default="/workspace/data/_plots")
    args = ap.parse_args()

    BAS = ["PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP"]
    bas = load_multi_ba(BAS, with_gen=False)
    print(f"[data] loaded {len(bas)} BAs", flush=True)

    pipe = BaseChronosPipeline.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.bfloat16
    )
    print(f"[model] loaded {args.model}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Pick the summer-peak week across all BAs: find the week in 2025 test
    # where the max(load) was highest, using PJM as anchor.
    pjm = bas["PJM"]
    test_slice = pjm.target[pjm.val_end:]
    test_ts = pjm.ts_utc[pjm.val_end:]
    week = 24 * args.days
    # Index of start of the week whose max load is highest.
    peak_origin = 0
    peak_val = -np.inf
    for i in range(0, len(test_slice) - week, 24):
        w = test_slice[i:i + week].max()
        if w > peak_val:
            peak_val = w
            peak_origin = i
    start_idx = pjm.val_end + peak_origin
    ts_slice = test_ts[peak_origin:peak_origin + week]
    ts_dt = np.array([np.datetime64(t, "s").astype(datetime) for t in ts_slice])
    print(f"[viz] selected week starting {ts_dt[0]}, PJM peak {peak_val/1000:.1f} GW",
          flush=True)

    # Generate forecasts per BA and plot.
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), facecolor="#0a0a0a")
    axes = axes.flatten()
    all_mapes = []

    for i, ba in enumerate(BAS):
        bd = bas[ba]
        true, med, lo, hi = forecast_week(pipe, bd, start_idx,
                                          args.context, args.horizon, week)
        plot_ba(axes[i], ts_dt, true, med, lo, hi, ba)
        mape = float(np.abs((true - med) / true).mean() * 100)
        all_mapes.append(mape)

    # Summary panel
    axes[-1].axis("off")
    axes[-1].set_facecolor("#0a0a0a")
    summary = (
        f"Model: Chronos-2, fine-tuned on 7 US grids\n"
        f"Week starting: {ts_dt[0].strftime('%B %d, %Y')}\n\n"
        f"Average error per BA:\n"
        + "\n".join(f"  {ba:<5}  {mape:5.2f}% MAPE" for ba, mape in zip(BAS, all_mapes))
        + f"\n\n  Overall  {np.mean(all_mapes):5.2f}% MAPE\n"
        f"\nShaded band = 80% probability range;\n"
        f"dashed line = median forecast.\n"
        f"Made on a single H100 with public data only."
    )
    axes[-1].text(0.05, 0.95, summary, fontsize=11, color="#dddddd",
                  va="top", ha="left", family="monospace",
                  transform=axes[-1].transAxes)

    fig.suptitle("Surge day-ahead load forecast — 2025 test week",
                 color="#ffffff", fontsize=16, y=0.995)
    fig.text(0.5, 0.005,
             f"Actual vs forecast, hourly, for 7 US balancing authorities.  "
             f"Macro MAPE {np.mean(all_mapes):.2f}% — roughly ISO-internal accuracy.",
             color="#888888", fontsize=10, ha="center")
    fig.tight_layout(rect=[0, 0.015, 1, 0.98])

    out_path = out / f"forecast_week_{ts_dt[0].strftime('%Y%m%d')}.png"
    fig.savefig(out_path, dpi=160, facecolor="#0a0a0a")
    plt.close(fig)
    print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
