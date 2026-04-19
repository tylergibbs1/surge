"""EIA DF vs surge head-to-head on the same evaluation window.

Fetches EIA's day-ahead demand forecast (type=DF) and actuals for each of
the 7 RTOs over a window, aligns on (ba, ts_utc), and computes MAE / RMSE
/ MAPE / MASE. The MASE denominator is the same per-BA train-set
seasonal-naive (m=24) MAE that surge uses, so the numbers are directly
comparable to the `experiments/results.tsv` test-set figures.

Usage:
    python scripts/compare_eia_df.py --start 2025-01-01 --end 2026-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime

import numpy as np
import polars as pl

from surge import store

HDRS = {
    "User-Agent": "surge-playground/1.0 (+github.com/tylergibbs1/surge)",
    "Referer": (
        "https://www.eia.gov/electricity/gridmonitor/dashboard/"
        "electric_overview/US48/US48"
    ),
}

RTOS = ["PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP"]


def _eia_fetch(ba: str, kind: str, start: datetime, end: datetime) -> dict[str, float]:
    """kind: 'D' (actual demand) or 'DF' (day-ahead forecast)."""
    qs = urllib.parse.urlencode({
        "respondent[0]": ba,
        "type[0]": kind,
        "start": start.strftime("%m%d%Y %H:00:00"),
        "end":   end.strftime("%m%d%Y %H:00:00"),
        "frequency": "hourly",
        "timezone": "Eastern",
        "limit": "10000",
        "offset": "0",
    })
    url = f"https://www.eia.gov/electricity/930-api/region_data/series_data?{qs}"
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
    try:
        vals = payload[0]["data"][0]["VALUES"]
    except (IndexError, KeyError):
        return {}
    out: dict[str, float] = {}
    for ts, v in zip(vals.get("DATES", []), vals.get("DATA", [])):
        if v is None:
            continue
        m, d, rest = ts.split("/", 2)
        y, tp = rest.split(" ", 1)
        out[f"{y}-{m}-{d}T{tp}Z"] = float(v)
    return out


def _eia_fetch_paged(ba: str, kind: str, start: datetime, end: datetime) -> dict[str, float]:
    """Fetch a long window in ≤10k-row pages so we don't truncate. EIA's
    endpoint returns at most 10,000 rows per request."""
    combined: dict[str, float] = {}
    cursor = start
    # 8760 hours fits in one page; pad with a 330-day cap to stay safe.
    while cursor < end:
        window_end = min(end, cursor.replace(hour=0) + _days(330))
        chunk = _eia_fetch(ba, kind, cursor, window_end)
        combined.update(chunk)
        cursor = window_end
        time.sleep(0.2)  # courtesy pause between pages
    return combined


def _days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


def _train_denom(ba: str) -> float:
    """Per-BA train-set seasonal-naive (m=24) MAE, matching experiments/eval.py."""
    df = (
        store.scan("load_hourly", dedupe_on=["ts_utc", "ba"])
        .filter(pl.col("ba") == ba)
        .sort("ts_utc")
        .collect()
    )
    df = df.with_columns(
        pl.when(pl.col("load_mw") > 200_000).then(None).otherwise(pl.col("load_mw")).alias("load_mw")
    )
    y = df["load_mw"].to_numpy().astype(np.float64)
    years = df["ts_utc"].dt.year().to_numpy()
    train_end = int(np.searchsorted(years, 2024, side="left"))
    train = y[:train_end]
    mask = ~np.isnan(train)
    train = train[mask]
    return float(np.nanmean(np.abs(train[24:] - train[:-24])))


def _score(actuals: dict[str, float], preds: dict[str, float], denom: float) -> dict:
    common = sorted(set(actuals) & set(preds))
    if not common:
        return {"n": 0}
    a = np.array([actuals[t] for t in common])
    p = np.array([preds[t]    for t in common])
    err = p - a
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    mape = float(np.mean(np.abs(err) / np.maximum(a, 1.0)) * 100)
    return {
        "n": len(common),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mase": mae / denom if denom > 0 else float("nan"),
        "bias": float(np.mean(err)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end",   default="2026-01-01")
    ap.add_argument("--bas",   nargs="+", default=RTOS)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    print(f"window: {start.date()} → {end.date()}  bas={args.bas}")
    per_ba: dict[str, dict] = {}
    for ba in args.bas:
        t0 = time.monotonic()
        actuals = _eia_fetch_paged(ba, "D",  start, end)
        df      = _eia_fetch_paged(ba, "DF", start, end)
        denom   = _train_denom(ba)
        m = _score(actuals, df, denom)
        m["wall_s"] = round(time.monotonic() - t0, 1)
        per_ba[ba] = m
        print(
            f"  {ba:<6} n={m['n']:>5}  MAE={m.get('mae', 0):>7.1f}  "
            f"RMSE={m.get('rmse', 0):>7.1f}  MAPE={m.get('mape', 0):>4.2f}%  "
            f"MASE={m.get('mase', 0):.4f}  bias={m.get('bias', 0):+.0f}  ({m['wall_s']}s)",
            flush=True,
        )

    # Macro average.
    keys = [k for k in ("mae", "rmse", "mape", "mase") if per_ba]
    macro = {k: float(np.mean([v[k] for v in per_ba.values() if v.get("n")])) for k in keys}
    print("\nmacro:")
    for k, v in macro.items():
        print(f"  {k}={v:.4f}")
    # Dump so the next step can pick it up.
    json.dump({"per_ba": per_ba, "macro": macro,
               "window": {"start": args.start, "end": args.end}},
              sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
