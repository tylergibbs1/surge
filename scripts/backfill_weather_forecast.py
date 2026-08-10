"""Backfill archived day-ahead forecast temperature for BA centroids.

    python scripts/backfill_weather_forecast.py --bas PJM CISO ... --start 2021-03-01

Writes `weather_fcst_hourly`. This is the causal counterpart to
`weather_hourly`: what was *forecast* for each hour roughly a day earlier,
rather than what was observed. See surge.scrapers.openmeteo for why that
distinction is the whole point.
"""
from __future__ import annotations

import argparse
from datetime import date

from surge import bas as _bas
from surge.scrapers.openmeteo import ARCHIVE_START, fetch_ba

RTOS = ["PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", nargs="+", default=RTOS,
                    help="BA codes; 'all' for every demand-reporting BA")
    ap.add_argument("--start", default=ARCHIVE_START.isoformat())
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--lead-days", type=int, default=1,
                    help="forecast lead time in days; 1 = day-ahead")
    args = ap.parse_args()

    codes = _bas.demand_codes() if args.bas == ["all"] else args.bas
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    total = 0
    for ba in codes:
        try:
            df = fetch_ba(ba, start, end, lead_days=args.lead_days)
        except Exception as e:                     # keep going; one BA is not fatal
            print(f"[{ba}] FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        total += df.height
        if df.is_empty():
            print(f"[{ba}] no rows", flush=True)
        else:
            print(f"[{ba}] {df.height:,} rows  "
                  f"{df['ts_utc'].min()} .. {df['ts_utc'].max()}", flush=True)
    print(f"TOTAL {total:,} rows across {len(codes)} BAs", flush=True)


if __name__ == "__main__":
    main()
