"""Backfill EIA-930 hourly load, year-by-year, writing through to the store.

Usage:
    python scripts/backfill_eia_load.py [--ba PJM] [--start 2018] [--end 2026]

Dedup is handled by `surge.store.write_through`: each (ba, year) is keyed
separately so the script is safely restartable.
"""

from __future__ import annotations

import argparse
import time

from surge.scrapers import eia


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba", default="PJM", help="EIA BA code (e.g. PJM, CISO, ERCO)")
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2026, help="exclusive")
    args = ap.parse_args()

    total_rows = 0
    for year in range(args.start, args.end):
        t0 = time.monotonic()
        df = eia.load(ba=args.ba, start=f"{year}-01-01", end=f"{year + 1}-01-01")
        dt = time.monotonic() - t0
        total_rows += df.height
        print(f"  {args.ba} {year}: {df.height:>5} rows  ({dt:5.1f}s)")

    print(f"\nTotal: {total_rows} rows for {args.ba} {args.start}-{args.end - 1}")


if __name__ == "__main__":
    main()
