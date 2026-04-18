"""Backfill hourly ASOS temperature for all 7 BAs, 2018-2025."""
from __future__ import annotations

import time

from surge.scrapers.asos import BA_STATIONS, fetch_station


def main() -> None:
    total = 0
    for ba, station in BA_STATIONS.items():
        t0 = time.monotonic()
        df = fetch_station(station, ba, "2018-01-01", "2026-01-01")
        dt = time.monotonic() - t0
        total += df.height
        print(f"  {ba:<5} station=K{station}: {df.height:>5} hourly rows ({dt:5.1f}s)")
    print(f"\nTotal: {total:,} rows")


if __name__ == "__main__":
    main()
