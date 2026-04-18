"""Incremental data refresh: pull the last N days of load + weather for all BAs.

Usage:
    python -m surge.ingest                 # last 7 days, all configured BAs
    python -m surge.ingest --days 30
    python -m surge.ingest --bas PJM CISO

Cron-safe: every call is idempotent (the store's write_through dedups by
(source, key), so overlapping windows don't duplicate rows).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from surge.scrapers import eia
from surge.scrapers.asos import BA_STATIONS, fetch_station


BAS = ("PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP")
log = logging.getLogger("surge.ingest")


def _today_utc() -> datetime:
    return datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def refresh(bas: list[str], days: int) -> dict[str, int]:
    """Pull the last `days` of load + ASOS temp for each BA. Returns row counts."""
    end = _today_utc() + timedelta(days=1)
    start = end - timedelta(days=days)
    start_s = start.date().isoformat()
    end_s = end.date().isoformat()

    counts: dict[str, int] = {}
    for ba in bas:
        # Load (EIA-930 type=D demand)
        try:
            df = eia.load(ba=ba, start=start_s, end=end_s)
            counts[f"load:{ba}"] = df.height
            log.info("load %s: %d rows [%s..%s]", ba, df.height, start_s, end_s)
        except Exception as e:  # pragma: no cover
            log.error("load %s failed: %s", ba, e)
            counts[f"load:{ba}"] = -1

        # Weather via ASOS
        station = BA_STATIONS.get(ba)
        if station is None:
            continue
        try:
            df = fetch_station(station, ba, start_s, end_s)
            counts[f"weather:{ba}"] = df.height
            log.info("weather %s(%s): %d rows", ba, station, df.height)
        except Exception as e:  # pragma: no cover
            log.error("weather %s failed: %s", ba, e)
            counts[f"weather:{ba}"] = -1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Surge incremental data refresh.")
    ap.add_argument("--bas", nargs="+", default=list(BAS),
                    help="EIA BA codes to refresh (default: all 7)")
    ap.add_argument("--days", type=int, default=7,
                    help="Refresh the last N days (default: 7)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    counts = refresh(args.bas, args.days)
    total = sum(v for v in counts.values() if v > 0)
    failed = [k for k, v in counts.items() if v < 0]
    log.info("done — %d rows written, %d failures", total, len(failed))
    if failed:
        log.error("failures: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
