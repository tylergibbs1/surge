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
import time
from datetime import UTC, datetime, timedelta

from surge import bas as _bas
from surge.scrapers import eia
from surge.scrapers.asos import BA_STATIONS, fetch_station

# Default to every BA that publishes a demand (load) series — the
# forecastable subset. Callers can override with --bas. See `surge.bas` for
# the full registry (which also includes generator-only BAs).
BAS = tuple(_bas.demand_codes())
log = logging.getLogger("surge.ingest")


def _today_utc() -> datetime:
    return datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def refresh(
    bas: list[str],
    days: int,
    *,
    skip_load: bool = False,
    skip_weather: bool = False,
) -> dict[str, int]:
    """Pull the last `days` of load + ASOS temp for each BA. Returns row counts.

    `skip_load` / `skip_weather` let you split an ingest across sources —
    handy when Iowa Mesonet is rate-limiting and you want to complete the
    EIA pull first without racing the weather retries.
    """
    end = _today_utc() + timedelta(days=1)
    start = end - timedelta(days=days)
    start_s = start.date().isoformat()
    end_s = end.date().isoformat()

    counts: dict[str, int] = {}
    for i, ba in enumerate(bas):
        # Small inter-BA pause — prevents us hammering Iowa Mesonet and
        # gives EIA-930 its quota a breather on long multi-BA pulls. The
        # per-scraper retry/backoff still runs on top of this.
        if i > 0:
            time.sleep(1.5)

        if not skip_load:
            try:
                df = eia.load(ba=ba, start=start_s, end=end_s)
                counts[f"load:{ba}"] = df.height
                log.info("load %s: %d rows [%s..%s]", ba, df.height, start_s, end_s)
            except Exception as e:  # pragma: no cover
                log.error("load %s failed: %s", ba, e)
                counts[f"load:{ba}"] = -1

        if not skip_weather:
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
                    help=f"EIA BA codes to refresh (default: all {len(BAS)} "
                         "BAs with a demand series; see surge.bas)")
    ap.add_argument("--days", type=int, default=7,
                    help="Refresh the last N days (default: 7)")
    ap.add_argument("--skip-load", action="store_true",
                    help="Skip EIA-930 demand pull (weather only)")
    ap.add_argument("--skip-weather", action="store_true",
                    help="Skip Iowa Mesonet weather pull (load only)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    counts = refresh(
        args.bas, args.days,
        skip_load=args.skip_load, skip_weather=args.skip_weather,
    )
    total = sum(v for v in counts.values() if v > 0)
    failed = [k for k, v in counts.items() if v < 0]
    log.info("done — %d rows written, %d failures", total, len(failed))
    if failed:
        log.error("failures: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
