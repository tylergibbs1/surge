"""Append settled outcome verifications for mature live forecasts."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

from surge import ledger, verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--maturity-hours", type=int, default=verification.MATURITY_HOURS)
    args = parser.parse_args()

    now = datetime.now(tz=UTC)
    records = ledger.list_forecasts(
        mode=ledger.ForecastMode.LIVE,
        limit=args.limit,
    )
    eligible = 0
    committed = 0
    failures: list[str] = []
    for record in records:
        if now < record.last_valid_at_utc + timedelta(hours=args.maturity_hours):
            continue
        eligible += 1
        try:
            committed += verification.verify_forecast(
                record.issuance_id,
                verified_at_utc=now,
                maturity_hours=args.maturity_hours,
            )
        except Exception as exc:
            failures.append(f"{record.issuance_id}: {exc}")

    print(
        f"eligible={eligible} committed_points={committed} failures={len(failures)}"
    )
    for failure in failures:
        print(failure)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
