"""Backfill EIA-930 hourly wind + solar generation for every demand BA, 2018-2025."""
from __future__ import annotations

import time

from surge import bas as _bas
from surge.scrapers.eia_genmix import fetch

BAS = _bas.demand_codes()
FUELS = ["WND", "SUN"]


def main() -> None:
    for ba in BAS:
        for fuel in FUELS:
            for year in range(2018, 2026):
                t0 = time.monotonic()
                df = fetch(ba, fuel, f"{year}-01-01", f"{year + 1}-01-01")
                dt = time.monotonic() - t0
                print(f"  {ba:<5} {fuel} {year}: {df.height:>5} rows ({dt:5.1f}s)")


if __name__ == "__main__":
    main()
