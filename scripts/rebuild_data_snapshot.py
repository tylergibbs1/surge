"""Compact the on-disk parquet store into `data_snapshot/` for Modal/Docker.

Reads from SURGE_DATA_DIR (default ~/.surge/data), rewrites one parquet per
(BA, year, month) partition into `<repo>/data_snapshot/` so Modal's
add_local_dir ships a tidy copy rather than thousands of tiny files.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from surge import store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_snapshot"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in ("load_hourly", "weather_hourly"):
        df = store.scan(ds).collect()
        if df.is_empty():
            print(f"[{ds}] empty; skipping")
            continue
        for (year, month), sub in df.group_by(["year", "month"]):
            p = OUT / ds / f"year={int(year):04d}" / f"month={int(month):02d}"
            p.mkdir(parents=True, exist_ok=True)
            sub.drop(["year", "month"]).write_parquet(p / "part.parquet", compression="zstd")
        n_files = sum(1 for _ in (OUT / ds).rglob("*.parquet"))
        print(f"[{ds}] {df.height:,} rows → {n_files} files")


if __name__ == "__main__":
    main()
