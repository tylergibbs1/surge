"""Parquet-backed datastore for Surge.

Layout on disk (root configurable via SURGE_DATA_DIR or ~/.surge/data):

    data/
      load_hourly/
        year=2024/month=01/part-<uuid>.parquet
        year=2024/month=02/...
      lmp/
        year=2024/month=01/...
      gen_by_fuel/...
      solar_actual/...
      wind_actual/...
      _manifest/
        <dataset>.parquet      (tracks which doc_ids have been ingested)

Design:
- Hive-style partitioning on (year, month) so DuckDB / polars can prune.
- One file per append; never mutated. Dedup is handled on read via the
  manifest, or at query time by `distinct` on the primary key.
- Manifest is a separate parquet per dataset. Rows:
    {source, key, fetched_at, n_rows, sha256}.

No schema enforcement at write time — callers are expected to produce frames
matching `surge.schemas`. The store's job is durability and partitioning.
"""

from __future__ import annotations

import hashlib
import io
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

DATASETS = ("load_hourly", "lmp", "gen_by_fuel", "solar_actual", "wind_actual",
            "renewable_forecast", "system_lambda", "weather_point")


def _root() -> Path:
    env = os.environ.get("SURGE_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".surge" / "data"


def dataset_path(name: str) -> Path:
    return _root() / name


def _partition_path(name: str, year: int, month: int) -> Path:
    return dataset_path(name) / f"year={year:04d}" / f"month={month:02d}"


def _sha256(df: pl.DataFrame) -> str:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="uncompressed")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def append(
    name: str,
    df: pl.DataFrame,
    *,
    ts_col: str = "ts_utc",
) -> Path:
    """Append a frame to a dataset, Hive-partitioned on ts_col's (year, month).

    Returns the root dataset path. Partitions are created on demand.
    """
    if df.is_empty():
        return dataset_path(name)
    if ts_col not in df.columns:
        raise ValueError(f"dataframe missing partition column '{ts_col}'")

    df = df.with_columns(
        pl.col(ts_col).dt.year().alias("_year"),
        pl.col(ts_col).dt.month().alias("_month"),
    )
    for (year, month), chunk in df.group_by(["_year", "_month"]):  # type: ignore[misc]
        partition = _partition_path(name, int(year), int(month))
        partition.mkdir(parents=True, exist_ok=True)
        fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
        chunk.drop(["_year", "_month"]).write_parquet(
            partition / fname, compression="zstd"
        )
    return dataset_path(name)


def scan(
    name: str,
    *,
    dedupe_on: list[str] | None = None,
    recency_col: str = "as_of",
) -> pl.LazyFrame:
    """Lazy scan of an entire dataset. Returns a LazyFrame for .filter / .select.

    When `dedupe_on` is set, the scan deduplicates on those columns
    keeping the row with the greatest `recency_col` (latest write wins).
    Use this whenever `append()` may have written the same business key
    twice — e.g. overlapping backfill windows, EIA in-place revisions.
    Without it, aggregates silently double.

    Examples:
        # Raw append-only stream
        store.scan("load_hourly")

        # One canonical row per (ts_utc, ba), latest write wins
        store.scan("load_hourly", dedupe_on=["ts_utc", "ba"])
    """
    root = dataset_path(name)
    if not root.exists():
        return pl.LazyFrame()
    lf = pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)
    if dedupe_on is None:
        return lf
    # Sort on `recency_col` too when it exists so the latest write wins
    # ties (EIA in-place revisions). Datasets without it (test fixtures,
    # older snapshots) still dedupe deterministically on the key alone.
    has_recency = recency_col in lf.collect_schema().names()
    if has_recency:
        sort_cols = [*dedupe_on, recency_col]
        descending = [False] * len(dedupe_on) + [True]
    else:
        sort_cols = list(dedupe_on)
        descending = [False] * len(dedupe_on)
    return (
        lf.sort(sort_cols, descending=descending)
          .unique(subset=dedupe_on, keep="first")
    )


# --- Manifest (dedup / backfill tracking) ---------------------------------
def _manifest_path(name: str) -> Path:
    return _root() / "_manifest" / f"{name}.parquet"


def record(
    name: str,
    *,
    source: str,
    key: str,
    n_rows: int,
    sha256: str | None = None,
) -> None:
    """Record an ingested batch in the dataset's manifest."""
    path = _manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pl.DataFrame({
        "source": [source],
        "key": [key],
        "fetched_at": [datetime.now(tz=UTC)],
        "n_rows": [n_rows],
        "sha256": [sha256 or ""],
    })
    if path.exists():
        pl.concat([pl.read_parquet(path), row]).write_parquet(path, compression="zstd")
    else:
        row.write_parquet(path, compression="zstd")


def ingested_keys(name: str, source: str) -> set[str]:
    """Return set of (key,) already ingested for this source. Used to skip re-downloads."""
    path = _manifest_path(name)
    if not path.exists():
        return set()
    m = pl.read_parquet(path).filter(pl.col("source") == source)
    return set(m.get_column("key").to_list())


def write_through(
    name: str,
    df: pl.DataFrame,
    *,
    source: str,
    key: str,
    ts_col: str = "ts_utc",
) -> None:
    """Append + record in one shot, with idempotent dedup on (source, key)."""
    if key in ingested_keys(name, source):
        return
    append(name, df, ts_col=ts_col)
    record(name, source=source, key=key, n_rows=df.height, sha256=_sha256(df))
