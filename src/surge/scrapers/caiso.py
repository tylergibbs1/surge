"""CAISO OASIS scraper. No credentials required.

Endpoint: https://oasis.caiso.com/oasisapi/SingleZip
Docs:     https://www.caiso.com/documents/oasis-apispecification.pdf

Every query returns a zipfile containing one or more CSVs. Responses are
capped at ~31 days per call, so historical backfills must be chunked.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import polars as pl

from surge import store
from surge.scrapers.base import DEFAULT_TIMEOUT, client, get

BASE_URL = "https://oasis.caiso.com/oasisapi/SingleZip"

# CAISO OASIS queryname -> (default version, default market_run_id)
QUERIES: dict[str, tuple[int, str]] = {
    # LMP / SPP
    "PRC_LMP":        (12, "DAM"),        # Day-ahead hourly LMP
    "PRC_INTVL_LMP":  (3,  "RTM"),        # Real-time 5-min LMP
    "PRC_RTPD_LMP":   (3,  "RTPD"),       # 15-min LMP
    # Load forecast
    "SLD_FCST":       (1,  "DAM"),        # System load forecast
    "SLD_REN_FCST":   (1,  "DAM"),        # Wind/solar forecast
    # Generation / schedules
    "ENE_SLRS":       (1,  "DAM"),        # Scheduled load/resources
    # Ancillary services
    "AS_RESULTS":     (1,  "DAM"),
}


def _fmt(dt: datetime) -> str:
    """CAISO wants YYYYMMDDTHH:mm-0000 (UTC with zulu offset literal)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y%m%dT%H:%M-0000")


def _iter_windows(start: datetime, end: datetime, days: int = 28) -> Iterable[tuple[datetime, datetime]]:
    """CAISO caps most queries at ~31 days. Emit (start, end) windows covering the range."""
    cur = start
    delta = timedelta(days=days)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt


def fetch(
    queryname: str,
    start: datetime,
    end: datetime,
    *,
    node: str | None = None,
    market_run_id: str | None = None,
    version: int | None = None,
    extra: dict[str, str] | None = None,
    persist: bool = True,
) -> list[pl.DataFrame]:
    """Fetch one OASIS query, auto-windowing the date range.

    Returns a list of DataFrames (one per CSV member across all windows).
    When persist=True, each window is written to the `caiso_{queryname}`
    dataset keyed by the (node, start, end) tuple.
    """
    dver, dmarket = QUERIES.get(queryname, (1, "DAM"))
    version = version or dver
    market_run_id = market_run_id or dmarket

    ds_name = f"caiso_{queryname.lower()}"
    frames: list[pl.DataFrame] = []
    with client(timeout=DEFAULT_TIMEOUT) as c:
        for win_start, win_end in _iter_windows(start, end):
            params = {
                "queryname": queryname,
                "startdatetime": _fmt(win_start),
                "enddatetime": _fmt(win_end),
                "version": version,
                "market_run_id": market_run_id,
                "resultformat": 6,  # CSV
            }
            if node:
                params["node"] = node
            if extra:
                params.update(extra)
            r = get(c, BASE_URL, params=params)
            win_frames = _unzip(r.content)
            frames.extend(win_frames)
            if persist and win_frames:
                key = f"{node or 'ALL'}:{_fmt(win_start)}:{_fmt(win_end)}:{market_run_id}"
                merged = pl.concat(win_frames, how="diagonal_relaxed").with_columns(
                    pl.lit(win_start).alias("_window_start"),
                )
                store.write_through(
                    ds_name, merged,
                    source="caiso-oasis", key=key, ts_col="_window_start",
                )
    return frames


# Zip-bomb defences (match ERCOT scraper).
_MAX_ZIP_MEMBERS = 64
_MAX_ZIP_DECOMPRESSED = 256 * 1024 * 1024  # 256 MB


def _unzip(blob: bytes) -> list[pl.DataFrame]:
    """Extract every CSV member from an OASIS response zip.

    Rejects archives that exceed safe caps or contain traversal member
    names.
    """
    out: list[pl.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = zf.namelist()
        if len(members) > _MAX_ZIP_MEMBERS:
            raise ValueError(f"zip has {len(members)} members (cap {_MAX_ZIP_MEMBERS})")
        total = 0
        for name in members:
            if name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
                raise ValueError(f"unsafe zip member name: {name!r}")
            if not name.lower().endswith(".csv"):
                continue
            total += zf.getinfo(name).file_size
            if total > _MAX_ZIP_DECOMPRESSED:
                raise ValueError("zip decompressed size exceeds cap")
            with zf.open(name) as fh:
                data = fh.read(_MAX_ZIP_DECOMPRESSED)
                if data.strip():
                    out.append(pl.read_csv(data))
    return out


# --- Convenience wrappers --------------------------------------------------
def lmp_dam(
    node: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Day-ahead hourly LMP for one node. Returns a single concatenated frame."""
    frames = fetch("PRC_LMP", start, end, node=node, market_run_id="DAM")
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def lmp_rtm(
    node: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Real-time 5-minute LMP for one node."""
    frames = fetch("PRC_INTVL_LMP", start, end, node=node, market_run_id="RTM")
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def load_forecast(start: datetime, end: datetime, run: str = "DAM") -> pl.DataFrame:
    frames = fetch("SLD_FCST", start, end, market_run_id=run)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def renewable_forecast(start: datetime, end: datetime, run: str = "DAM") -> pl.DataFrame:
    frames = fetch("SLD_REN_FCST", start, end, market_run_id=run)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
