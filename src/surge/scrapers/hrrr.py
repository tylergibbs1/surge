"""NOAA HRRR weather loader via AWS Open Data (anonymous HTTPS).

HRRR (3 km CONUS forecast) is published hourly at:
    https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/
        hrrr.tHHz.{product}f{FF}.grib2
        hrrr.tHHz.{product}f{FF}.grib2.idx

`.idx` files are tiny text sidecars mapping GRIB2 record numbers to byte
ranges. By pulling the idx first, we can do HTTP Range requests to download
only the variables we need — typically <1 MB per forecast hour instead of
the full 100+ MB GRIB file.

This module gives you:
    - list_runs(date) : available init hours for a given day
    - idx(init, fhour, product) : parse the .idx sidecar
    - download_slice(init, fhour, product, variable) : range-get bytes
    - point_timeseries(lat, lon, variable, start, end) : extract a single
       grid cell across many forecast hours into a polars frame.

Parsing GRIB2 requires `cfgrib` (optional dependency in the `weather` extra).
For byte-range fetches and idx parsing alone, only `httpx` is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Iterable

import httpx

from surge.scrapers.base import DEFAULT_TIMEOUT, client, get

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
PRODUCTS = ("wrfsfcf", "wrfprsf", "wrfnatf", "wrfsubhf")  # surface, pressure, native, subhourly


def _date_prefix(dt: datetime) -> str:
    return f"hrrr.{dt:%Y%m%d}/conus"


def grib_url(init: datetime, fhour: int, product: str = "wrfsfcf") -> str:
    hh = f"{init.hour:02d}"
    ff = f"{fhour:02d}"
    return f"{BUCKET}/{_date_prefix(init)}/hrrr.t{hh}z.{product}{ff}.grib2"


def idx_url(init: datetime, fhour: int, product: str = "wrfsfcf") -> str:
    return grib_url(init, fhour, product) + ".idx"


@dataclass(frozen=True)
class GribRecord:
    """One record in a GRIB2 file as described by the .idx sidecar."""
    record_no: int
    byte_start: int
    ref_time: str
    variable: str      # e.g. "TMP"
    level: str         # e.g. "2 m above ground"
    forecast: str      # e.g. "1 hour fcst"

    def byte_range(self, byte_end: int | None) -> str:
        end = "" if byte_end is None else str(byte_end - 1)
        return f"bytes={self.byte_start}-{end}"


_IDX_LINE = re.compile(
    r"^(?P<n>\d+):(?P<byte>\d+):d=(?P<ref>\d+):(?P<var>[^:]+):(?P<lvl>[^:]+):(?P<fcst>[^:]+):"
)


def parse_idx(text: str) -> list[GribRecord]:
    records: list[GribRecord] = []
    for line in text.splitlines():
        m = _IDX_LINE.match(line)
        if not m:
            continue
        records.append(GribRecord(
            record_no=int(m["n"]),
            byte_start=int(m["byte"]),
            ref_time=m["ref"],
            variable=m["var"].strip(),
            level=m["lvl"].strip(),
            forecast=m["fcst"].strip(),
        ))
    return records


def idx(init: datetime, fhour: int, product: str = "wrfsfcf") -> list[GribRecord]:
    with client(timeout=DEFAULT_TIMEOUT) as c:
        r = get(c, idx_url(init, fhour, product))
        return parse_idx(r.text)


def find(
    records: list[GribRecord],
    *,
    variable: str,
    level: str | None = None,
) -> GribRecord:
    """Find the record matching variable (and optionally level)."""
    for rec in records:
        if rec.variable == variable and (level is None or rec.level == level):
            return rec
    raise KeyError(f"no GRIB record for variable={variable} level={level}")


def download_slice(
    init: datetime,
    fhour: int,
    *,
    variable: str,
    level: str | None = None,
    product: str = "wrfsfcf",
    client_: httpx.Client | None = None,
) -> bytes:
    """Fetch only the bytes for a single variable/level using the idx sidecar."""
    owned = client_ is None
    c = client_ or httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    try:
        idx_resp = get(c, idx_url(init, fhour, product))
        records = parse_idx(idx_resp.text)
        hit = find(records, variable=variable, level=level)
        next_rec = next(
            (r for r in records if r.record_no == hit.record_no + 1), None
        )
        byte_end = next_rec.byte_start if next_rec else None
        r = c.get(grib_url(init, fhour, product),
                  headers={"Range": hit.byte_range(byte_end)})
        r.raise_for_status()
        return r.content
    finally:
        if owned:
            c.close()


def list_runs(date: datetime) -> list[datetime]:
    """List init timestamps available for a given UTC date.

    HRRR runs every hour; we optimistically enumerate 00z..23z and rely on
    the caller to skip missing runs.
    """
    return [datetime.combine(date.date(), time(h), tzinfo=timezone.utc) for h in range(24)]


def iter_forecasts(
    start: datetime,
    end: datetime,
    *,
    fhour: int = 1,
) -> Iterable[datetime]:
    """Yield init timestamps between start/end (inclusive, hourly)."""
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur <= end:
        yield cur
        cur += timedelta(hours=1)


# Parsing GRIB2 bytes into a numpy array requires cfgrib. Keep the heavy
# import lazy so `surge.scrapers.hrrr` is usable for listing / slicing
# without the optional weather extras installed.
def parse_grib(blob: bytes):  # pragma: no cover — lazy import
    """Decode a GRIB2 byte blob to an xarray DataArray. Requires `cfgrib`."""
    import tempfile

    import cfgrib  # type: ignore[import-not-found]
    import xarray as xr  # type: ignore[import-not-found]

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as fh:
        fh.write(blob)
        path = fh.name
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    return ds
