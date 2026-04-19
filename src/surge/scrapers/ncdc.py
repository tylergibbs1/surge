"""NOAA NCDC / NCEI weather data scraper.

Two pipelines in one module:

1. Storm Events Database — direct CSV download, no auth required.
   https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
   Public flat files, one per year, gzipped. Useful for outage modeling.

2. CDO web service (GHCND / GSOM / GSOY daily & aggregated station obs).
   https://www.ncei.noaa.gov/cdo-web/api/v2/
   Requires NCDC_TOKEN in env. Rate-limited to 5 req/sec, 10k req/day.
"""

from __future__ import annotations

import gzip
import io
import os
import re
from datetime import UTC, datetime
from typing import Any

import polars as pl

from surge import store
from surge.scrapers.base import client, get

STORM_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"
CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"


# --- Storm Events ----------------------------------------------------------
_STORM_FILE = re.compile(
    r'href="(StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz)"'
)


def storm_events_index() -> list[tuple[int, str, str]]:
    """List (year, revision_date, filename) tuples available on the NCEI index.

    The index is a plain-Apache listing; we scrape hrefs.
    """
    with client() as c:
        r = get(c, STORM_BASE + "/")
    rows: list[tuple[int, str, str]] = []
    for m in _STORM_FILE.finditer(r.text):
        rows.append((int(m.group(2)), m.group(3), m.group(1)))
    rows.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return rows


def storm_events(year: int, *, persist: bool = True) -> pl.DataFrame:
    """Download one year of storm events (details). Newest revision wins."""
    index = storm_events_index()
    hit = next((row for row in index if row[0] == year), None)
    if hit is None:
        raise ValueError(f"no storm events file available for year {year}")
    _, _, fname = hit
    with client() as c:
        r = get(c, f"{STORM_BASE}/{fname}")
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    df = pl.read_csv(io.StringIO(text), infer_schema_length=10_000, ignore_errors=True)
    if persist:
        ts_col = "BEGIN_DATE_TIME" if "BEGIN_DATE_TIME" in df.columns else None
        if ts_col:
            df2 = df.with_columns(
                pl.col(ts_col).str.to_datetime(
                    "%d-%b-%y %H:%M:%S", strict=False, time_zone="UTC"
                ).alias("ts_utc")
            )
        else:
            df2 = df.with_columns(
                pl.lit(datetime(year, 1, 1, tzinfo=UTC)).alias("ts_utc")
            )
        store.write_through(
            "storm_events", df2,
            source="ncei-swdi", key=f"year={year}:file={fname}",
        )
    return df


# --- CDO web service -------------------------------------------------------
def _token() -> str:
    tok = os.environ.get("NCDC_TOKEN")
    if not tok:
        raise RuntimeError(
            "NCDC_TOKEN missing. Request one at https://www.ncdc.noaa.gov/cdo-web/token"
        )
    return tok


def _cdo_paged(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Page through a CDO v2 endpoint, 1000 records per call."""
    items: list[dict[str, Any]] = []
    params = {**params, "limit": 1000, "offset": 1}
    with client(headers={"token": _token()}) as c:
        while True:
            r = get(c, f"{CDO_BASE}/{path}", params=params)
            payload = r.json()
            results = payload.get("results", [])
            items.extend(results)
            meta = payload.get("metadata", {}).get("resultset", {})
            count = int(meta.get("count", 0))
            if params["offset"] + len(results) - 1 >= count or not results:
                break
            params["offset"] += len(results)
    return items


def ghcnd(
    station_id: str,
    start: str,
    end: str,
    *,
    datatypes: list[str] | None = None,
    persist: bool = True,
) -> pl.DataFrame:
    """Daily summaries for one GHCN-D station.

    datatypes: GHCND variables to fetch (e.g. ["TMAX","TMIN","PRCP"]).
               Default: all available.
    """
    params: dict[str, Any] = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start,
        "enddate": end,
        "units": "metric",
    }
    if datatypes:
        params["datatypeid"] = datatypes

    rows = _cdo_paged("data", params)
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows).with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_zone="UTC").alias("ts_utc"),
        pl.lit(station_id).alias("station"),
        pl.lit("ghcnd").alias("source"),
    )
    if persist:
        store.write_through(
            "weather_daily", df,
            source="ghcnd", key=f"{station_id}:{start}:{end}",
        )
    return df


def stations(
    *,
    dataset: str = "GHCND",
    location: str | None = None,
    extent: str | None = None,
) -> pl.DataFrame:
    """List stations. `location` is a CDO location code (e.g. 'FIPS:48' for Texas)."""
    params: dict[str, Any] = {"datasetid": dataset}
    if location:
        params["locationid"] = location
    if extent:
        params["extent"] = extent
    rows = _cdo_paged("stations", params)
    return pl.DataFrame(rows) if rows else pl.DataFrame()
