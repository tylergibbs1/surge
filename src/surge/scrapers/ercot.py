"""ERCOT public file-drop scraper (no credentials required).

ERCOT publishes every public report as a ZIP file drop. The listing and the
downloads are fully anonymous — no subscription key, no OAuth, no signup.

Two endpoints are used:

    GET https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={id}
        → JSON catalog of recent documents for a report type.
           Each entry has a DocID and metadata (FriendlyName, PublishDate,
           Extension, ReportName). Retention is roughly the last 7-31 days
           depending on the product.

    GET https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={DocID}
        → ZIP archive containing the CSV/XML payload.

For historical data older than the retention window, use the authenticated
Public API (see the `ercot_api.py` sibling module, if present).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import httpx
import polars as pl

from surge import store
from surge.scrapers.base import DEFAULT_TIMEOUT, client, get

LIST_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload"


# --- Report registry ------------------------------------------------------
# Maps friendly names to ERCOT reportTypeIds. Sourced from the EMIL catalog at
# https://www.ercot.com/mp/data-products.
REPORTS: dict[str, int] = {
    # Load
    "load_by_weather_zone":    13101,  # NP6-345-CD
    "load_by_forecast_zone":   14836,  # NP6-346-CD
    # Day-ahead
    "dam_hourly_lmp":          12328,  # NP4-183-CD
    "dam_spp":                 12331,  # NP4-190-CD
    "dam_system_lambda":       13113,  # NP4-523-CD
    # Real-time
    "rt_lmp":                  12300,  # NP6-788-CD — LMPs by Resource Node/Zone/Hub
    "rt_spp":                  12301,  # NP6-905-CD — Settlement Point Prices
    "rtd_indicative_lmp":      13073,  # NP6-970-CD
    "sced_system_lambda":      13114,  # NP6-322-CD
    # Wind
    "wind_actual_hourly":      13028,  # NP4-732-CD
    "wind_actual_5min":        13071,  # NP4-733-CD
    "wind_by_region_hourly":   14787,  # NP4-742-CD
    # Solar
    "solar_actual_hourly":     13483,  # NP4-737-CD
    "solar_actual_5min":       13484,  # NP4-738-CD
    "solar_by_region_hourly":  21809,  # NP4-745-CD
}


@dataclass(frozen=True)
class ErcotDoc:
    doc_id: str
    friendly_name: str
    constructed_name: str
    extension: str
    publish_date: datetime
    report_name: str
    report_type_id: int
    content_size: int

    @property
    def download_url(self) -> str:
        return f"{DOWNLOAD_URL}?doclookupId={self.doc_id}"


def _parse_doc(raw: dict[str, Any]) -> ErcotDoc:
    d = raw["Document"]
    return ErcotDoc(
        doc_id=d["DocID"],
        friendly_name=d.get("FriendlyName", ""),
        constructed_name=d.get("ConstructedName", ""),
        extension=d.get("Extension", "").lower(),
        publish_date=datetime.fromisoformat(d["PublishDate"]),
        report_name=d.get("ReportName", ""),
        report_type_id=int(d["ReportTypeID"]),
        content_size=int(d.get("ContentSize", 0)),
    )


def list_docs(
    report: str | int,
    csv_only: bool = True,
) -> list[ErcotDoc]:
    """List documents currently available for a report.

    Args:
        report: friendly name in REPORTS, or a raw integer reportTypeId.
        csv_only: filter to CSV payloads only (ERCOT publishes both csv and xml zips).
    """
    report_type_id = REPORTS[report] if isinstance(report, str) else int(report)
    with client() as c:
        r = get(c, LIST_URL, params={"reportTypeId": report_type_id})
        raw = r.json()
    docs = [_parse_doc(x) for x in raw["ListDocsByRptTypeRes"]["DocumentList"]]
    if csv_only:
        # ERCOT ships csv- and xml-suffixed zips. Keep csv zips only.
        docs = [d for d in docs if "csv" in d.friendly_name.lower() or d.extension == "csv"]
    docs.sort(key=lambda d: d.publish_date, reverse=True)
    return docs


def download_zip(doc: ErcotDoc, client_: httpx.Client | None = None) -> bytes:
    """Download a single document as raw zip bytes."""
    owned = client_ is None
    c = client_ or httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    try:
        r = get(c, doc.download_url)
        return r.content
    finally:
        if owned:
            c.close()


# Zip-bomb defences. ERCOT payloads are typically <1 MB zipped / <10 MB
# decompressed; these caps are well above that but way below anything
# pathological.
_MAX_ZIP_MEMBERS = 64
_MAX_ZIP_DECOMPRESSED = 256 * 1024 * 1024   # 256 MB


def read_zip_as_frames(blob: bytes) -> dict[str, pl.DataFrame]:
    """Unpack a zip of CSVs into {member_name: DataFrame}.

    Rejects archives that have too many members, absolute/traversal paths,
    or a decompressed size over the cap (zip-bomb protection).
    """
    out: dict[str, pl.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = zf.namelist()
        if len(members) > _MAX_ZIP_MEMBERS:
            raise ValueError(f"zip has {len(members)} members (cap {_MAX_ZIP_MEMBERS})")
        total = 0
        for name in members:
            # Reject traversal / absolute members.
            if name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
                raise ValueError(f"unsafe zip member name: {name!r}")
            if not name.lower().endswith(".csv"):
                continue
            info = zf.getinfo(name)
            total += info.file_size
            if total > _MAX_ZIP_DECOMPRESSED:
                raise ValueError("zip decompressed size exceeds cap")
            with zf.open(name) as fh:
                out[name] = pl.read_csv(fh.read(_MAX_ZIP_DECOMPRESSED))
    return out


def fetch_report(
    report: str | int,
    *,
    since: datetime | None = None,
    limit: int | None = None,
    persist: bool = True,
) -> pl.DataFrame:
    """Download every recent document for a report and concatenate the CSVs.

    Args:
        report: friendly name or integer reportTypeId.
        since: only include docs published on/after this UTC-aware timestamp.
        limit: cap on number of docs to fetch (newest first).
        persist: write each doc to the `ercot_{report_name}` dataset,
                 keyed by DocID so repeated calls are idempotent.
    """
    docs = list_docs(report)
    if since is not None:
        docs = [d for d in docs if d.publish_date >= since]
    if limit is not None:
        docs = docs[:limit]

    ds_name = f"ercot_{report}" if isinstance(report, str) else f"ercot_rt_{report}"
    frames: list[pl.DataFrame] = []
    already = store.ingested_keys(ds_name, "ercot-mis") if persist else set()
    with client() as c:
        for d in docs:
            if persist and d.doc_id in already:
                continue
            blob = download_zip(d, client_=c)
            doc_frames: list[pl.DataFrame] = []
            for frame in read_zip_as_frames(blob).values():
                f2 = frame.with_columns(
                    pl.lit(d.doc_id).alias("_doc_id"),
                    pl.lit(d.publish_date).alias("_publish_date"),
                )
                doc_frames.append(f2)
                frames.append(f2)
            if persist and doc_frames:
                merged = pl.concat(doc_frames, how="diagonal_relaxed")
                store.write_through(
                    ds_name, merged,
                    source="ercot-mis", key=d.doc_id, ts_col="_publish_date",
                )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def iter_docs(report: str | int) -> Iterable[ErcotDoc]:
    """Yield docs newest-first. Handy for streaming long catalogs."""
    yield from list_docs(report)
