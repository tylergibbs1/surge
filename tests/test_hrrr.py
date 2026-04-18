"""NOAA HRRR scraper tests (unit-only; no AWS hits)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from surge.scrapers import hrrr


def test_urls_have_correct_shape() -> None:
    dt = datetime(2026, 1, 2, 6, tzinfo=timezone.utc)
    assert hrrr.grib_url(dt, 3) == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
        "hrrr.20260102/conus/hrrr.t06z.wrfsfcf03.grib2"
    )
    assert hrrr.idx_url(dt, 3).endswith(".grib2.idx")


SAMPLE_IDX = (
    "1:0:d=2026010206:TMP:2 m above ground:1 hour fcst:\n"
    "2:524288:d=2026010206:DPT:2 m above ground:1 hour fcst:\n"
    "3:1048576:d=2026010206:UGRD:10 m above ground:1 hour fcst:\n"
)


def test_parse_idx_yields_records() -> None:
    records = hrrr.parse_idx(SAMPLE_IDX)
    assert len(records) == 3
    assert records[0].variable == "TMP"
    assert records[0].level == "2 m above ground"
    assert records[0].byte_start == 0
    assert records[2].variable == "UGRD"


def test_find_matches_variable_and_level() -> None:
    records = hrrr.parse_idx(SAMPLE_IDX)
    rec = hrrr.find(records, variable="DPT", level="2 m above ground")
    assert rec.record_no == 2


def test_download_slice_issues_range_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".idx"):
            return httpx.Response(200, text=SAMPLE_IDX)
        captured["range"] = request.headers.get("Range")
        return httpx.Response(206, content=b"\x00" * 128)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda *a, **kw: real_client(*a, transport=transport, **kw))

    dt = datetime(2026, 1, 2, 6, tzinfo=timezone.utc)
    blob = hrrr.download_slice(dt, 1, variable="TMP", level="2 m above ground")
    assert len(blob) == 128
    assert captured["range"] == "bytes=0-524287"
