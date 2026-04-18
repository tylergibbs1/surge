"""NOAA NCDC / NCEI scraper tests."""

from __future__ import annotations

import gzip
import json

import httpx

from surge.scrapers import ncdc


INDEX_HTML = """
<html><body>
<a href="StormEvents_details-ftp_v1.0_d2024_c20250203.csv.gz">f1</a>
<a href="StormEvents_details-ftp_v1.0_d2024_c20240301.csv.gz">f2</a>
<a href="StormEvents_details-ftp_v1.0_d2023_c20240915.csv.gz">f3</a>
<a href="StormEvents_locations-ftp_v1.0_d2024_c20250203.csv.gz">ignored</a>
</body></html>
"""


def test_storm_events_index_parses_and_sorts(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=INDEX_HTML)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda *a, **kw: real_client(*a, transport=transport, **kw))

    rows = ncdc.storm_events_index()
    # Two 2024 files + one 2023; newest revision of 2024 first.
    assert rows[0] == (2024, "20250203", "StormEvents_details-ftp_v1.0_d2024_c20250203.csv.gz")
    assert rows[1][0] == 2024
    assert rows[2][0] == 2023


def test_ghcnd_paginates_and_shapes_frame(monkeypatch) -> None:
    monkeypatch.setenv("NCDC_TOKEN", "fake")
    monkeypatch.setenv("SURGE_DATA_DIR", str(monkeypatch.undo))  # noqa — just avoid writes

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        offset = int(request.url.params["offset"])
        # Page 1: 2 rows; page 2: 1 row; total = 3.
        if offset == 1:
            results = [
                {"date": "2024-01-01T00:00:00", "datatype": "TMAX", "value": 3.2},
                {"date": "2024-01-02T00:00:00", "datatype": "TMAX", "value": 5.1},
            ]
        else:
            results = [{"date": "2024-01-03T00:00:00", "datatype": "TMAX", "value": 4.4}]
        payload = {
            "metadata": {"resultset": {"count": 3}},
            "results": results,
        }
        assert request.headers["token"] == "fake"
        return httpx.Response(200, content=json.dumps(payload).encode(),
                              headers={"content-type": "application/json"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda *a, **kw: real_client(*a, transport=transport, **kw))

    df = ncdc.ghcnd("GHCND:USW00023174", "2024-01-01", "2024-01-03", persist=False)
    assert df.height == 3
    assert "ts_utc" in df.columns
    assert calls["n"] == 2
