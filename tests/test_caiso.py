"""CAISO OASIS scraper tests."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import httpx

from surge.scrapers import caiso


def test_fmt_utc() -> None:
    dt = datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)
    assert caiso._fmt(dt) == "20260101T12:30-0000"


def test_iter_windows_chunks_31d() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 4, 1, tzinfo=timezone.utc)
    wins = list(caiso._iter_windows(start, end, days=28))
    assert len(wins) >= 3
    assert wins[0][0] == start
    assert wins[-1][1] == end


def test_unzip_extracts_csvs() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "ts,value\n2024-01-01T00,100\n")
        zf.writestr("readme.txt", "ignored")
    frames = caiso._unzip(buf.getvalue())
    assert len(frames) == 1
    assert frames[0].columns == ["ts", "value"]


def test_fetch_windows_and_sends_correct_params(monkeypatch) -> None:
    calls: list[dict] = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("out.csv", "ts,value\n2024-01-01T00,100\n")
    response = buf.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return httpx.Response(200, content=response,
                              headers={"content-type": "application/zip"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda *a, **kw: real_client(*a, transport=transport, **kw))

    frames = caiso.fetch(
        "PRC_LMP",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 15, tzinfo=timezone.utc),
        node="TH_NP15_GEN-APND",
    )
    assert len(calls) == 2  # 46-day range auto-chunks into two windows
    assert calls[0]["queryname"] == "PRC_LMP"
    assert calls[0]["node"] == "TH_NP15_GEN-APND"
    assert calls[0]["market_run_id"] == "DAM"
    assert calls[0]["resultformat"] == "6"
    assert len(frames) == 2
