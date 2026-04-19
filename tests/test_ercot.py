"""ERCOT public file-drop scraper tests."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime

import httpx

from surge.scrapers import ercot


def _sample_catalog() -> dict:
    return {
        "ListDocsByRptTypeRes": {
            "DocumentList": [
                {"Document": {
                    "DocID": "111",
                    "FriendlyName": "LMPSROSNODENP6788_20260101_000000_csv",
                    "ConstructedName": "cdr.12300.xxx.csv.zip",
                    "Extension": "zip",
                    "PublishDate": "2026-01-01T00:00:00-05:00",
                    "ReportName": "LMPs by Resource Node",
                    "ReportTypeID": "12300",
                    "ContentSize": "1024",
                }},
                {"Document": {
                    "DocID": "222",
                    "FriendlyName": "LMPSROSNODENP6788_20260101_000500_xml",
                    "ConstructedName": "cdr.12300.xxx.xml.zip",
                    "Extension": "zip",
                    "PublishDate": "2026-01-01T00:05:00-05:00",
                    "ReportName": "LMPs by Resource Node",
                    "ReportTypeID": "12300",
                    "ContentSize": "2048",
                }},
            ]
        }
    }


def _make_zip(name: str, csv_body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv_body)
    return buf.getvalue()


def test_list_docs_filters_to_csv_only(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "reportTypeId=12300" in str(request.url)
        return httpx.Response(200, content=json.dumps(_sample_catalog()).encode(),
                              headers={"content-type": "application/json"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: real_client(*a, transport=transport, **kw),
    )

    docs = ercot.list_docs("rt_lmp", csv_only=True)
    assert [d.doc_id for d in docs] == ["111"]
    assert docs[0].extension == "zip"
    assert docs[0].report_type_id == 12300


def test_list_docs_accepts_raw_integer_id(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "reportTypeId=99999" in str(request.url)
        return httpx.Response(200, content=json.dumps(
            {"ListDocsByRptTypeRes": {"DocumentList": []}}
        ).encode())

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda *a, **kw: real_client(*a, transport=transport, **kw))
    assert ercot.list_docs(99999) == []


def test_read_zip_as_frames_decodes_csv_members() -> None:
    blob = _make_zip("hub_lmp.csv", "ts,node,lmp\n2026-01-01,HB_NORTH,42.5\n")
    out = ercot.read_zip_as_frames(blob)
    assert "hub_lmp.csv" in out
    df = out["hub_lmp.csv"]
    assert df.columns == ["ts", "node", "lmp"]
    assert df.height == 1
    assert df["lmp"][0] == 42.5


def test_fetch_report_applies_since_and_limit(monkeypatch) -> None:
    csv = "ts,value\n2026-01-01T00,1.0\n"
    zblob = _make_zip("payload.csv", csv)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/misapp/servlets/IceDocListJsonWS":
            return httpx.Response(200, content=json.dumps(_sample_catalog()).encode())
        if request.url.path == "/misdownload/servlets/mirDownload":
            return httpx.Response(200, content=zblob)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda *a, **kw: real_client(*a, transport=transport, **kw))

    since = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    df = ercot.fetch_report("rt_lmp", since=since, limit=1)
    assert df.height == 1
    assert "_doc_id" in df.columns
    assert df["_doc_id"][0] == "111"
