"""PJM scraper tests."""

from __future__ import annotations

import httpx
import pytest

from surge.scrapers import pjm


def test_date_range_uses_feed_date_column() -> None:
    out = pjm.date_range("lmp_da_hourly", "01/01/2024", "01/02/2024")
    assert out == {"datetime_beginning_ept": "01/01/2024 to 01/02/2024"}


def test_date_range_unknown_feed_raises() -> None:
    with pytest.raises(KeyError):
        pjm.date_range("not_a_feed", "a", "b")


def test_page_size_capped_at_pjm_max(monkeypatch) -> None:
    monkeypatch.setenv("PJM_SUBSCRIPTION_KEY", "fake")
    with pytest.raises(ValueError, match="50000"):
        pjm.fetch_feed("lmp_da_hourly", page_size=60_000)


def test_missing_subscription_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("PJM_SUBSCRIPTION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PJM_SUBSCRIPTION_KEY"):
        pjm.fetch_feed("lmp_da_hourly")


def test_fetch_feed_paginates_until_short_page(monkeypatch) -> None:
    monkeypatch.setenv("PJM_SUBSCRIPTION_KEY", "fake")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startRow"])
        rc = int(request.url.params["rowCount"])
        calls.append(start)
        # two full pages, then a short final page
        if start == 1:
            items = [{"value": i} for i in range(rc)]
        elif start == rc + 1:
            items = [{"value": i + 1000} for i in range(rc)]
        else:
            items = [{"value": 9000}]
        return httpx.Response(200, json={"items": items, "links": []})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **kw: real_client(*a, transport=transport, **kw),
    )

    df = pjm.fetch_feed(
        "lmp_da_hourly",
        page_size=3,
        params=pjm.date_range("lmp_da_hourly", "01/01/2024", "01/02/2024"),
    )
    assert df.height == 7
    assert calls == [1, 4, 7]
