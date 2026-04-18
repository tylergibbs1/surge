"""Shared HTTP helpers for scrapers."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
DEFAULT_HEADERS = {
    "User-Agent": "surge/0.0.1 (+https://github.com/surge-grid/surge)",
    "Accept": "application/json,text/csv,application/zip,*/*",
}


def client(**kwargs) -> httpx.Client:
    return httpx.Client(
        timeout=kwargs.pop("timeout", DEFAULT_TIMEOUT),
        headers={**DEFAULT_HEADERS, **kwargs.pop("headers", {})},
        follow_redirects=True,
        **kwargs,
    )


_transient = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(_transient),
    reraise=True,
)
def get(c: httpx.Client, url: str, **kwargs) -> httpx.Response:
    r = c.get(url, **kwargs)
    if r.status_code in (429, 500, 502, 503, 504):
        raise httpx.RemoteProtocolError(f"transient status {r.status_code}", request=r.request)
    r.raise_for_status()
    return r
