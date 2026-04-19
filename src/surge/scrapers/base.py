"""Shared HTTP helpers for scrapers."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
DEFAULT_HEADERS = {
    "User-Agent": "surge/0.0.1 (+https://github.com/surge-grid/surge)",
    "Accept": "application/json,text/csv,application/zip,*/*",
}

# Patterns that look like secrets in URLs / exception bodies. Matched
# against query-string values; matching key is replaced with "REDACTED".
_SECRET_PARAMS = frozenset({
    "api_key", "apikey", "token", "subscription-key", "access_token",
    "password", "secret",
})


def _scrub_url(url: str) -> str:
    """Return `url` with any secret-looking query params masked."""
    try:
        s = urlsplit(url)
    except ValueError:
        return "<unparseable>"
    if not s.query:
        return url
    pairs = []
    for kv in s.query.split("&"):
        if "=" not in kv:
            pairs.append(kv)
            continue
        k, v = kv.split("=", 1)
        if k.lower() in _SECRET_PARAMS:
            pairs.append(f"{k}=REDACTED")
        else:
            pairs.append(kv)
    return urlunsplit((s.scheme, s.netloc, s.path, "&".join(pairs), s.fragment))


_SECRET_RE = re.compile(
    r"(?i)(api_key|apikey|token|subscription-key|access_token|password)=([^&\s\"']+)"
)


def _scrub_text(text: str) -> str:
    return _SECRET_RE.sub(r"\1=REDACTED", text)


def client(**kwargs) -> httpx.Client:
    # follow_redirects is False by default — a scraper that wants redirect
    # following must opt in explicitly. Prevents SSRF via redirect to
    # internal addresses if user input ever reaches a scraper.
    return httpx.Client(
        timeout=kwargs.pop("timeout", DEFAULT_TIMEOUT),
        headers={**DEFAULT_HEADERS, **kwargs.pop("headers", {})},
        follow_redirects=kwargs.pop("follow_redirects", False),
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
    try:
        r = c.get(url, **kwargs)
    except httpx.HTTPError as e:
        # httpx exceptions expose .request.url which contains credentials
        # when passed as query params (EIA, NREL). Re-raise with a scrubbed
        # message so tracebacks in Modal/Sentry/stdout don't leak keys.
        raise type(e)(_scrub_text(str(e))) from None
    if r.status_code in (429, 500, 502, 503, 504):
        raise httpx.RemoteProtocolError(
            f"transient status {r.status_code} on {_scrub_url(url)}",
            request=r.request,
        )
    r.raise_for_status()
    return r
