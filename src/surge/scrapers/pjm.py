"""PJM Data Miner 2 public API client.

Base URL: https://api.pjm.com/api/v1/{feed}
Auth:     Ocp-Apim-Subscription-Key header (free key via apiportal.pjm.com)

Registration: set PJM_SUBSCRIPTION_KEY in the environment.

The API is an Azure APIM front end to PJM's public feeds. Each feed supports:
    GET /api/v1/{feed}/metadata        — schema description, no auth required on some feeds
    GET /api/v1/{feed}                 — paginated data search
    GET /api/v1/{feed}?download=true   — streaming download, ignores page-size cap

Pagination (from PJM's API guide):
    startRow   1-based index of the first record to return.
    rowCount   max 50 000, required when any other parameter is set.
    sort, order  order by field, asc/desc.

Date filters use PJM's per-feed date column names (e.g. datetime_beginning_ept,
datetime_beginning_utc, pricing_date). The filter VALUE supports either a
keyword ("Today", "CurrentMonth", "LastHour", "5MinutesAgo") or a literal range
"MM/DD/YYYY HH:MM to MM/DD/YYYY HH:MM". Max range is 366 days.

Response envelope:
    {items: [...], links: [...], totalRows (as X-TotalRows header when download=true)}

We flatten `items` into a polars frame and re-page via startRow/rowCount.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import polars as pl

from surge.scrapers.base import DEFAULT_HEADERS, DEFAULT_TIMEOUT, get

BASE_URL = "https://api.pjm.com/api/v1"
MAX_ROW_COUNT = 50_000


# --- Feed registry ---------------------------------------------------------
# Maps friendly names → (feed slug, primary date column). The date column is
# the field you pass as a filter param alongside a date value/range.
FEEDS: dict[str, tuple[str, str]] = {
    # Load
    "load_estimated":          ("hrl_load_estimated",           "datetime_beginning_ept"),
    "load_metered":            ("hrl_load_metered",             "datetime_beginning_ept"),
    "load_prelim":             ("hrl_load_prelim",              "datetime_beginning_ept"),
    "load_instantaneous":      ("inst_load",                    "datetime_beginning_ept"),
    # Load forecasts
    "load_forecast_7day":      ("load_frcstd_7_day",            "forecast_datetime_beginning_ept"),
    "load_forecast_5min":      ("very_short_load_frcst",        "evaluated_at_ept"),
    "load_forecast_historical":("load_frcstd_hist",             "evaluated_at_ept"),
    # LMPs
    "lmp_da_hourly":           ("da_hrl_lmps",                  "datetime_beginning_ept"),
    "lmp_rt_hourly":           ("rt_hrl_lmps",                  "datetime_beginning_ept"),
    "lmp_rt_5min":             ("rt_fivemin_hrl_lmps",          "datetime_beginning_ept"),
    "lmp_rt_unverified_5min":  ("rt_unverified_fivemin_lmps",   "datetime_beginning_ept"),
    "lmp_rt_unverified_hourly":("rt_unverified_hrl_lmps",       "datetime_beginning_ept"),
    # Generation
    "gen_by_fuel_type":        ("gen_by_fuel",                  "datetime_beginning_ept"),
    "solar_gen":               ("solar_gen",                    "datetime_beginning_ept"),
    "wind_gen":                ("wind_gen",                     "datetime_beginning_ept"),
    "solar_forecast_hourly":   ("solar_forecast",               "datetime_beginning_ept"),
    "wind_forecast_hourly":    ("hourly_wind_power_forecast",   "datetime_beginning_ept"),
    # Interchange / system
    "sched_interchange":       ("act_sch_interchange",          "datetime_beginning_ept"),
    "area_control_error":      ("area_control_error",           "datetime_beginning_ept"),
    "pricing_nodes":           ("pnode",                        "effective_date_ept"),
}


def _api_key() -> str:
    key = os.environ.get("PJM_SUBSCRIPTION_KEY")
    if not key:
        raise RuntimeError(
            "PJM_SUBSCRIPTION_KEY is not set. Register at https://apiportal.pjm.com"
            " and subscribe to the 'Data Miner 2 Non-Member API' product."
        )
    return key


def _headers() -> dict[str, str]:
    return {**DEFAULT_HEADERS, "Ocp-Apim-Subscription-Key": _api_key()}


def _flatten_items(items: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(items) if items else pl.DataFrame()


def fetch_feed(
    feed: str,
    params: dict[str, Any] | None = None,
    page_size: int = 10_000,
    max_pages: int | None = None,
) -> pl.DataFrame:
    """Fetch a PJM feed, paginated via startRow/rowCount.

    Args:
        feed: friendly name in FEEDS, or raw feed slug (e.g. "da_hrl_lmps").
        params: extra query parameters — typically includes a date filter like
            `{"datetime_beginning_ept": "01/01/2024 to 02/01/2024"}`.
        page_size: rows per request, capped at 50 000.
        max_pages: optional cap for testing.
    """
    if feed in FEEDS:
        slug, _ = FEEDS[feed]
    else:
        slug = feed
    if page_size > MAX_ROW_COUNT:
        raise ValueError(f"page_size exceeds PJM max of {MAX_ROW_COUNT}")

    url = f"{BASE_URL}/{slug}"
    q: dict[str, Any] = {"startRow": 1, "rowCount": page_size}
    if params:
        q.update(params)

    frames: list[pl.DataFrame] = []
    with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=_headers(), follow_redirects=True) as c:
        page = 0
        while True:
            r = get(c, url, params=q)
            payload = r.json()
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            frames.append(_flatten_items(items))
            page += 1
            # PJM does not return totalRows in the JSON body by default; we
            # stop when a page returns fewer rows than requested.
            if len(items) < q["rowCount"] or (max_pages and page >= max_pages):
                break
            q["startRow"] += q["rowCount"]

    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def date_range(feed: str, start: str, end: str) -> dict[str, str]:
    """Build the primary date filter for a registered feed.

    Dates should match PJM's format: "MM/DD/YYYY" or "MM/DD/YYYY HH:MM".
    """
    if feed not in FEEDS:
        raise KeyError(f"unknown feed '{feed}'. known: {sorted(FEEDS)}")
    _, col = FEEDS[feed]
    return {col: f"{start} to {end}"}


def feed_metadata(feed: str) -> dict[str, Any]:
    """GET /{feed}/metadata — schema description."""
    slug = FEEDS[feed][0] if feed in FEEDS else feed
    with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=_headers(), follow_redirects=True) as c:
        r = get(c, f"{BASE_URL}/{slug}/metadata")
        return r.json()
