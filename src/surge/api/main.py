"""Surge forecast API.

Endpoints:
    GET  /                        metadata
    GET  /health                  readiness + data freshness
    GET  /bas                     supported BA list
    GET  /forecast/{ba}           one-BA probabilistic forecast
    GET  /forecast                all-BAs forecast in one response
    GET  /forecast/stream         NDJSON stream (one BA per line as it's produced)

Run:
    uvicorn surge.api.main:app --host 0.0.0.0 --port 8000

Env:
    SURGE_DATA_DIR       default ~/.surge/data
    SURGE_MODEL_PATH     default <repo>/models/chronos2_full_v2
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from surge import __version__
from surge import bas as _bas
from surge.api import forecaster, live_load
from surge.api.schemas import (
    SUPPORTED_BAS,
    ActualPoint,
    ActualsResponse,
    BAListResponse,
    BAMeta,
    CurrentLoadResponse,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
)

log = logging.getLogger("surge.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the Chronos-2 pipeline once at startup, release at shutdown."""
    import torch
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log.info("loading %s on %s / %s", forecaster.MODEL_PATH, device, dtype)
    # When MODEL_PATH is an HF repo id (not a local path) we also pin a
    # specific revision — see forecaster.MODEL_REVISION for rationale.
    load_kwargs: dict = {"device_map": device, "torch_dtype": dtype}
    if "/" in forecaster.MODEL_PATH and not forecaster.MODEL_PATH.startswith("/"):
        load_kwargs["revision"] = forecaster.MODEL_REVISION
    pipe = BaseChronosPipeline.from_pretrained(forecaster.MODEL_PATH, **load_kwargs)
    app.state.pipe = pipe
    app.state.model_name = forecaster.MODEL_NAME
    app.state.device = device
    log.info("model loaded")
    try:
        yield
    finally:
        app.state.pipe = None
        log.info("model released")


_CORS_ORIGINS_ENV = os.environ.get("SURGE_ALLOWED_ORIGINS", "")
_DEFAULT_ORIGINS = [
    "https://surgeforecast.com",
    "https://www.surgeforecast.com",
    # Keep the Vercel-assigned aliases whitelisted so preview deploys and
    # deployment-URL probes still work without a secondary config.
    "https://surge-omega-nine.vercel.app",
    "https://surge-grayhaven.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]
ALLOWED_ORIGINS = (
    [o.strip() for o in _CORS_ORIGINS_ENV.split(",") if o.strip()]
    if _CORS_ORIGINS_ENV
    else _DEFAULT_ORIGINS
)

# Per-IP rate limit. slowapi uses the X-Forwarded-For header when present,
# which Vercel sets; direct-to-Modal calls fall back to the socket peer.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(
    title="Surge — open forecasts for the US power grid",
    version=__version__,
    description=(
        f"Open, probabilistic day-ahead load forecasts for {len(SUPPORTED_BAS)} "
        "US balancing authorities (every EIA-930 BA with a demand series). "
        "Model: Chronos-2 fine-tuned on 7 years of EIA-930 load, ASOS hourly "
        "temperatures, and US calendar features. "
        "For research and reference use only — not for trading or bankable decisions."
    ),
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_pipe(request: Request) -> Any:
    pipe = getattr(request.app.state, "pipe", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return pipe


PipeDep = Annotated[Any, Depends(get_pipe)]


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "surge",
        "version": __version__,
        "model": forecaster.MODEL_NAME,
        "docs_url": "/docs",
        "supported_bas": list(SUPPORTED_BAS),
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    pipe = getattr(request.app.state, "pipe", None)
    return HealthResponse(
        status="ok" if pipe is not None else "loading",
        model_loaded=pipe is not None,
        model_name=getattr(request.app.state, "model_name", None) if pipe else None,
        data_end_utc=forecaster.data_end_utc(),
    )


@app.get("/current-load", response_model=CurrentLoadResponse, tags=["meta"])
@limiter.limit("60/minute")
async def current_load(
    request: Request,  # slowapi pulls the rate-limit key from this
    hours: int = 24,
) -> CurrentLoadResponse:
    """Rolling aggregate US demand for the last `hours` hours.

    Sums `load_hourly` across every BA that's reported for each hour.
    Not a forecast — this is the actuals series the playground's live
    hero widget polls to feel tied to the real grid.

    Async so the parquet scan runs under asyncio.to_thread and doesn't
    block the event loop; backed by a per-hour TTL cache so repeated
    hits inside the same clock hour (Vercel edge-cache misses, direct
    callers) don't rescan the store each time.
    """
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=422, detail="hours must be in 1..168")
    try:
        payload = await live_load.aggregate_load(hours=hours)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="no load data available") from None
    return CurrentLoadResponse(**payload)


@app.get("/actuals/{ba}", response_model=ActualsResponse, tags=["forecast"])
@limiter.limit("60/minute")
def actuals_one(
    ba: str,
    request: Request,  # slowapi pulls the rate-limit key from this
    response: Response,
    hours: int = 48,
) -> ActualsResponse:
    """Last `hours` of realized load_mw for one BA.

    Powers the playground chart's historical context line so readers can
    see the observed ramp running into the forecast — the forecast-start
    break is meaningless without the actuals behind it.

    Cheap: one polars scan, filtered + tailed. No model inference.
    """
    import polars as pl

    from surge import store

    ba = ba.upper()
    if ba not in SUPPORTED_BAS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported BA '{ba}'. Supported: {', '.join(SUPPORTED_BAS)}",
        )
    if hours < 1 or hours > 720:
        raise HTTPException(status_code=422, detail="hours must be in 1..720")

    try:
        df = (
            store.scan("load_hourly")
            .filter(pl.col("ba") == ba)
            .filter(pl.col("load_mw").is_not_null())
            .filter(pl.col("load_mw") > 0)
            .filter(pl.col("load_mw") < 200_000)
            .sort("ts_utc")
            .tail(hours)
            .collect()
        )
    except Exception:
        log.exception("actuals scan failed for %s", ba)
        raise HTTPException(status_code=500, detail="actuals scan failed") from None

    if df.is_empty():
        raise HTTPException(status_code=503, detail=f"no load data for {ba}")

    points = [
        ActualPoint(ts_utc=row["ts_utc"], load_mw=float(row["load_mw"]))
        for row in df.iter_rows(named=True)
    ]
    # 5-minute cache — same as /forecast. Hourly-granularity data doesn't
    # change inside a cache window so the edge can absorb repeat hits.
    response.headers["Cache-Control"] = "public, max-age=300, s-maxage=300"
    return ActualsResponse(
        ba=ba,
        as_of_utc=datetime.now(tz=UTC),
        hours=len(points),
        points=points,
    )


@app.get("/bas", response_model=BAListResponse, tags=["meta"])
def bas(include_gen_only: bool = False) -> BAListResponse:
    """List forecastable BAs (default) or the full EIA-930 registry.

    Set `include_gen_only=true` to also return the 14 generator- or
    transmission-only BAs that have no demand series (shown on the map
    for completeness but not forecasted).
    """
    codes = _bas.all_codes() if include_gen_only else _bas.demand_codes()
    metadata = [
        BAMeta(
            code=b.code,
            name=b.name,
            interconnect=b.interconnect,
            utc_offset=b.utc_offset,
            station=b.station,
            has_demand=b.has_demand,
            is_rto=b.is_rto,
            centroid=b.centroid,
            peak_mw=b.peak_mw,
        )
        for c in codes
        for b in (_bas.get(c),)
    ]
    return BAListResponse(bas=codes, count=len(codes), metadata=metadata)


def _build_response(ba: str, horizon: int, result: dict, model_name: str) -> ForecastResponse:
    return ForecastResponse(
        ba=ba,
        model=model_name,
        as_of_utc=datetime.now(tz=UTC),
        context_start_utc=result["context_start_utc"],
        context_end_utc=result["context_end_utc"],
        horizon=horizon,
        points=[ForecastPoint(**p) for p in result["points"]],
    )


# NOTE: declare fixed paths before path-parameter routes — FastAPI matches
# in declaration order, so `/forecast/stream` must come before `/forecast/{ba}`.
@app.get("/forecast", response_model=list[ForecastResponse], tags=["forecast"])
@limiter.limit("10/minute")  # 7x more work than /forecast/{ba}, so stricter
def forecast_all(
    pipe: PipeDep,
    request: Request,
    horizon: int = 24,
) -> list[ForecastResponse]:
    out: list[ForecastResponse] = []
    for ba in SUPPORTED_BAS:
        try:
            result = forecaster.forecast_ba(pipe, ba, horizon=horizon)
            out.append(_build_response(ba, horizon, result, request.app.state.model_name))
        except Exception as e:
            log.warning("skipping %s: %s", ba, e)
    return out


@app.get("/forecast/stream", tags=["forecast"])
@limiter.limit("10/minute")
def forecast_stream(
    pipe: PipeDep,
    request: Request,
    horizon: int = 24,
) -> StreamingResponse:
    """NDJSON stream — one line per BA as its forecast finishes.

    Example:
        curl http://localhost:8000/forecast/stream
    """
    model_name = request.app.state.model_name

    def gen():
        for ba in SUPPORTED_BAS:
            try:
                result = forecaster.forecast_ba(pipe, ba, horizon=horizon)
                resp = _build_response(ba, horizon, result, model_name)
                yield resp.model_dump_json() + "\n"
            except Exception as e:
                yield json.dumps({"ba": ba, "error": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _with_cache_headers(r: ForecastResponse, response) -> ForecastResponse:
    # 5-minute edge + browser cache. Forecasts refresh hourly via cron, so
    # serving a ≤5-min-stale reply across all readers is fine; dramatically
    # reduces inference load when the link is on HN.
    response.headers["Cache-Control"] = "public, max-age=300, s-maxage=300"
    return r


@app.get("/forecast/{ba}", response_model=ForecastResponse, tags=["forecast"])
@limiter.limit("60/minute")
def forecast_one(
    ba: str,
    pipe: PipeDep,
    request: Request,
    response: Response,
    horizon: int = 24,
) -> ForecastResponse:
    ba = ba.upper()
    if ba not in SUPPORTED_BAS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported BA '{ba}'. Supported: {', '.join(SUPPORTED_BAS)}",
        )
    if horizon < 1 or horizon > 168:
        raise HTTPException(status_code=422, detail="horizon must be in 1..168")

    try:
        result = forecaster.forecast_ba(pipe, ba, horizon=horizon)
    except ValueError:
        # Deliberately generic — the upstream error may mention paths,
        # column names, or library internals.
        raise HTTPException(status_code=400, detail="invalid forecast request") from None
    except Exception:  # pragma: no cover
        log.exception("forecast failed for %s", ba)
        raise HTTPException(status_code=500, detail="forecast failed") from None
    return _with_cache_headers(
        _build_response(ba, horizon, result, request.app.state.model_name),
        response,
    )
