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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from surge import __version__
from surge.api import forecaster
from surge.api.schemas import (
    SUPPORTED_BAS,
    BAListResponse,
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
    pipe = BaseChronosPipeline.from_pretrained(
        forecaster.MODEL_PATH, device_map=device, torch_dtype=dtype,
    )
    app.state.pipe = pipe
    app.state.model_name = forecaster.MODEL_NAME
    app.state.device = device
    log.info("model loaded")
    try:
        yield
    finally:
        app.state.pipe = None
        log.info("model released")


app = FastAPI(
    title="Surge — open forecasts for the US power grid",
    version=__version__,
    description=(
        "Open, probabilistic day-ahead load forecasts for 7 major US balancing "
        "authorities. Model: Chronos-2 fine-tuned on 7 years of EIA-930 load, "
        "ASOS hourly temperatures, and US calendar features. Test MASE 0.45 "
        "across 7 BAs — beats seasonal-naive-24 by 56.6%. "
        "For research and reference use only — not for trading or bankable decisions."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/bas", response_model=BAListResponse, tags=["meta"])
def bas() -> BAListResponse:
    return BAListResponse(bas=list(SUPPORTED_BAS), count=len(SUPPORTED_BAS))


def _build_response(ba: str, horizon: int, result: dict, model_name: str) -> ForecastResponse:
    return ForecastResponse(
        ba=ba,
        model=model_name,
        as_of_utc=datetime.now(tz=timezone.utc),
        context_start_utc=result["context_start_utc"],
        context_end_utc=result["context_end_utc"],
        horizon=horizon,
        points=[ForecastPoint(**p) for p in result["points"]],
    )


# NOTE: declare fixed paths before path-parameter routes — FastAPI matches
# in declaration order, so `/forecast/stream` must come before `/forecast/{ba}`.
@app.get("/forecast", response_model=list[ForecastResponse], tags=["forecast"])
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


@app.get("/forecast/{ba}", response_model=ForecastResponse, tags=["forecast"])
def forecast_one(
    ba: str,
    pipe: PipeDep,
    request: Request,
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # pragma: no cover
        log.exception("forecast failed for %s", ba)
        raise HTTPException(status_code=500, detail=f"forecast failed: {e}") from e
    return _build_response(ba, horizon, result, request.app.state.model_name)
