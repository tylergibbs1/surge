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
    SURGE_MODEL_PATH     default amazon/chronos-2
    SURGE_MODEL_REVISION pinned for the default model; otherwise required
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from surge import __version__, ledger, store, verification
from surge import bas as _bas
from surge.api import forecaster, ledger_api, live_load
from surge.api.schemas import (
    SUPPORTED_BAS,
    ActualPoint,
    ActualsResponse,
    BAListResponse,
    BAMeta,
    CurrentLoadResponse,
    ForecastResponse,
    HealthResponse,
    LedgerBakeResponse,
    LedgerForecastListResponse,
    LedgerRunResponse,
    LedgerScoreboardResponse,
    LiveResponse,
    ScoreSummaryResponse,
)

log = logging.getLogger("surge.api")

RTO_BAS = ledger.REQUIRED_RTO_BAS
MAX_RELEASE_SLOT_AGE = timedelta(minutes=90)
DATA_WARN_AFTER = timedelta(hours=float(os.environ.get("SURGE_DATA_WARN_HOURS", "6")))
DATA_CRITICAL_AFTER = timedelta(hours=float(os.environ.get("SURGE_DATA_CRITICAL_HOURS", "12")))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the Chronos-2 pipeline once at startup, release at shutdown."""
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log.info("loading %s on %s / %s", forecaster.MODEL_PATH, device, dtype)
    load_kwargs: dict = {"device_map": device, "dtype": dtype}
    load_revision = forecaster.model_load_revision()
    if load_revision is not None:
        load_kwargs["revision"] = load_revision
    from surge.model_loader import load_chronos2

    pipe = load_chronos2(forecaster.MODEL_PATH, **load_kwargs)
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
    title="Surge Grid — open, auditable forecasts for the US power grid",
    version=__version__,
    description=(
        f"Open, probabilistic day-ahead load forecasts for {len(SUPPORTED_BAS)} "
        "US balancing authorities (every EIA-930 BA with a demand series). "
        "Every committed forecast records its input cutoff, exact model and code "
        "revision, feature specification, quantiles, and immutable issuance ID. "
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
        "name": "surge-grid",
        "version": __version__,
        "model": forecaster.MODEL_NAME,
        "docs_url": "/docs",
        "supported_bas": list(SUPPORTED_BAS),
    }


def _health_payload(request: Request) -> HealthResponse:
    now = datetime.now(tz=UTC)
    pipe = getattr(request.app.state, "pipe", None)
    source_watermarks = forecaster.data_watermarks_utc()
    available_watermarks = [value for value in source_watermarks.values() if value is not None]
    data_end = min(available_watermarks) if available_watermarks else None
    source_ages: dict[str, float | None] = {
        source: (
            None
            if watermark is None
            else max(now - watermark, timedelta(0)).total_seconds() / 3600.0
        )
        for source, watermark in source_watermarks.items()
    }
    reasons: list[str] = []
    data_age_hours: float | None = None
    if pipe is None:
        status = "loading"
        reasons.append("model is not loaded")
    elif any(value is None for value in source_watermarks.values()):
        status = "error"
        missing = [source for source, value in source_watermarks.items() if value is None]
        reasons.append(f"required data unavailable: {', '.join(missing)}")
    else:
        assert data_end is not None
        age = max(now - data_end, timedelta(0))
        data_age_hours = age.total_seconds() / 3600.0
        stalest_source = max(
            source_ages,
            key=lambda source: source_ages[source] if source_ages[source] is not None else -1,
        )
        if age > DATA_CRITICAL_AFTER:
            status = "error"
            reasons.append(f"{stalest_source} is critically stale ({data_age_hours:.1f} hours old)")
        elif age > DATA_WARN_AFTER:
            status = "degraded"
            reasons.append(f"{stalest_source} is delayed ({data_age_hours:.1f} hours old)")
        else:
            status = "ok"

    try:
        recent = ledger.list_forecasts(mode=ledger.ForecastMode.LIVE, limit=1)
        latest_issuance = recent[0].issued_at_utc if recent else None
    except Exception:
        latest_issuance = None
        reasons.append("forecast ledger could not be read")
        if status == "ok":
            status = "degraded"

    return HealthResponse(
        checked_at_utc=now,
        status=status,
        model_loaded=pipe is not None,
        model_name=getattr(request.app.state, "model_name", None) if pipe else None,
        model_revision=forecaster.MODEL_REVISION if pipe else None,
        data_end_utc=data_end,
        data_age_hours=data_age_hours,
        source_watermarks_utc=source_watermarks,
        source_age_hours=source_ages,
        latest_issuance_utc=latest_issuance,
        reasons=reasons,
    )


@app.get("/live", response_model=LiveResponse, tags=["meta"])
def live() -> LiveResponse:
    """Process liveness only; this does not claim model or data readiness."""
    return LiveResponse(checked_at_utc=datetime.now(tz=UTC))


@app.get("/ready", response_model=HealthResponse, tags=["meta"])
def ready(request: Request, response: Response) -> HealthResponse:
    payload = _health_payload(request)
    if payload.status in {"error", "loading"}:
        response.status_code = 503
    return payload


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request, response: Response) -> HealthResponse:
    payload = _health_payload(request)
    if payload.status in {"error", "loading"}:
        response.status_code = 503
    return payload


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
        # Dedupe at the store layer — `append` is not idempotent and
        # duplicate rows crash Recharts with "two children with the same
        # key" on the frontend. Filtered to one BA, so ts_utc alone is
        # the business key here.
        df = (
            store.scan("load_hourly", dedupe_on=["ts_utc", "ba"])
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


@app.get(
    "/ledger/scoreboard",
    response_model=LedgerScoreboardResponse,
    tags=["ledger"],
)
def ledger_scoreboard() -> LedgerScoreboardResponse:
    """Latest atomically published live run and settled score for each major RTO."""
    return ledger_api.build_scoreboard(list(RTO_BAS))


@app.get(
    "/ledger/issuances",
    response_model=LedgerForecastListResponse,
    tags=["ledger"],
)
@limiter.limit("30/minute")
def ledger_issuances(
    request: Request,
    ba: str | None = None,
    mode: str | None = "live",
    limit: int = Query(default=100, ge=1, le=100),
) -> LedgerForecastListResponse:
    try:
        parsed_mode = ledger.ForecastMode(mode) if mode is not None else None
        records = ledger.list_forecasts(ba=ba, mode=parsed_mode, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    forecasts = [ledger_api.summary_from_record(record) for record in records]
    return LedgerForecastListResponse(forecasts=forecasts, count=len(forecasts))


@app.post(
    "/ledger/runs/{run_id}/publish",
    response_model=LedgerRunResponse,
    tags=["ledger"],
)
def publish_ledger_run(run_id: str, request: Request) -> LedgerRunResponse:
    """Idempotently publish a staged run after all seven RTOs pass validation."""
    _authorize_ledger_commit(request)
    try:
        marker = ledger.publish_run_if_complete(run_id)
    except (store.ImmutableConflictError, ValueError):
        raise HTTPException(
            status_code=409,
            detail="staged run failed complete-run validation",
        ) from None
    if marker is None:
        raise HTTPException(
            status_code=409,
            detail="staged run does not contain all seven required RTO issuances",
        )
    return ledger_api.run_response(marker)


@app.get(
    "/ledger/issuances/{issuance_id}",
    response_model=ForecastResponse,
    tags=["ledger"],
)
def ledger_issuance(issuance_id: str) -> ForecastResponse:
    try:
        record = ledger.get_forecast(issuance_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="issuance not found") from None
    return ledger_api.response_from_record(record)


@app.get(
    "/ledger/issuances/{issuance_id}/score",
    response_model=ScoreSummaryResponse,
    tags=["ledger"],
)
def ledger_issuance_score(issuance_id: str) -> ScoreSummaryResponse:
    try:
        score = verification.score_forecast(issuance_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="issuance not found") from None
    except ValueError:
        raise HTTPException(status_code=404, detail="issuance has no settled score") from None
    return ScoreSummaryResponse(**score.model_dump())


def _scheduled_slot(issued_at_utc: datetime, requested: datetime | None) -> datetime:
    if requested is None:
        return issued_at_utc.replace(minute=0, second=0, microsecond=0)
    if requested.tzinfo is None or requested.utcoffset() is None:
        raise ValueError("scheduled_for_utc must be timezone-aware")
    scheduled = requested.astimezone(UTC)
    if scheduled > issued_at_utc + timedelta(minutes=5):
        raise ValueError("scheduled_for_utc cannot be in the future")
    return scheduled


def _record_from_result(
    ba: str,
    result: dict[str, Any],
    model_name: str,
    *,
    scheduled_for_utc: datetime | None,
) -> ledger.ForecastRecord:
    issued_at = result["issued_at_utc"]
    provenance = result.get("provenance", {})
    warnings = list(result.get("warnings", []))
    if model_name == "surge-fm-v3":
        warnings.append(
            "surge-fm-v3 has legacy training provenance; this live issuance is "
            "point-in-time safe but must not be blended with its oracle-covariate backtest"
        )
    code_revision = str(provenance.get("code_revision") or forecaster.CODE_REVISION)
    if code_revision == "unknown":
        warnings.append("code revision was not configured")
    point_kind = result.get("point_estimate_kind", "median")
    if point_kind == "p50":
        point_kind = "median"

    return ledger.ForecastRecord(
        ba=ba,
        mode=ledger.ForecastMode.LIVE,
        scheduled_for_utc=_scheduled_slot(issued_at, scheduled_for_utc),
        feature_cutoff_utc=result["feature_cutoff_utc"],
        issued_at_utc=issued_at,
        context_start_utc=result["context_start_utc"],
        context_end_utc=result["context_end_utc"],
        model_name=model_name,
        model_revision=str(provenance.get("model_revision") or forecaster.MODEL_REVISION),
        model_artifact_sha256=provenance.get("model_artifact_sha256"),
        code_revision=code_revision,
        feature_spec_version=result["feature_spec_version"],
        feature_spec_sha256=result["feature_spec_sha256"],
        feature_snapshot_sha256=result["feature_snapshot_sha256"],
        availability_mode=ledger.AvailabilityMode(result["availability_mode"]),
        point_estimate_kind=ledger.PointEstimateKind(point_kind),
        mase_scale_24=float(result["mase_scale_24"]),
        warnings=tuple(dict.fromkeys(warnings)),
        points=tuple(
            ledger.ForecastPointRecord(
                valid_at_utc=point.get("valid_at_utc", point["ts_utc"]),
                mean_mw=point["mean_mw"],
                p10_mw=point["p10_mw"],
                p50_mw=point.get("p50_mw", point["median_mw"]),
                p90_mw=point["p90_mw"],
                future_temp_c=point.get("future_temp_c"),
                future_temp_vintage_id=point.get("future_temp_vintage_id"),
            )
            for point in result["points"]
        ),
    )


def _build_response(
    ba: str,
    result: dict[str, Any],
    model_name: str,
    *,
    scheduled_for_utc: datetime | None = None,
) -> tuple[ledger.ForecastRecord, ForecastResponse]:
    record = _record_from_result(
        ba,
        result,
        model_name,
        scheduled_for_utc=scheduled_for_utc,
    )
    return record, ledger_api.response_from_record(record, committed=False)


def _authorize_ledger_commit(request: Request) -> None:
    expected = os.environ.get("SURGE_LEDGER_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="ledger commits are not configured")
    provided = request.headers.get("x-surge-ledger-key", "")
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid ledger commit key")


def _existing_scheduled_forecast(
    ba: str,
    scheduled_for_utc: datetime,
    horizon: int,
    model_name: str,
) -> ledger.ForecastRecord | None:
    """Return a committed retry target before paying for model inference."""
    for record in ledger.list_forecasts(
        ba=ba,
        mode=ledger.ForecastMode.LIVE,
        limit=100,
        include_unpublished=True,
    ):
        if (
            record.scheduled_for_utc == scheduled_for_utc
            and record.model_name == model_name
            and record.model_revision == forecaster.MODEL_REVISION
            and record.model_artifact_sha256 == forecaster.model_artifact_identity()
            and record.code_revision == forecaster.CODE_REVISION
            and record.feature_spec_version == forecaster.LOAD_V2_CORE.version
            and record.feature_spec_sha256 == forecaster.LOAD_V2_CORE.sha256
            and record.horizon_hours == horizon
        ):
            # Re-running the exact staged write also repairs the narrow crash
            # window between the seventh issuance and its complete-run marker.
            ledger.commit_forecast(record)
            return record
    return None


def _release_slot(issued_at_utc: datetime, requested: datetime | None) -> datetime:
    scheduled = _scheduled_slot(issued_at_utc, requested)
    if issued_at_utc - scheduled > MAX_RELEASE_SLOT_AGE:
        raise ValueError("scheduled_for_utc is too old for a live release")
    return scheduled


@app.post(
    "/ledger/runs/bake",
    response_model=LedgerBakeResponse,
    tags=["ledger"],
)
def bake_ledger_run(
    pipe: PipeDep,
    request: Request,
    response: Response,
    horizon: int = 168,
    scheduled_for_utc: datetime | None = None,
) -> LedgerBakeResponse:
    """Commit and publish one complete seven-RTO run in this request.

    The Modal deployment exposes this route through a publisher function with
    ``max_containers=1``. Keeping all seven writes and the complete-run marker
    in one request removes cross-container filesystem locking from the release
    correctness boundary.
    """
    _authorize_ledger_commit(request)
    if horizon < 1 or horizon > 168:
        raise HTTPException(status_code=422, detail="horizon must be in 1..168")
    if not forecaster.model_release_safe():
        raise HTTPException(
            status_code=503,
            detail="configured model has no load-v2-core training contract",
        )

    issued_at = datetime.now(tz=UTC)
    try:
        scheduled_slot = _release_slot(issued_at, scheduled_for_utc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    try:
        for ba in RTO_BAS:
            existing = _existing_scheduled_forecast(
                ba,
                scheduled_slot,
                horizon,
                request.app.state.model_name,
            )
            if existing is not None:
                continue
            result = forecaster.forecast_ba(
                pipe,
                ba,
                horizon=horizon,
                issued_at_utc=issued_at,
                feature_cutoff_utc=issued_at,
            )
            record = _record_from_result(
                ba,
                result,
                request.app.state.model_name,
                scheduled_for_utc=scheduled_slot,
            )
            if record.code_revision == "unknown":
                raise HTTPException(
                    status_code=503,
                    detail="ledger commits require SURGE_CODE_REVISION",
                )
            ledger.commit_forecast(record)

        run_id = ledger.deterministic_run_id(ledger.ForecastMode.LIVE, scheduled_slot)
        marker = ledger.publish_run_if_complete(run_id)
        if marker is None:
            raise HTTPException(
                status_code=409,
                detail="staged run does not contain all seven required RTO issuances",
            )
        records = ledger.forecasts_for_run(marker.run_id)
    except HTTPException:
        raise
    except store.ImmutableConflictError:
        raise HTTPException(
            status_code=409,
            detail="scheduled run already exists with different content",
        ) from None
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid forecast request") from None
    except Exception:  # pragma: no cover
        log.exception("seven-RTO ledger bake failed")
        raise HTTPException(status_code=500, detail="forecast run bake failed") from None

    forecasts = [ledger_api.response_from_record(record) for record in records]
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Surge-Run-ID"] = marker.run_id
    return LedgerBakeResponse(
        run=ledger_api.run_response(marker),
        forecasts=forecasts,
        committed_regions=len(forecasts),
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
    issued_at = datetime.now(tz=UTC)
    for ba in SUPPORTED_BAS:
        try:
            result = forecaster.forecast_ba(
                pipe,
                ba,
                horizon=horizon,
                issued_at_utc=issued_at,
                feature_cutoff_utc=issued_at,
            )
            _, response = _build_response(ba, result, request.app.state.model_name)
            out.append(response)
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
    issued_at = datetime.now(tz=UTC)

    def gen():
        for ba in SUPPORTED_BAS:
            try:
                result = forecaster.forecast_ba(
                    pipe,
                    ba,
                    horizon=horizon,
                    issued_at_utc=issued_at,
                    feature_cutoff_utc=issued_at,
                )
                _, resp = _build_response(ba, result, model_name)
                yield resp.model_dump_json() + "\n"
            except Exception as e:
                yield json.dumps({"ba": ba, "error": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _with_cache_headers(
    r: ForecastResponse,
    response: Response,
    *,
    committed: bool = False,
) -> ForecastResponse:
    # 5-minute edge + browser cache. Forecasts refresh hourly via cron, so
    # serving a ≤5-min-stale reply across all readers is fine; dramatically
    # reduces inference load when the link is on HN.
    response.headers["Cache-Control"] = (
        "no-store" if committed else "public, max-age=300, s-maxage=300"
    )
    if committed:
        response.headers["X-Surge-Issuance-ID"] = r.issuance_id
    return r


@app.get("/forecast/{ba}", response_model=ForecastResponse, tags=["forecast"])
@limiter.limit("60/minute")
def forecast_one(
    ba: str,
    pipe: PipeDep,
    request: Request,
    response: Response,
    horizon: int = 24,
    commit: bool = False,
    scheduled_for_utc: datetime | None = None,
) -> ForecastResponse:
    ba = ba.upper()
    if ba not in SUPPORTED_BAS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported BA '{ba}'. Supported: {', '.join(SUPPORTED_BAS)}",
        )
    if horizon < 1 or horizon > 168:
        raise HTTPException(status_code=422, detail="horizon must be in 1..168")

    if commit:
        _authorize_ledger_commit(request)
        if not forecaster.model_release_safe():
            raise HTTPException(
                status_code=503,
                detail="configured model has no load-v2-core training contract",
            )
    issued_at = datetime.now(tz=UTC)
    scheduled_slot = _scheduled_slot(issued_at, scheduled_for_utc)
    if commit:
        existing = _existing_scheduled_forecast(
            ba,
            scheduled_slot,
            horizon,
            request.app.state.model_name,
        )
        if existing is not None:
            return _with_cache_headers(
                ledger_api.response_from_record(existing),
                response,
                committed=True,
            )
    try:
        result = forecaster.forecast_ba(
            pipe,
            ba,
            horizon=horizon,
            issued_at_utc=issued_at,
            feature_cutoff_utc=issued_at,
        )
        record, payload = _build_response(
            ba,
            result,
            request.app.state.model_name,
            scheduled_for_utc=scheduled_slot,
        )
        if commit:
            if record.code_revision == "unknown":
                raise HTTPException(
                    status_code=503,
                    detail="ledger commits require SURGE_CODE_REVISION",
                )
            ledger.commit_forecast(record)
            payload = ledger_api.response_from_record(record)
    except HTTPException:
        raise
    except store.ImmutableConflictError:
        raise HTTPException(
            status_code=409,
            detail="scheduled issuance already exists with different content",
        ) from None
    except ValueError:
        # Deliberately generic — the upstream error may mention paths,
        # column names, or library internals.
        raise HTTPException(status_code=400, detail="invalid forecast request") from None
    except Exception:  # pragma: no cover
        log.exception("forecast failed for %s", ba)
        raise HTTPException(status_code=500, detail="forecast failed") from None
    return _with_cache_headers(
        payload,
        response,
        committed=commit,
    )
