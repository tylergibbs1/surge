"""Modal deployment of the Surge forecast API.

Wraps the existing FastAPI app from `surge.api.main` — no route rewrites.

Deploy:
    modal deploy modal_app/app.py

The public URL prints at the end of deploy and also appears under
    https://modal.com/apps/<account>/surge-api

Configure the playground to call it by setting in
`web/playground/.env.local`:
    SURGE_API_URL=https://<name>--<account>.modal.run

Prerequisites (one-time):
    modal secret create eia-api EIA_API_KEY=<your-key>
    modal secret create surge-ledger SURGE_LEDGER_KEY=<random-secret>

Architecture:
    - `data_vol` holds the load_hourly + weather_hourly parquet store.
      Image snapshot at /workspace/seed seeds the volume on first
      container boot; after that the volume is the source of truth.
    - `bootstrap_data` initializes a clean volume with load and weather.
    - `ingest_hour` refreshes EIA load hourly; `ingest_weather` refreshes
      seven-RTO ASOS observations every six hours. Both fail loudly on stale
      or incomplete release-critical coverage.
    - `settle_forecasts` pins eligible EIA outcomes 72 hours after the final
      valid point and commits immutable verification rows every hour.
    - `fastapi_app` reads from the same volume. A lightweight middleware
      reloads the volume at most once per 60 s so fresh rows land within
      a minute of publication, without paying the reload cost per-request.
    - `ledger_publisher_app` is capped at one container. The web bake sends
      one seven-RTO batch request here; middleware reloads before the write and
      commits the Volume before acknowledging it.

"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = os.environ.get("SURGE_MODAL_APP", "surge-api-v02")
# Safe zero-shot default. Override both path and immutable revision only with a
# model trained under the load-v2-core contract.
HF_MODEL_ID = os.environ.get("SURGE_MODEL_PATH", "amazon/chronos-2")
HF_MODEL_REVISION = os.environ.get("SURGE_MODEL_REVISION")
HF_MODEL_FEATURE_SPEC_SHA256 = os.environ.get("SURGE_MODEL_FEATURE_SPEC_SHA256", "")
HF_MODEL_ARTIFACT_SHA256 = os.environ.get("SURGE_MODEL_ARTIFACT_SHA256", "")
ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = Path(os.environ.get("SURGE_SEED_DIR", str(ROOT / "data_snapshot")))
VOLUME_NAME = os.environ.get("SURGE_MODAL_VOLUME", "surge-grid-v02")
RTO_BAS = ("PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP")


def _code_revision() -> str:
    configured = os.environ.get("SURGE_CODE_REVISION")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


CODE_REVISION = _code_revision()

MODEL_ENV = {
    "PYTHONPATH": "/workspace/src",
    "SURGE_DATA_DIR": "/workspace/data",
    # On the volume, not container-local disk. The hourly ingest refetches the
    # last 72 h every hour, which is exactly the cadence that captures EIA's
    # in-place revisions -- but only if the archive survives the container.
    "SURGE_VINTAGE_DIR": "/workspace/data/vintage",
    "SURGE_MODEL_PATH": HF_MODEL_ID,
    "SURGE_MODEL_FEATURE_SPEC_SHA256": HF_MODEL_FEATURE_SPEC_SHA256,
    "SURGE_MODEL_ARTIFACT_SHA256": HF_MODEL_ARTIFACT_SHA256,
    "SURGE_CODE_REVISION": CODE_REVISION,
}
if HF_MODEL_REVISION:
    MODEL_ENV["SURGE_MODEL_REVISION"] = HF_MODEL_REVISION

# Persistent writable store for EIA data. Shared across the FastAPI web
# function and the hourly ingest cron. The snapshot baked into the image
# at /workspace/seed only seeds this volume on first boot; after that,
# the volume is the source of truth and gets hourly updates without any
# redeploy.
data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# EIA API key for the hourly ingest. Create once with:
#   modal secret create eia-api EIA_API_KEY=<key>
eia_secret = modal.Secret.from_name("eia-api")
ledger_secret = modal.Secret.from_name("surge-ledger")

# Image:
#   - Python 3.12 slim
#   - pip install the repo (pulls surge[api] deps: torch, chronos, etc.)
#   - copy the parquet snapshot into /workspace/seed (volume-seed only;
#     the writable mount at /workspace/data shadows it at runtime)
#   - pre-download the model weights at build time so cold start is pure boot
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]>=0.115",
        "uvicorn>=0.30",
        "slowapi>=0.1.9",
        # Known-good H100 stack; keep synchronized with pyproject extras.
        "chronos-forecasting==2.3.1",
        "holidays==0.101",
        "numpy==2.1.2",
        "peft==0.20.0",  # required when SURGE_MODEL_PATH selects a v0.2 LoRA adapter
        "polars==1.43.2",
        "pyarrow==25.0.0",
        "torch==2.8.0",  # retains the safe weights_only=True default
        "transformers==4.57.6",
        "httpx>=0.27",
        "tenacity>=9.0",
        "platformdirs>=4.0",
        "tqdm>=4.66",
        "pydantic>=2.6",
        "zstandard>=0.22",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/workspace/src", copy=True)
    .env(MODEL_ENV)
    .run_commands(
        # Pre-warm the HF cache inside the image so first-request latency is
        # just torch-import + model-load from a local path.
        "python -c 'from surge.api import forecaster; "
        "from surge.model_loader import load_chronos2; "
        "load_chronos2(forecaster.MODEL_PATH, "
        "revision=forecaster.model_load_revision())'",
    )
)

if SEED_DIR.exists():
    image = image.add_local_dir(
        str(SEED_DIR),
        remote_path="/workspace/seed",
        copy=True,
    )
else:
    # A clean tag is deployable without an ignored local snapshot. Readiness
    # remains false until `bootstrap_data` establishes enough context.
    image = image.run_commands("mkdir -p /workspace/seed")

app = modal.App(APP_NAME, image=image)


def _seed_volume_if_empty() -> None:
    """Copy the image-baked snapshot into the volume the first time a
    container mounts it. The sentinel file makes this a no-op once the
    volume is populated — the cron is the source of truth thereafter.
    """
    import shutil

    data = Path("/workspace/data")
    seed = Path("/workspace/seed")
    sentinel = data / ".seeded"

    data.mkdir(parents=True, exist_ok=True)
    if sentinel.exists():
        return
    if seed.exists() and seed.is_dir():
        for item in seed.iterdir():
            dest = data / item.name
            if dest.exists():
                # Partial seed (e.g. a previous crash): leave the existing
                # data in place. The cron will fill any gaps.
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    sentinel.touch()
    data_vol.commit()


def _validate_recent_dataset(
    name: str,
    value_column: str,
    expected_bas: tuple[str, ...],
    *,
    max_age_hours: int = 12,
    min_usable_rows: int = 2_048,
) -> dict[str, object]:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from surge import store

    frame = store.scan(name, dedupe_on=["ts_utc", "ba"]).collect()
    if frame.is_empty():
        raise RuntimeError(f"{name} contains no rows")
    now = datetime.now(tz=UTC)
    required = {"ts_utc", "ba", value_column, "as_of"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"{name} missing required columns: {','.join(missing_columns)}")

    selected = frame.filter(pl.col("ba").is_in(expected_bas))
    observed = set(selected.get_column("ba").unique().to_list())
    missing = sorted(set(expected_bas) - observed)
    if missing:
        raise RuntimeError(f"{name} missing BAs: {','.join(missing)}")

    per_ba: dict[str, dict[str, object]] = {}
    for ba in expected_bas:
        usable = selected.filter(
            (pl.col("ba") == ba)
            & pl.col(value_column).is_not_null()
            & pl.col(value_column).is_finite()
        )
        usable_rows = usable.height
        if usable_rows < min_usable_rows:
            raise RuntimeError(
                f"{name} has only {usable_rows} usable {value_column} rows for {ba}; "
                f"need {min_usable_rows}"
            )
        latest = usable.get_column("ts_utc").max()
        if not isinstance(latest, datetime):
            raise RuntimeError(f"{name} has no usable timestamp for {ba}")
        if latest > now + timedelta(hours=2):
            raise RuntimeError(f"{name} latest timestamp for {ba} is unexpectedly in the future")
        age = now - latest
        if age > timedelta(hours=max_age_hours):
            raise RuntimeError(f"{name} is stale for {ba} by {age}")
        per_ba[ba] = {
            "usable_rows": usable_rows,
            "latest_utc": latest.isoformat(),
            "age_hours": round(age.total_seconds() / 3600.0, 2),
        }

    return {
        "dataset": name,
        "rows": frame.height,
        "required_bas": len(expected_bas),
        "min_usable_rows": min_usable_rows,
        "per_ba": per_ba,
    }


def _probe_all_forecasts(issued_at_utc) -> dict[str, dict[str, object]]:
    """Load the configured model and prove all seven live paths end to end."""
    import math

    import torch

    from surge.api import forecaster
    from surge.model_loader import load_chronos2

    if not forecaster.model_release_safe():
        raise RuntimeError("configured model has no load-v2-core training contract")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    load_kwargs = {"device_map": device, "dtype": dtype}
    load_revision = forecaster.model_load_revision()
    if load_revision is not None:
        load_kwargs["revision"] = load_revision
    pipe = load_chronos2(forecaster.MODEL_PATH, **load_kwargs)

    checks: dict[str, dict[str, object]] = {}
    for ba in RTO_BAS:
        result = forecaster.forecast_ba(
            pipe,
            ba,
            horizon=24,
            issued_at_utc=issued_at_utc,
            feature_cutoff_utc=issued_at_utc,
        )
        points = result["points"]
        if len(points) != 24:
            raise RuntimeError(f"forecast probe returned {len(points)} points for {ba}")
        for point in points:
            values = (
                point["mean_mw"],
                point["p10_mw"],
                point["p50_mw"],
                point["p90_mw"],
            )
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise RuntimeError(f"forecast probe returned invalid values for {ba}")
            if not point["p10_mw"] <= point["p50_mw"] <= point["p90_mw"]:
                raise RuntimeError(f"forecast probe returned crossing quantiles for {ba}")
        checks[ba] = {
            "points": len(points),
            "first_valid_utc": points[0]["valid_at_utc"].isoformat(),
            "last_valid_utc": points[-1]["valid_at_utc"].isoformat(),
            "feature_snapshot_sha256": result["feature_snapshot_sha256"],
        }
    return checks


@app.function(
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={"/workspace/data": data_vol},
    secrets=[eia_secret],
)
def bootstrap_data(days: int = 120, all_bas: bool = False) -> None:
    """Build enough load and observed-weather context in a new v0.2 volume."""
    import json
    from datetime import UTC, datetime, timedelta

    from surge import bas
    from surge.scrapers import eia
    from surge.scrapers.asos import BA_STATIONS, fetch_station

    if days < 90 or days > 366:
        raise ValueError("days must be in 90..366")
    _seed_volume_if_empty()
    data_vol.reload()
    now = datetime.now(tz=UTC)
    start = (now - timedelta(days=days)).date().isoformat()
    end = (now + timedelta(days=1)).date().isoformat()
    codes = tuple(bas.demand_codes()) if all_bas else RTO_BAS
    failures: list[str] = []
    load_rows = 0
    weather_rows = 0
    for code in codes:
        try:
            load_rows += eia.load(code, start, end, force=True).height
        except Exception as exc:
            failures.append(f"load:{code}({type(exc).__name__})")
        station = BA_STATIONS.get(code)
        if station is None:
            failures.append(f"weather:{code}(no-station)")
            continue
        try:
            weather_rows += fetch_station(station, code, start, end, persist=True).height
        except Exception as exc:
            failures.append(f"weather:{code}({type(exc).__name__})")
    data_vol.commit()
    load_quality = _validate_recent_dataset("load_hourly", "load_mw", RTO_BAS)
    weather_quality = _validate_recent_dataset("weather_hourly", "temp_c", RTO_BAS)
    forecast_probes = _probe_all_forecasts(now)
    event = {
        "event": "bootstrap_data",
        "at_utc": now.isoformat(),
        "days": days,
        "bas": len(codes),
        "load_rows": load_rows,
        "weather_rows": weather_rows,
        "failed": failures,
        "load_quality": load_quality,
        "weather_quality": weather_quality,
        "forecast_probes": forecast_probes,
    }
    print(json.dumps(event, sort_keys=True))
    if failures or load_rows == 0 or weather_rows == 0:
        raise RuntimeError("bootstrap did not complete cleanly")


@app.function(
    cpu=1.0,
    memory=1024,
    timeout=900,
    # :05 past every hour — EIA publishes new hours around :00, and :05
    # gives them a few minutes to land. Timezone is UTC (Modal default).
    schedule=modal.Cron("5 * * * *"),
    volumes={"/workspace/data": data_vol},
    secrets=[eia_secret],
)
def ingest_hour() -> None:
    """Pull the last 72 h of EIA-930 demand for every BA; commit volume.

    Why 72 h: EIA backfills recent hours in-place for a few hours after
    first publication (small BAs lag, RTO corrections). Re-fetching the
    last 3 days every hour catches every revision with negligible cost.
    Duplicates collapse on read via `store.scan(dedupe_on=["ts_utc","ba"])`.
    """
    import json
    from datetime import UTC, datetime, timedelta

    from surge import bas
    from surge.scrapers import eia

    _seed_volume_if_empty()
    # Pick up any writes from prior cron runs in other containers.
    data_vol.reload()

    now = datetime.now(tz=UTC)
    start = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    rows = 0
    failed: list[str] = []
    for ba in bas.demand_codes():
        try:
            df = eia.load(ba=ba, start=start, end=end, force=True)
            rows += df.height
        except Exception as e:
            failed.append(f"{ba}({type(e).__name__})")

    data_vol.commit()
    quality = _validate_recent_dataset("load_hourly", "load_mw", RTO_BAS)
    event = {
        "event": "ingest_hour",
        "at_utc": now.isoformat(timespec="seconds"),
        "rows": rows,
        "failed_count": len(failed),
        "failed": failed,
        "quality": quality,
    }
    print(json.dumps(event, sort_keys=True))
    failed_rtos = [item for item in failed if item.split("(", 1)[0] in RTO_BAS]
    if rows == 0 or failed_rtos or len(failed) > 5:
        raise RuntimeError("EIA ingest failed its coverage gate")


@app.function(
    cpu=1.0,
    memory=1024,
    timeout=900,
    schedule=modal.Cron("35 */6 * * *"),
    volumes={"/workspace/data": data_vol},
)
def ingest_weather() -> None:
    """Refresh observed ASOS temperature for the seven v0.2 RTO regions."""
    import json
    from datetime import UTC, datetime, timedelta

    from surge import store
    from surge.scrapers.asos import BA_STATIONS, fetch_station

    _seed_volume_if_empty()
    data_vol.reload()
    now = datetime.now(tz=UTC)
    start = (now - timedelta(days=7)).date().isoformat()
    end = (now + timedelta(days=1)).date().isoformat()
    rows = 0
    failed: list[str] = []
    for code in RTO_BAS:
        station = BA_STATIONS.get(code)
        if station is None:
            failed.append(f"{code}(no-station)")
            continue
        try:
            frame = fetch_station(station, code, start, end, persist=False)
            store.append("weather_hourly", frame)
            rows += frame.height
        except Exception as exc:
            failed.append(f"{code}({type(exc).__name__})")
    data_vol.commit()
    quality = _validate_recent_dataset("weather_hourly", "temp_c", RTO_BAS)
    event = {
        "event": "ingest_weather",
        "at_utc": now.isoformat(timespec="seconds"),
        "rows": rows,
        "failed_count": len(failed),
        "failed": failed,
        "quality": quality,
    }
    print(json.dumps(event, sort_keys=True))
    if rows == 0 or failed:
        raise RuntimeError("ASOS ingest failed its seven-RTO coverage gate")


@app.function(
    cpu=1.0,
    memory=1024,
    timeout=900,
    schedule=modal.Cron("45 * * * *"),
    volumes={"/workspace/data": data_vol},
)
def settle_forecasts() -> None:
    """Pin mature +72 h EIA outcomes for every published live issuance."""
    import json
    from datetime import UTC, datetime, timedelta

    from surge import ledger, verification

    _seed_volume_if_empty()
    data_vol.reload()
    now = datetime.now(tz=UTC)
    records = ledger.list_forecasts(
        mode=ledger.ForecastMode.LIVE,
        limit=1_000,
    )
    eligible = 0
    committed_points = 0
    failures: list[str] = []
    for record in records:
        if now < record.last_valid_at_utc + timedelta(hours=verification.MATURITY_HOURS):
            continue
        eligible += 1
        try:
            committed_points += verification.verify_forecast(
                record.issuance_id,
                verified_at_utc=now,
            )
        except Exception as exc:
            failures.append(f"{record.issuance_id}({type(exc).__name__})")

    if committed_points:
        data_vol.commit()
    event = {
        "event": "settle_forecasts",
        "at_utc": now.isoformat(timespec="seconds"),
        "eligible_issuances": eligible,
        "committed_points": committed_points,
        "failed_count": len(failures),
        "failed": failures,
        "outcome_policy_version": verification.OUTCOME_POLICY_VERSION,
        "maturity_hours": verification.MATURITY_HOURS,
    }
    print(json.dumps(event, sort_keys=True))
    if failures:
        raise RuntimeError("forecast settlement failed")


def _volume_backed_api(*, allow_ledger_mutations: bool):
    """Build one API container with read refresh and durable ledger writes."""
    import time

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    from surge.api.main import app as api_app

    _seed_volume_if_empty()

    # Container-local TTL on the reload so a burst of requests only
    # triggers one I/O round-trip to Modal's volume service.
    state = {"last_reload_s": 0.0}
    RELOAD_TTL_S = 60.0

    class VolumeReloadMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            path = request.url.path.rstrip("/")
            batch_publication = request.method == "POST" and path == "/ledger/runs/bake"
            ledger_mutation = (
                batch_publication
                or (
                    request.method == "POST"
                    and path.startswith("/ledger/runs/")
                    and path.endswith("/publish")
                )
                or (
                    request.method == "GET"
                    and path.startswith("/forecast/")
                    and request.query_params.get("commit", "").lower()
                    in {"1", "true", "on", "yes"}
                )
            )
            if allow_ledger_mutations and not batch_publication:
                return JSONResponse(
                    {"detail": "publisher exposes only POST /ledger/runs/bake"},
                    status_code=404,
                )
            if ledger_mutation and not allow_ledger_mutations:
                return JSONResponse(
                    {"detail": "ledger writes require the serialized publisher endpoint"},
                    status_code=404,
                )
            now_s = time.monotonic()
            if ledger_mutation:
                try:
                    data_vol.reload()
                except Exception as exc:
                    print(f"[vol mutation reload] {exc}")
                    return JSONResponse(
                        {"detail": "ledger volume reload failed"}, status_code=503
                    )
                state["last_reload_s"] = now_s
            elif now_s - state["last_reload_s"] > RELOAD_TTL_S:
                try:
                    data_vol.reload()
                    state["last_reload_s"] = now_s
                except Exception as exc:
                    # Read-only traffic may use the last committed snapshot for
                    # at most one retry interval when the Volume service hiccups.
                    print(f"[vol reload] {exc}")

            if not ledger_mutation:
                return await call_next(request)

            try:
                response = await call_next(request)
            except BaseException:
                # Preserve any immutable staged records written before a failed
                # batch so an exact retry can resume safely.
                data_vol.commit()
                raise
            try:
                # The response is not returned to Vercel until Modal confirms
                # these ledger files and the complete-run marker are durable.
                data_vol.commit()
            except Exception as exc:
                print(f"[vol mutation commit] {exc}")
                return JSONResponse(
                    {"detail": "ledger volume commit failed"}, status_code=503
                )
            response.headers["X-Surge-Volume-Committed"] = "true"
            return response

    api_app.add_middleware(VolumeReloadMiddleware)
    return api_app


@app.function(
    cpu=2.0,
    memory=4096,
    # HN-launch config: keep one warm to kill cold-start on spike traffic,
    # allow scale-out to 20 for burst, reclaim idle workers after 10 min.
    scaledown_window=600,
    min_containers=1,
    max_containers=20,
    timeout=60,
    volumes={"/workspace/data": data_vol},
    secrets=[ledger_secret],
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """Serve read traffic and backward-compatible one-BA inference."""
    return _volume_backed_api(allow_ledger_mutations=False)


@app.function(
    cpu=2.0,
    memory=4096,
    scaledown_window=600,
    max_containers=1,
    timeout=600,
    volumes={"/workspace/data": data_vol},
    secrets=[ledger_secret],
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def ledger_publisher_app():
    """Serialized endpoint for the complete seven-RTO publication batch."""
    return _volume_backed_api(allow_ledger_mutations=True)
