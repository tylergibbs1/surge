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
    # Secret holding the EIA Open Data key so the ingest cron can fetch.
    modal secret create eia-api EIA_API_KEY=<your-key>

Architecture:
    - `data_vol` holds the load_hourly + weather_hourly parquet store.
      Image snapshot at /workspace/seed seeds the volume on first
      container boot; after that the volume is the source of truth.
    - `ingest_hour` runs :05 past every hour, pulls EIA's last 72 h for
      every BA (catches in-place revisions), commits the volume.
    - `fastapi_app` reads from the same volume. A lightweight middleware
      reloads the volume at most once per 60 s so fresh rows land within
      a minute of publication, without paying the reload cost per-request.

Cost sketch (as of 2026):
    CPU cold start: ~4 s to import torch, another ~10 s to download
        surge-fm-v3 from HF on first call; subsequent requests in the same
        warm window (600 s idle): ~300 ms.
    Hourly cron: ~1 min of CPU-seconds per run. Free $30/mo credit
        covers everything comfortably.
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "surge-api"
# 53-BA generalist. Override with SURGE_MODEL_PATH=Tylerbry1/surge-fm-v2
# in the container env to serve the older 7-RTO specialist.
HF_MODEL_ID = "Tylerbry1/surge-fm-v3"
ROOT = Path(__file__).resolve().parents[1]

# Persistent writable store for EIA data. Shared across the FastAPI web
# function and the hourly ingest cron. The snapshot baked into the image
# at /workspace/seed only seeds this volume on first boot; after that,
# the volume is the source of truth and gets hourly updates without any
# redeploy.
data_vol = modal.Volume.from_name("surge-load-hourly-v1", create_if_missing=True)

# EIA API key for the hourly ingest. Create once with:
#   modal secret create eia-api EIA_API_KEY=<key>
eia_secret = modal.Secret.from_name("eia-api")

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
        "chronos-forecasting",
        "torch>=2.6",        # weights_only=True default; blocks pickle-RCE
        "holidays>=0.55",
        "polars>=1.15",
        "pyarrow>=17.0",
        "httpx>=0.27",
        "tenacity>=9.0",
        "platformdirs>=4.0",
        "tqdm>=4.66",
        "pydantic>=2.6",
        "zstandard>=0.22",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/workspace/src", copy=True)
    .add_local_dir(str(ROOT / "data_snapshot"), remote_path="/workspace/seed", copy=True)
    .env({
        "PYTHONPATH": "/workspace/src",
        "SURGE_DATA_DIR": "/workspace/data",
        "SURGE_MODEL_PATH": HF_MODEL_ID,
    })
    .run_commands(
        # Pre-warm the HF cache inside the image so first-request latency is
        # just torch-import + model-load from a local path.
        f"python -c 'from chronos import BaseChronosPipeline; "
        f"BaseChronosPipeline.from_pretrained(\"{HF_MODEL_ID}\")'",
    )
)

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
    print(
        f"[ingest_hour] {now.isoformat(timespec='seconds')} "
        f"rows={rows} failed={len(failed)} "
        + (f"codes={','.join(failed[:10])}" if failed else "")
    )


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
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """Serve the Surge FastAPI app as-is, with periodic volume reload.

    The ingest cron writes to the same volume; without an explicit reload
    the warm container serves its snapshot from cold-start forever. The
    middleware below reloads at most once per 60 s so new rows surface
    within a minute of cron commit, not at the next container churn.
    """
    import time

    from starlette.middleware.base import BaseHTTPMiddleware

    from surge.api.main import app as fastapi_app

    _seed_volume_if_empty()

    # Container-local TTL on the reload so a burst of requests only
    # triggers one I/O round-trip to Modal's volume service.
    state = {"last_reload_s": 0.0}
    RELOAD_TTL_S = 60.0

    class VolumeReloadMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            now_s = time.monotonic()
            if now_s - state["last_reload_s"] > RELOAD_TTL_S:
                state["last_reload_s"] = now_s
                try:
                    data_vol.reload()
                except Exception as e:
                    # Don't fail requests on reload hiccups — worst case
                    # the reader serves up-to-60-s-stale rows.
                    print(f"[vol reload] {e}")
            return await call_next(request)

    fastapi_app.add_middleware(VolumeReloadMiddleware)
    return fastapi_app
