"""Modal deployment of the Surge forecast API.

Wraps the existing FastAPI app from `surge.api.main` — no route rewrites.

Deploy:
    modal deploy modal_app/app.py

The public URL prints at the end of deploy and also appears under
    https://modal.com/apps/<account>/surge-api

Configure the playground to call it by setting in
`web/playground/.env.local`:
    SURGE_API_URL=https://<name>--<account>.modal.run

Cost sketch (as of 2026):
    CPU cold start: ~4 s to import torch, another ~10 s to download
        surge-fm-v3 from HF on first call; subsequent requests in the same
        warm window (300 s idle): ~300 ms.
    Free $30/mo credit covers ~2.5 M CPU-seconds which is way more than
        Surge's expected traffic.
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "surge-api"
# 53-BA generalist. Override with SURGE_MODEL_PATH=Tylerbry1/surge-fm-v2
# in the container env to serve the older 7-RTO specialist.
HF_MODEL_ID = "Tylerbry1/surge-fm-v3"
ROOT = Path(__file__).resolve().parents[1]

# Image:
#   - Python 3.12 slim
#   - pip install the repo (pulls surge[api] deps: torch, chronos, etc.)
#   - copy the parquet snapshot into /workspace/data
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
    .add_local_dir(str(ROOT / "data_snapshot"), remote_path="/workspace/data", copy=True)
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


@app.function(
    cpu=2.0,
    memory=4096,
    # HN-launch config: keep one warm to kill cold-start on spike traffic,
    # allow scale-out to 20 for burst, reclaim idle workers after 10 min.
    scaledown_window=600,
    min_containers=1,
    max_containers=20,
    timeout=60,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """Serve the Surge FastAPI app as-is."""
    from surge.api.main import app as fastapi_app
    return fastapi_app
