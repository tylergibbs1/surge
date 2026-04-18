"""RunPod serverless handler for Surge forecasts.

Deployment (GitHub direct):
    RunPod → Serverless → Deploy from GitHub → point at this repo
    Command:   python -u runpod/handler.py
    GPU:       any (CPU also works; Chronos-2 is ~120M params)
    Env vars:  SURGE_MODEL_HF_ID  (default: Tylerbry1/surge-fm-v2)
               SURGE_DATA_URL     (optional; where to pull live load + weather from)

Request shape:
    { "input": { "ba": "PJM", "horizon": 24 } }

Response shape (same as our FastAPI):
    { "ba": "PJM", "model": "...", "as_of_utc": "...", "horizon": 24,
      "units": "MW", "points": [ {ts_utc, median_mw, p10_mw, p90_mw}, ...] }
"""
from __future__ import annotations

import os
from pathlib import Path

import runpod  # type: ignore[import-not-found]
import torch

from chronos import BaseChronosPipeline

# The pipeline is loaded once at worker startup (module-import time) and
# reused across requests for the lifetime of that worker.
MODEL_ID = os.environ.get("SURGE_MODEL_HF_ID", "Tylerbry1/surge-fm-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"[surge] loading {MODEL_ID} on {DEVICE}", flush=True)
_PIPE = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=DTYPE)
print("[surge] model loaded", flush=True)

# The forecaster module reads data from the parquet store; on a serverless
# worker we ship a recent snapshot into /workspace/data (or download from
# HF datasets at startup). Import lazily so cold-start isn't blocked by
# data-file resolution.
from surge.api import forecaster  # noqa: E402


def handler(job: dict) -> dict:
    """Entry point for every RunPod invocation."""
    inp = (job or {}).get("input", {}) or {}
    ba = str(inp.get("ba", "PJM")).upper()
    horizon = int(inp.get("horizon", 24))
    if horizon < 1 or horizon > 168:
        return {"error": f"horizon {horizon} out of range (1..168)"}

    try:
        result = forecaster.forecast_ba(_PIPE, ba, horizon=horizon)
    except ValueError as e:
        return {"error": str(e), "ba": ba}
    except Exception as e:
        return {"error": f"internal: {e.__class__.__name__}: {e}", "ba": ba}

    return {
        "ba": ba,
        "model": forecaster.MODEL_NAME,
        "horizon": horizon,
        "units": "MW",
        "context_start_utc": result["context_start_utc"].isoformat(),
        "context_end_utc":   result["context_end_utc"].isoformat(),
        "points": [
            {
                "ts_utc": p["ts_utc"].isoformat(),
                "median_mw": p["median_mw"],
                "p10_mw":   p["p10_mw"],
                "p90_mw":   p["p90_mw"],
            }
            for p in result["points"]
        ],
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
