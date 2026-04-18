"""Provision the Surge serverless endpoint on RunPod via the Python SDK.

Idempotent:
    - Creates or reuses the template named `surge-fm-v2-template`
    - Creates or reuses the endpoint named `surge-fm-v2`
    - Prints the endpoint ID + invocation URL

Usage:
    RUNPOD_API_KEY=... python scripts/deploy_runpod.py

Image source:
    ghcr.io/tylergibbs1/surge:latest — built and pushed by
    .github/workflows/docker.yml on every push to main.
"""
from __future__ import annotations

import os
import sys

import runpod


IMAGE = os.environ.get("SURGE_IMAGE", "ghcr.io/tylergibbs1/surge:latest")
TEMPLATE_NAME = "surge-fm-v2-template"
ENDPOINT_NAME = "surge-fm-v2"
# GPU selection: start with a CPU worker. Chronos-2 is ~120M params so
# inference on CPU is ~1-2s per request — fine for free-tier traffic.
# If you want GPU, pass --gpu to switch. Allowed values map to RunPod's
# GPU IDs (e.g. "AMPERE_16" = RTX A4000, "ADA_24" = RTX 4090).
GPU_IDS = os.environ.get("SURGE_GPU_IDS", None)  # None = CPU-only


def _find_by_name(items: list[dict], name: str) -> dict | None:
    for it in items:
        if it.get("name") == name:
            return it
    return None


def main() -> None:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: set RUNPOD_API_KEY in env", file=sys.stderr)
        print("  Get one at https://runpod.io/console/user/settings", file=sys.stderr)
        sys.exit(1)
    runpod.api_key = key

    # ---- Template --------------------------------------------------
    templates = runpod.get_templates() or []
    existing_tpl = _find_by_name(templates, TEMPLATE_NAME)
    if existing_tpl:
        tpl_id = existing_tpl["id"]
        print(f"[template] reusing '{TEMPLATE_NAME}' id={tpl_id}")
    else:
        print(f"[template] creating '{TEMPLATE_NAME}' from {IMAGE}")
        tpl = runpod.create_template(
            name=TEMPLATE_NAME,
            image_name=IMAGE,
            docker_start_cmd="python -u runpod/handler.py",
            container_disk_in_gb=15,
            is_serverless=True,
            env={
                "SURGE_MODEL_HF_ID": "Tylerbry1/surge-fm-v2",
                "PYTHONUNBUFFERED": "1",
            },
        )
        tpl_id = tpl["id"]
        print(f"[template] created id={tpl_id}")

    # ---- Endpoint --------------------------------------------------
    endpoints = runpod.get_endpoints() or []
    existing_ep = _find_by_name(endpoints, ENDPOINT_NAME)

    ep_kwargs = dict(
        name=ENDPOINT_NAME,
        template_id=tpl_id,
        workers_min=0,
        workers_max=2,
        idle_timeout=30,
        flashboot=True,
        scaler_type="QUEUE_DELAY",
        scaler_value=4,
    )
    if GPU_IDS:
        ep_kwargs["gpu_ids"] = GPU_IDS

    if existing_ep:
        ep_id = existing_ep["id"]
        print(f"[endpoint] reusing '{ENDPOINT_NAME}' id={ep_id}")
        runpod.update_endpoint_template(endpoint_id=ep_id, template_id=tpl_id)
        print("[endpoint] pinned to latest template")
    else:
        print(f"[endpoint] creating '{ENDPOINT_NAME}'")
        ep = runpod.create_endpoint(**ep_kwargs)
        ep_id = ep["id"]
        print(f"[endpoint] created id={ep_id}")

    url = f"https://api.runpod.ai/v2/{ep_id}"
    print()
    print("───────────────────────────────────────")
    print(f"  endpoint id   : {ep_id}")
    print(f"  runsync       : {url}/runsync")
    print(f"  run (async)   : {url}/run")
    print(f"  status        : {url}/status/<job-id>")
    print("───────────────────────────────────────")
    print()
    print("Test:")
    print(f'  curl -X POST "{url}/runsync" \\')
    print('    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \\')
    print('    -H "content-type: application/json" \\')
    print('    -d \'{"input":{"ba":"PJM","horizon":24}}\'')


if __name__ == "__main__":
    main()
