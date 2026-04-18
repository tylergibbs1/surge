# RunPod serverless deployment

Deploy the Surge forecast model as a RunPod serverless endpoint so the
Next.js playground can call it on-demand without running a 24/7 server.

## Prerequisites

1. **Model published to Hugging Face** — the handler loads from
   `Tylerbry1/surge-fm-v2` by default. If you haven't pushed, either
   (a) bake the model into the Docker image locally, or (b) point
   `SURGE_MODEL_HF_ID` at any other public Chronos-2 checkpoint.
2. **RunPod account + API key**.

## One-click deploy from GitHub

1. RunPod → Serverless → Deploy → "Deploy from GitHub"
2. Repo: `https://github.com/tylergibbs1/surge`
3. Dockerfile path: `runpod/Dockerfile`
4. Start command: `python -u runpod/handler.py`
5. GPU: any (T4 / A10 / CPU all work). Start with **CPU** — Chronos-2 is
   small and the cold-start math favours CPU for a low-traffic endpoint.
6. Env vars (optional):
   - `SURGE_MODEL_HF_ID` — override the HF model ID
7. Hit **Deploy**.

## Invoke

RunPod gives you an endpoint ID after deploy. Then:

```bash
curl -X POST \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "content-type: application/json" \
  -d '{"input":{"ba":"PJM","horizon":24}}'
```

## Wire the Next.js playground

In `web/playground/.env.local`:

```
SURGE_API_URL=https://api.runpod.ai/v2/<endpoint-id>
RUNPOD_API_KEY=...
```

Then update `app/api/forecast/route.ts` to POST to `/runsync` with the
Bearer token instead of GETing `/forecast/{ba}`. One-line change.

## Cost

- CPU worker: ~$0.10/hour while running, free when idle
- T4 worker: ~$0.40/hour while running
- Cold start: 3–15 seconds (model + data load)
- Warm request: ~200 ms on GPU, 1–2 s on CPU

For the expected Surge traffic (dashboard visitors + occasional API users),
CPU serverless is fine.

## Local test before deploy

```bash
pip install runpod
python runpod/handler.py
# then in another shell:
curl -X POST http://127.0.0.1:8000/runsync \
  -d '{"input":{"ba":"PJM","horizon":24}}'
```
