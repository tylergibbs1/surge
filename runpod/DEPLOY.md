# Deploying Surge to RunPod serverless (CLI only, no web UI)

Fully programmatic. Two commits + two commands and you have an
always-available forecasting endpoint.

## Prerequisites

1. **GitHub repo** with Actions enabled (this one).
2. **RunPod API key** from https://runpod.io/console/user/settings.
3. Local Python env with the SDK: `pip install runpod`.

## Step 1 — Image build (automatic, via GitHub Actions)

`.github/workflows/docker.yml` builds `runpod/Dockerfile` and pushes to
`ghcr.io/<owner>/<repo>:latest` on every push to `main`. First run takes
~8 min (big PyTorch layer + Chronos-2 pre-download). Subsequent builds are
<1 min with GHA layer cache.

Enable it by ensuring your `gh` token has the `workflow` scope
(`gh auth refresh -s workflow`) and pushing the workflow file:

```bash
git add .github/workflows/docker.yml
git commit -m "ci: docker build + push"
git push
```

Verify on https://github.com/<owner>/<repo>/pkgs/container/surge — you
should see `latest` after the job finishes.

## Step 2 — Endpoint provisioning (one command)

```bash
export RUNPOD_API_KEY=...
python scripts/deploy_runpod.py
```

Output looks like:

```
[template] creating 'surge-fm-v2-template' from ghcr.io/tylergibbs1/surge:latest
[template] created id=abc123xyz
[endpoint] creating 'surge-fm-v2'
[endpoint] created id=xyz789abc
───────────────────────────────────────
  endpoint id   : xyz789abc
  runsync       : https://api.runpod.ai/v2/xyz789abc/runsync
  run (async)   : https://api.runpod.ai/v2/xyz789abc/run
───────────────────────────────────────
```

The script is **idempotent** — re-run it after each `docker.yml` rebuild
and it'll re-pin the endpoint to the latest template without duplicating.

## Step 3 — Test

```bash
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "content-type: application/json" \
  -d '{"input":{"ba":"PJM","horizon":24}}'
```

First call: ~15-30 s cold start. Subsequent calls in the same warm window
(30 s idle timeout by default): ~1-2 s each on CPU.

## Wire the Next.js playground

Set two env vars in `web/playground/.env.local`:

```
SURGE_API_URL=https://api.runpod.ai/v2/<endpoint-id>
RUNPOD_API_KEY=<your-key>
```

Then update `app/api/forecast/route.ts` — the POST body is documented in
`runpod/handler.py`. Done.

## Scaling knobs

Edit `scripts/deploy_runpod.py`:

- `workers_min=0` → scale-to-zero (free when idle); use `1` to keep one
  warm worker permanently (~$0.10/h CPU).
- `workers_max=2` → max concurrent workers; increase for burst traffic.
- `idle_timeout=30` (seconds) → how long to keep a worker warm after the
  last request.
- `GPU_IDS=AMPERE_16` → rent a T4/A4000 instead of CPU. 10× latency,
  ~4× cost per request. Worth it above ~100 req/min.
